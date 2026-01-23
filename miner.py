import os
from traceback import print_exc
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import pandas as pd
from pathlib import Path
import nova_ph2
from itertools import combinations

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening
from moleculesv2 import (
    generate_valid_random_molecules_batch,
    select_diverse_elites,
    build_component_weights,
    compute_tanimoto_similarity_to_pool,
    sample_random_valid_molecules,
    compute_maccs_entropy,
    SynthonLibrary,
    generate_molecules_from_synthon_library,
    validate_molecules,
    ComponentSynergyMatrix,
    compute_quality_score,
    select_similar_molecules,
)

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")


target_models = []
antitarget_models = []


def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


def initialize_models(config: dict):
    """Initialize separate model instances for each target and antitarget sequence."""
    global target_models, antitarget_models
    target_models = []
    antitarget_models = []
    
    for seq in config["target_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        target_models.append(wrapper)
    
    for seq in config["antitarget_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        antitarget_models.append(wrapper)


# ---------- scoring helpers (reuse pre-initialized models) ----------
def target_score_from_data(data: pd.Series):
    """Score molecules against all target models."""
    global target_models, antitarget_models
    try:
        target_scores = []
        smiles_list = data.tolist()
        for target_model in target_models:
            scores = target_model.score_molecules(smiles_list)
            # Share smiles_dict with antitarget models
            for antitarget_model in antitarget_models:
                antitarget_model.smiles_list = smiles_list
                antitarget_model.smiles_dict = target_model.smiles_dict
            
            scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
            target_scores.append(scores["target"])
        
        # Average across all targets
        target_series = pd.DataFrame(target_scores).mean(axis=0)
        return target_series
    except Exception as e:
        bt.logging.error(f"Target scoring error: {e}")
        return pd.Series(dtype=float)


def antitarget_scores():
    """Score molecules against all antitarget models."""
    global antitarget_models
    try:
        antitarget_scores = []
        for i, antitarget_model in enumerate(antitarget_models):
            antitarget_model.create_screen_loader(antitarget_model.protein_dict, antitarget_model.smiles_dict)
            antitarget_model.screen_df = virtual_screening(
                antitarget_model.screen_df, 
                antitarget_model.model, 
                antitarget_model.screen_loader,
                os.getcwd(),
                save_interpret=False,
                ligand_dict=antitarget_model.smiles_dict, 
                device=antitarget_model.device,
                save_cluster=False,
            )
            scores = antitarget_model.screen_df[['predicted_binding_affinity']]
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            antitarget_scores.append(scores[f"anti_{i}"])
        
        if not antitarget_scores:
            return pd.Series(dtype=float)
        
        # Average across antitargets
        anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
        return anti_series
    except Exception as e:
        bt.logging.error(f"Antitarget scoring error: {e}")
        return pd.Series(dtype=float)


def _cpu_random_candidates_with_similarity(
    iteration: int,
    n_samples: int,
    subnet_config: dict,
    top_pool_df: pd.DataFrame,
    avoid_inchikeys: set[str] | None = None,
    thresh: float = 0.8
) -> pd.DataFrame:
    """
    CPU-side helper to generate random molecules similar to top pool.
    Returns DataFrame with name, smiles, InChIKey.
    """
    try:
        random_df = sample_random_valid_molecules(
            n_samples=n_samples,
            subnet_config=subnet_config,
            avoid_inchikeys=avoid_inchikeys,
            focus_neighborhood_of=top_pool_df
        )
        if random_df.empty or top_pool_df.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

        sims = compute_tanimoto_similarity_to_pool(
            candidate_smiles=random_df["smiles"],
            pool_smiles=top_pool_df["smiles"],
        )
        random_df = random_df.copy()
        random_df["tanimoto_similarity"] = sims.reindex(random_df.index).fillna(0.0)
        random_df = random_df.sort_values(by="tanimoto_similarity", ascending=False)
        random_df_filtered = random_df[random_df["tanimoto_similarity"] >= thresh]
            
        if random_df_filtered.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
            
        random_df_filtered = random_df_filtered.reset_index(drop=True)
        return random_df_filtered[["name", "smiles", "InChIKey"]]
    except Exception as e:
        bt.logging.warning(f"[Miner] CPU similarity search failed: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])


def select_diverse_subset(pool, top_95_smiles, subset_size=5, entropy_threshold=0.1):
    """Select diverse subset based on entropy threshold."""
    smiles_list = pool["smiles"].tolist()
    for combination in combinations(smiles_list, subset_size):
        test_subset = top_95_smiles + list(combination)
        entropy = compute_maccs_entropy(test_subset)
        if entropy >= entropy_threshold:
            bt.logging.info(f"Entropy Threshold Met: {entropy:.4f}")
            return pool[pool["smiles"].isin(combination)]
    
    bt.logging.warning("No combination exceeded the given entropy threshold.")
    return pd.DataFrame()


def generate_multi_range_synthon_candidates(synthon_lib, top_pool, n_samples, has_very_high_score_or_late_stage, rxn_id):
    """Generate synthon candidates using multi-range strategy."""
    if has_very_high_score_or_late_stage:
        # Late stage or high score: focus heavily on top 1
        synthon_dfs = [
            generate_molecules_from_synthon_library(synthon_lib, top_pool.head(1), int(n_samples * 0.21), min_similarity=0.93, n_per_base=50),
            generate_molecules_from_synthon_library(synthon_lib, top_pool.head(5), int(n_samples * 0.07), min_similarity=0.87, n_per_base=30),
            generate_molecules_from_synthon_library(synthon_lib, top_pool.iloc[10:40] if len(top_pool) > 40 else top_pool.iloc[5:], int(n_samples * 0.21), min_similarity=0.82 if rxn_id==3 else 0.55, n_per_base=15),
            generate_molecules_from_synthon_library(synthon_lib, top_pool.head(50), int(n_samples * 0.21), min_similarity=0.40, n_per_base=20),
        ]
        bt.logging.info(f"[Miner] Multi-range: {[len(df) for df in synthon_dfs]} candidates (TOP-1, TIGHT, MEDIUM, BROAD)")
    else:
        # Normal multi-range strategy
        synthon_dfs = [
            generate_molecules_from_synthon_library(synthon_lib, top_pool.head(5), int(n_samples * 0.28), min_similarity=0.80, n_per_base=30),
            generate_molecules_from_synthon_library(synthon_lib, top_pool.iloc[10:40] if len(top_pool) > 40 else top_pool.iloc[5:], int(n_samples * 0.21), min_similarity=0.55, n_per_base=15),
            generate_molecules_from_synthon_library(synthon_lib, top_pool.head(50), int(n_samples * 0.21), min_similarity=0.40, n_per_base=20),
        ]
        bt.logging.info(f"[Miner] Multi-range: {[len(df) for df in synthon_dfs]} candidates (TIGHT, MEDIUM, BROAD)")
    
    return pd.concat(synthon_dfs, ignore_index=True).drop_duplicates(subset=["name"], keep="first")


def generate_candidates(iteration, n_samples, config, top_pool, rxn_id, elite_names, elite_frac, 
                       mutation_prob, seen_inchikeys, component_weights, synthon_lib, 
                       use_synthon_search, score_improvement_rate, no_improvement_counter, 
                       start_time, n_samples_first_iteration):
    """
    Generate molecular candidates based on current state and strategy.
    Returns DataFrame with name, smiles, InChIKey.
    """
    # Diversity restart when stuck
    if no_improvement_counter >= 5:
        bt.logging.info(f"[RESTART] Diversity restart (stuck for {no_improvement_counter} iterations)")
        restart_candidates = []
        
        # Population A: Tight exploitation if synthon available
        if use_synthon_search and synthon_lib is not None:
            try:
                pop_A = generate_molecules_from_synthon_library(
                    synthon_lib, top_pool.head(10), int(n_samples * 0.35), 
                    min_similarity=0.75, n_per_base=25
                )
                if not pop_A.empty:
                    pop_A = validate_molecules(pop_A, config)
                    restart_candidates.append(pop_A)
                    bt.logging.info(f"[RESTART] Pop A (tight): {len(pop_A)} candidates")
            except Exception as e:
                bt.logging.warning(f"[RESTART] Pop A failed: {e}")
        
        # Population B: Medium exploration
        if use_synthon_search and synthon_lib is not None and len(top_pool) >= 60:
            try:
                pop_B = generate_molecules_from_synthon_library(
                    synthon_lib, top_pool.iloc[20:60], int(n_samples * 0.25), 
                    min_similarity=0.50, n_per_base=15
                )
                if not pop_B.empty:
                    pop_B = validate_molecules(pop_B, config)
                    restart_candidates.append(pop_B)
                    bt.logging.info(f"[RESTART] Pop B (medium): {len(pop_B)} candidates")
            except Exception as e:
                bt.logging.warning(f"[RESTART] Pop B failed: {e}")
        
        # Population C: Pure random exploration
        try:
            pop_C = generate_valid_random_molecules_batch(
                rxn_id, n_samples=int(n_samples * 1.5), db_path=DB_PATH, 
                subnet_config=config, batch_size=400, elite_names=None, 
                elite_frac=0.0, mutation_prob=1.0, avoid_inchikeys=seen_inchikeys, 
                component_weights=None
            )
            if not pop_C.empty:
                restart_candidates.append(pop_C)
                bt.logging.info(f"[RESTART] Pop C (random): {len(pop_C)} candidates")
        except Exception as e:
            bt.logging.warning(f"[RESTART] Pop C failed: {e}")
        
        if restart_candidates:
            data = pd.concat(restart_candidates, ignore_index=True).drop_duplicates(subset=["name"], keep="first")
            bt.logging.info(f"[RESTART] Total: {len(data)} candidates")
            return data
        else:
            bt.logging.warning(f"[RESTART] All populations failed, using fallback")
            return generate_valid_random_molecules_batch(
                rxn_id, n_samples=n_samples * 2, db_path=DB_PATH, subnet_config=config,
                batch_size=400, elite_names=None, elite_frac=0.0, mutation_prob=1.0,
                avoid_inchikeys=seen_inchikeys, component_weights=None
            )
    
    # First iteration: broad random sampling
    elif iteration == 1:
        bt.logging.info(f"[Miner] Iteration {iteration}: Initial broad random sampling")
        return generate_valid_random_molecules_batch(
            rxn_id, n_samples=n_samples_first_iteration, db_path=DB_PATH, 
            subnet_config=config, batch_size=400, elite_names=None, 
            elite_frac=0.0, mutation_prob=1.0, avoid_inchikeys=seen_inchikeys, 
            component_weights=None
        )
    
    # Smart synthon search (iteration > 2)
    elif use_synthon_search and iteration > 2 and not top_pool.empty:
        bt.logging.info(f"[Miner] Iteration {iteration}: Smart synthon similarity search")
        current_max_score = top_pool['score'].max() if not top_pool.empty else None
        time_elapsed = time.time() - start_time
        has_very_high_score = current_max_score is not None and current_max_score > 0.015
        is_very_late_stage = time_elapsed > 1500
        
        # Adaptive strategy based on improvement rate
        if score_improvement_rate > 0.05:
            # High improvement: tight similarity
            sim_threshold, n_per_base, n_seeds, synthon_ratio = 0.75, 15, 20, 0.75
            bt.logging.info(f"[Miner] High improvement ({score_improvement_rate:.4f}), tight similarity (0.75)")
        elif score_improvement_rate > 0.02:
            # Good improvement: medium-tight similarity
            sim_threshold, n_per_base, n_seeds, synthon_ratio = 0.70, 18, 25, 0.75
            bt.logging.info(f"[Miner] Good improvement ({score_improvement_rate:.4f}), medium-tight similarity (0.70)")
        elif score_improvement_rate > 0.005:
            # Moderate improvement: medium similarity
            sim_threshold, n_per_base, n_seeds, synthon_ratio = 0.65, 20, 30, 0.70
            bt.logging.info(f"[Miner] Moderate improvement ({score_improvement_rate:.4f}), medium similarity (0.65)")
        else:
            # Low improvement: multi-range strategy
            bt.logging.info(f"[Miner] Low improvement ({score_improvement_rate:.4f}), multi-range strategy")
            synthon_df = generate_multi_range_synthon_candidates(
                synthon_lib, top_pool, n_samples, has_very_high_score or is_very_late_stage, rxn_id
            )
            
            if not synthon_df.empty:
                synthon_df = validate_molecules(synthon_df, config)
                bt.logging.info(f"[Miner] {len(synthon_df)} multi-range synthon candidates validated")
            
            n_traditional = max(0, n_samples - len(synthon_df))
            traditional_df = generate_valid_random_molecules_batch(
                rxn_id, n_samples=n_traditional, db_path=DB_PATH, subnet_config=config,
                batch_size=400, elite_names=elite_names, elite_frac=elite_frac,
                mutation_prob=mutation_prob, avoid_inchikeys=seen_inchikeys,
                component_weights=component_weights
            ) if n_traditional > 0 else pd.DataFrame(columns=["name", "smiles", "InChIKey"])
            
            data = pd.concat([synthon_df, traditional_df], ignore_index=True).drop_duplicates(subset=["name"], keep="first")
            bt.logging.info(f"[Miner] Combined: {len(data)} total ({len(synthon_df)} synthon + {len(traditional_df)} GA)")
            return data
        
        # Single-range synthon strategy (for higher improvement rates)
        n_synthon = int(n_samples * synthon_ratio)
        synthon_df = generate_molecules_from_synthon_library(
            synthon_lib, top_pool.head(n_seeds), n_synthon, 
            min_similarity=sim_threshold, n_per_base=n_per_base
        )
        bt.logging.info(f"[Miner] Generated {len(synthon_df)} synthon candidates")
        
        if not synthon_df.empty:
            synthon_df = validate_molecules(synthon_df, config)
            bt.logging.info(f"[Miner] {len(synthon_df)} synthon candidates validated")
        
        n_traditional = max(0, n_samples - len(synthon_df))
        traditional_df = generate_valid_random_molecules_batch(
            rxn_id, n_samples=n_traditional, db_path=DB_PATH, subnet_config=config,
            batch_size=300, elite_names=elite_names, elite_frac=elite_frac,
            mutation_prob=mutation_prob, avoid_inchikeys=seen_inchikeys,
            component_weights=component_weights
        ) if n_traditional > 0 else pd.DataFrame(columns=["name", "smiles", "InChIKey"])
        
        data = pd.concat([synthon_df, traditional_df], ignore_index=True).drop_duplicates(subset=["name"], keep="first")
        bt.logging.info(f"[Miner] Combined: {len(data)} total ({len(synthon_df)} synthon + {len(traditional_df)} GA)")
        return data
    
    # Standard genetic algorithm (no improvement for <3 iterations)
    elif no_improvement_counter < 3:
        bt.logging.info(f"[Miner] Iteration {iteration}: Standard genetic algorithm")
        return generate_valid_random_molecules_batch(
            rxn_id, n_samples=n_samples, db_path=DB_PATH, subnet_config=config,
            batch_size=400, elite_names=elite_names, elite_frac=elite_frac,
            mutation_prob=mutation_prob, avoid_inchikeys=seen_inchikeys,
            component_weights=component_weights
        )
    
    # Similarity exploration (stuck for 3-4 iterations)
    else:
        bt.logging.info(f"[Miner] Iteration {iteration}: Similarity exploration (stuck={no_improvement_counter})")
        return _cpu_random_candidates_with_similarity(
            iteration, 30, config, top_pool.head(50)[["name", "smiles", "InChIKey"]],
            seen_inchikeys, 0.65
        )


def apply_quality_filter(data, n_samples, component_weights, synergy_matrix):
    """Apply quality-based filtering to candidates."""
    try:
        original_count = len(data)
        data = data.copy()
        data['quality'] = data['name'].apply(
            lambda x: compute_quality_score(x, component_weights, synergy_matrix)
        )
        data = data.sort_values('quality', ascending=False)
        
        if len(data) > n_samples * 1.2:
            top_n = int(len(data) * 0.7)
            random_n = int(len(data) * 0.5)
            top_candidates = data.head(top_n)
            remaining = data.iloc[top_n:]
            if len(remaining) > 0:
                random_candidates = remaining.sample(n=min(random_n, len(remaining)))
                data = pd.concat([top_candidates, random_candidates])
            else:
                data = top_candidates
            
            bt.logging.info(f"[QUALITY] Filtered to {len(data)} candidates ({top_n} top + {len(data)-top_n} random)")
        
        data = data.drop(columns=['quality']).reset_index(drop=True)
        
        if len(data) < original_count:
            bt.logging.info(f"[QUALITY] Filtered {original_count - len(data)} low-quality candidates")
        
        return data
    except Exception as e:
        bt.logging.warning(f"[QUALITY] Quality filtering failed: {e}, continuing without filtering")
        return data


def handle_cpu_futures(cpu_futures, seed_df):
    """Process completed CPU similarity search futures."""
    for cpu_future, strategy_name in cpu_futures:
        try:
            cpu_df = cpu_future.result(timeout=0)
            if not cpu_df.empty:
                seed_df = pd.concat([seed_df, cpu_df], ignore_index=True) if not seed_df.empty else cpu_df.copy()
                bt.logging.info(f"[Miner] CPU similarity ({strategy_name}) found {len(cpu_df)} candidates")
        except TimeoutError:
            pass
        except Exception as e:
            bt.logging.warning(f"[Miner] CPU similarity ({strategy_name}) failed: {e}")
    
    if not seed_df.empty:
        seed_df = seed_df.drop_duplicates(subset=["InChIKey"], keep="first")
    
    return seed_df


def adjust_ga_parameters(dup_ratio, mutation_prob, elite_frac, top_pool, iteration):
    """Adjust genetic algorithm parameters based on duplication rate."""
    if dup_ratio > 0.7:
        mutation_prob = min(0.9, mutation_prob * 1.5)
        elite_frac = max(0.15, elite_frac * 0.7)
        bt.logging.warning(f"[Miner] SEVERE duplication ({dup_ratio:.2%})! mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
    elif dup_ratio > 0.5:
        mutation_prob = min(0.7, mutation_prob * 1.3)
        elite_frac = max(0.2, elite_frac * 0.8)
        bt.logging.warning(f"[Miner] High duplication ({dup_ratio:.2%}), mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
    elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
        mutation_prob = max(0.05, mutation_prob * 0.95)
        elite_frac = min(0.85, elite_frac * 1.05)
    
    return mutation_prob, elite_frac


def finalize_pool_with_entropy(top_pool, config):
    """Ensure final pool meets entropy requirements."""
    entropy = compute_maccs_entropy(top_pool.iloc[:config["num_molecules"]]['smiles'].to_list())
    if entropy > config['entropy_min_threshold']:
        top_pool = top_pool.head(config["num_molecules"])
        bt.logging.info(f"[Miner] Sufficient Entropy = {entropy:.4f}")
    else:
        try:
            top_95 = top_pool.iloc[:95]
            remaining_pool = top_pool.iloc[95:]
            additional_5 = select_diverse_subset(
                remaining_pool, top_95["smiles"].tolist(), 
                subset_size=5, entropy_threshold=config['entropy_min_threshold']
            )
            if not additional_5.empty:
                top_pool = pd.concat([top_95, additional_5]).reset_index(drop=True)
                entropy = compute_maccs_entropy(top_pool['smiles'].to_list())
                bt.logging.info(f"[Miner] Adjusted Entropy = {entropy:.4f}")
            else:
                top_pool = top_pool.head(config["num_molecules"])
        except Exception as e:
            bt.logging.warning(f"[Miner] Entropy handling failed: {e}")
    
    return top_pool


def main(config: dict):
    """Main optimization loop."""
    # Initialize state
    base_n_samples = 1200 if config["allowed_reaction"] in ["rxn:5", "rxn:4"] else 1600 if config["allowed_reaction"] == "rxn:3" else 1300
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    iteration = 0
    mutation_prob = 0.5
    elite_frac = 0.4
    seen_inchikeys = set()
    seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    sim_seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    start = time.time()
    prev_avg_score = None
    score_improvement_rate = 0.0
    no_improvement_counter = 0
    synthon_lib = None
    use_synthon_search = False
    synergy_matrix = ComponentSynergyMatrix()
    n_samples_first_iteration = base_n_samples * 6 if config["allowed_reaction"] != "rxn:5" else base_n_samples * 3
    total_gpu_time = 0.0
    total_gpu_run = 0
    total_run_time = 1800
    with ProcessPoolExecutor(max_workers=5) as cpu_executor:
        while time.time() - start < total_run_time:
            iteration += 1
            iter_start_time = time.time()
            remaining_time = total_run_time - (time.time() - start)
            
            if remaining_time > 300:
                if remaining_time > 1500:
                    n_samples = base_n_samples
                elif remaining_time > 900:
                    n_samples = int(base_n_samples * 1.15)
                elif remaining_time > 600:
                    n_samples = int(base_n_samples)
                else:
                    n_samples = int(base_n_samples * 0.85)
                
                if iteration == 2 and not top_pool.empty and synthon_lib is None:
                    try:
                        bt.logging.info("[Miner] Building synthon library...")
                        synthon_lib_start = time.time()
                        synthon_lib = SynthonLibrary(DB_PATH, rxn_id)
                        use_synthon_search = True
                        bt.logging.info(f"[Miner] Synthon library ready in {time.time() - synthon_lib_start:.2f}s")
                    except Exception as e:
                        bt.logging.warning(f"[Miner] Could not build synthon library: {e}")
                        use_synthon_search = False
                
                # Prepare elite selection and component weights
                component_weights = build_component_weights(top_pool, rxn_id) if not top_pool.empty else None
                elite_df = select_diverse_elites(top_pool, min(150, len(top_pool))) if not top_pool.empty else pd.DataFrame()
                elite_names = elite_df["name"].tolist() if not elite_df.empty else None
                
                # Generate candidates
                data = generate_candidates(
                    iteration, n_samples, config, top_pool, rxn_id, elite_names, 
                    elite_frac, mutation_prob, seen_inchikeys, component_weights, 
                    synthon_lib, use_synthon_search, score_improvement_rate, 
                    no_improvement_counter, start, n_samples_first_iteration
                )
                
                gen_time = time.time() - iter_start_time
                bt.logging.info(f"[Miner] Iteration {iteration}: {len(data)} samples generated in {gen_time:.2f}s")
                
                if data.empty:
                    bt.logging.warning(f"[Miner] Iteration {iteration}: No valid molecules produced")
                    continue
                
                # Merge with seed data
                if not seed_df.empty:
                    data = pd.concat([data, seed_df], ignore_index=True).drop_duplicates(subset=["InChIKey"], keep="first")
                    seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
                
                # Remove duplicates and adjust GA parameters
                try:
                    filtered_data = data[~data["InChIKey"].isin(seen_inchikeys)]
                    if len(filtered_data) < len(data):
                        bt.logging.warning(f"[Miner] Iteration {iteration}: {len(data) - len(filtered_data)} molecules were previously seen")
                    
                    dup_ratio = (len(data) - len(filtered_data)) / max(1, len(data))
                    mutation_prob, elite_frac = adjust_ga_parameters(dup_ratio, mutation_prob, elite_frac, top_pool, iteration)
                    data = filtered_data
                except Exception as e:
                    bt.logging.warning(f"[Miner] Pre-score deduplication failed: {e}")
                
                data = data.reset_index(drop=True)
                
                if len(data) == 0:
                    bt.logging.error(f"[Miner] Iteration {iteration}: No molecules to score")
                    continue
                
                # Launch CPU similarity searches (async)
                cpu_futures = []
                sim_cpu_futures = []

                if not top_pool.empty and iteration > 3 and score_improvement_rate < 0.01:
                    cpu_futures.append((
                        cpu_executor.submit(
                            _cpu_random_candidates_with_similarity, iteration, 40, config,
                            top_pool.head(5)[["name", "smiles", "InChIKey"]], seen_inchikeys, 0.80
                        ), "tight-top5"
                    ))
                    cpu_futures.append((
                        cpu_executor.submit(
                            _cpu_random_candidates_with_similarity, iteration, 30, config,
                            top_pool.head(20)[["name", "smiles", "InChIKey"]], seen_inchikeys, 0.65
                        ), "medium-top20"
                    ))
                
                if remaining_time <600:
                    for i in range(10):
                        sim_cpu_futures.append((
                            cpu_executor.submit(
                                select_similar_molecules, iteration, i, top_pool.iloc[i],
                                1000, 0.95, DB_PATH, rxn_id, config
                            ), f"similar-top{i+1}"
                        ))
                # Apply quality filtering
                if iteration > 3 and component_weights and len(data) > 0:
                    bt.logging.info(f"[QUALITY] Pre-ranking {len(data)} candidates")
                    data = apply_quality_filter(data, n_samples, component_weights, synergy_matrix)
                
                # GPU scoring
                gpu_start_time = time.time()
                data["Target"] = target_score_from_data(data["smiles"])
                data["Anti"] = antitarget_scores()
                data["score"] = data["Target"] - (config["antitarget_weight"] * data["Anti"])
                
                if data["score"].isna().all():
                    bt.logging.error(f"[Miner] Iteration {iteration}: Scoring failed (all NaN)")
                    continue
                
                for _, row in data.iterrows():
                    synergy_matrix.update(row['name'], row['score'])
                
                gpu_time = time.time() - gpu_start_time
                total_gpu_time += gpu_time
                total_gpu_run += len(data)
                bt.logging.info(f"[Miner] Iteration {iteration}: GPU scoring time {gpu_time:.2f}s")
                
                if cpu_futures:
                    seed_df = handle_cpu_futures(cpu_futures, seed_df)

                if sim_cpu_futures:
                    sim_seed_df = handle_cpu_futures(sim_cpu_futures, sim_seed_df)
                
                seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])
                total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
                
                if not total_data.empty:
                    top_pool = pd.concat([top_pool, total_data], ignore_index=True)
                    top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                    top_pool = top_pool.sort_values(by="score", ascending=False)
                else:
                    bt.logging.warning(f"[Miner] Iteration {iteration}: No valid scored data")
            
            else:
                sim_cpu_futures = []

                if not sim_seed_df.empty:
                    sim_seed_df = sim_seed_df[~sim_seed_df["InChIKey"].isin(seen_inchikeys)]

                total_run = int(len(sim_seed_df)*0.9*(total_gpu_time/total_gpu_run)//1)

                if len(sim_seed_df)*(total_gpu_time/total_gpu_run) < 10:
                    total_run = 10
                bt.logging.info(f"[Miner] Total run: {total_run} for Iteration {iteration}")

                for i in range(10):
                    sim_cpu_futures.append((
                        cpu_executor.submit(
                            select_similar_molecules, iteration, i, top_pool.iloc[i],
                            1000 if total_run>17 else 1600, 0.95, DB_PATH, rxn_id, config
                        ), f"similar-top{i+1}"
                    ))

                gpu_start_time = time.time()

                if not sim_seed_df.empty:
                    sim_seed_df['Target'] = target_score_from_data(sim_seed_df['smiles'])
                    sim_seed_df['Anti'] = antitarget_scores()
                    sim_seed_df['score'] = sim_seed_df['Target'] - (config["antitarget_weight"] * sim_seed_df['Anti'])
                    sim_seed_df = sim_seed_df.sort_values(by="score", ascending=False).head(config["num_molecules"])
                    seen_inchikeys.update([k for k in sim_seed_df["InChIKey"].tolist() if k])

                    top_pool = pd.concat([top_pool, sim_seed_df], ignore_index=True)
                    top_pool = top_pool.sort_values(by="score", ascending=False)
                    top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                    sim_seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])

                gpu_time = time.time() - gpu_start_time
                if gpu_time < 10:
                    time.sleep(10 - gpu_time)
                    bt.logging.info(f"[Miner] Iteration {iteration}: GPU scoring time {gpu_time:.2f}s, sleeping for {10 - gpu_time:.2f}s")
                total_gpu_time += gpu_time
                total_gpu_run += len(sim_seed_df)
                bt.logging.info(f"[Miner] Iteration {iteration}: GPU scoring time {gpu_time:.2f}s")

                if sim_cpu_futures:
                    sim_seed_df = handle_cpu_futures(sim_cpu_futures, sim_seed_df)
            
            if remaining_time <= 60:
                top_pool = finalize_pool_with_entropy(top_pool, config)
            else:
                top_pool = top_pool.head(config["num_molecules"])
            
            current_avg_score = top_pool['score'].mean() if not top_pool.empty else None

            if current_avg_score is not None and prev_avg_score is not None:
                score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
            prev_avg_score = current_avg_score
            
            if score_improvement_rate == 0.0:
                no_improvement_counter += 1
            
            iter_total_time = time.time() - iter_start_time
            total_time = time.time() - start
            bt.logging.info(
                f"Iteration {iteration} || Time: {iter_total_time:.2f}s | Total: {total_time:.2f}s | "
                f"Avg: {top_pool['score'].mean():.4f} | Max: {top_pool['score'].max():.4f} | "
                f"Min: {top_pool['score'].min():.4f} | Elite: {elite_frac:.2f} | "
                f"Mut: {mutation_prob:.2f} | Improve: {score_improvement_rate:.4f}"
            )
            
            top_entries = {"molecules": top_pool["name"].tolist()}
            with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    config = get_config()
    start_time = time.time()
    initialize_models(config)
    bt.logging.info(f"Model initialization took {time.time() - start_time:.2f}s")
    main(config)
