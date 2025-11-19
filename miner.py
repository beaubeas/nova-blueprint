import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import time
import torch
import bittensor as bt
import pandas as pd
from pathlib import Path
import nova_ph2
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
# import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening
from molecules import generate_valid_random_molecules_batch

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
    """Score molecules against all target models. target_sequence parameter kept for compatibility but not used."""
    global target_models, antitarget_models
    try:
        target_scores = []
        smiles_list = data.tolist()
        for target_model in target_models:
            scores = target_model.score_molecules(smiles_list)
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
    """Score molecules against all antitarget models. antitarget_sequence parameter kept for compatibility but not used."""
    global antitarget_models
    try:
        antitarget_scores = []
        for i, antitarget_model in enumerate(antitarget_models):
            torch.cuda.empty_cache()
            antitarget_model.create_screen_loader(antitarget_model.protein_dict, antitarget_model.smiles_dict)
            antitarget_model.screen_df = virtual_screening(antitarget_model.screen_df, 
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
        
        # average across antitargets
        anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
        return anti_series
    except Exception as e:
        bt.logging.error(f"Antitarget scoring error: {e}")
        return pd.Series(dtype=float)


# def plot_scores_history(iterations, avg_scores, max_scores, min_scores):
#     """Plot and save the score history across iterations."""
#     try:
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
#         ax1.plot(iterations, avg_scores, 'b-o', linewidth=2, markersize=6, label='Average Score')
#         ax1.fill_between(iterations, min_scores, max_scores, alpha=0.3, color='blue', label='Score Range')
#         ax1.set_xlabel('Iteration', fontsize=12)
#         ax1.set_ylabel('Average Score', fontsize=12)
#         ax1.set_title('Average Scores Over Iterations', fontsize=14, fontweight='bold')
#         ax1.grid(True, alpha=0.3)
#         ax1.legend()
#         ax1.set_xlim(left=0)
        
#         if len(iterations) > 1:
#             window = min(5, len(avg_scores))
#             if window > 1:
#                 moving_avg = pd.Series(avg_scores).rolling(window=window, min_periods=1).mean()
#                 ax2.plot(iterations, moving_avg, 'g-', linewidth=2, label=f'Moving Average (window={window})')
#             ax2.plot(iterations, avg_scores, 'b-o', linewidth=1, markersize=4, alpha=0.5, label='Average Score')
#             ax2.set_xlabel('Iteration', fontsize=12)
#             ax2.set_ylabel('Score', fontsize=12)
#             ax2.set_title('Score Trend (with Moving Average)', fontsize=14, fontweight='bold')
#             ax2.grid(True, alpha=0.3)
#             ax2.legend()
#             ax2.set_xlim(left=0)
        
#         plt.tight_layout()
        
#         plt.savefig('score_history.png', dpi=150, bbox_inches='tight')
#         plt.close()
        
#         bt.logging.info(f"[Miner] Score history plot saved")
#     except Exception as e:
#         bt.logging.error(f"Error plotting scores: {e}")
#         plt.close()

def main(config: dict):
    n_samples = config["num_molecules"] * 5
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.25
    seen_inchikeys = set()
    start = time.time()
    neighborhood_limit = 20
    
    # Track scores for visualization
    score_history = {
        'iterations': [],
        'avg_scores': [],
        'max_scores': [],
        'min_scores': [],
        'avg_target_scores': [],
        'avg_antitarget_scores': []
    }

    n_samples_first_iteration = n_samples if config["allowed_reaction"] == "rxn:5" else n_samples*4
    while time.time() - start < 1800:
        iteration += 1
        start_time = time.time()
        elite_names = top_pool['name'].tolist() if not top_pool.empty else []
        data = generate_valid_random_molecules_batch(rxn_id, n_samples=n_samples_first_iteration if iteration == 1 else n_samples, db_path=DB_PATH, subnet_config=config, batch_size=500, elite_names=elite_names, 
                                                     elite_frac = elite_frac, mutation_prob=mutation_prob, avoid_inchikeys=seen_inchikeys, neighborhood_limit=neighborhood_limit)
        
        bt.logging.info(f"[Miner] Iteration {iteration}: {len(data)} Samples Generated within {round(time.time() - start_time,2)}")
        
        if data.empty:
            bt.logging.warning(f"[Miner] Iteration {iteration}: No valid molecules produced; continuing")
            continue

        try:
            filterd_data = data[~data['InChIKey'].isin(seen_inchikeys)]
            if len(filterd_data) < len(data):
                bt.logging.warning(f"[Miner] Iteration {iteration}: {len(data) - len(filterd_data)} molecules were previously seen; continuing with unseen only")

            dup_ratio = (len(data) - len(filterd_data)) / max(1, len(data))
            if dup_ratio > 0.6:
                mutation_prob = min(0.5, mutation_prob * 1.5)
                elite_frac = max(0.2, elite_frac * 0.8)
            elif dup_ratio < 0.2 and not top_pool.empty:
                mutation_prob = max(0.05, mutation_prob * 0.9)
                elite_frac = min(0.8, elite_frac * 1.1)

            data = filterd_data

        except Exception as e:
            bt.logging.warning(f"[Miner] Pre-score deduplication failed; proceeding unfiltered: {e}")

        data = data.reset_index(drop=True)
        data['Target'] = target_score_from_data(data['smiles'])
        data['Anti'] = antitarget_scores()
        data['score'] = data['Target'] - (config['antitarget_weight'] * data['Anti'])
        bt.logging.info(f"[Miner] Iteration {iteration}: Inference finished within {round(time.time() - start_time,2)}")
        seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])
        # Keep Target and Anti columns for statistics
        total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
        top_pool = pd.concat([top_pool, total_data])
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.sort_values(by="score", ascending=False)
        top_pool = top_pool.head(config["num_molecules"])
        
        # Calculate and log statistics
        avg_score = top_pool['score'].mean()
        max_score = top_pool['score'].max()
        min_score = top_pool['score'].min()
        avg_target = top_pool['Target'].mean() if 'Target' in top_pool.columns else 0
        avg_antitarget = top_pool['Anti'].mean() if 'Anti' in top_pool.columns else 0
        
        bt.logging.info(f"[Miner] Iteration {iteration}: Average top score: {avg_score:.4f}")
        bt.logging.info(f"[Miner] Iteration {iteration}: Max score: {max_score:.4f}, Min score: {min_score:.4f}")
        bt.logging.info(f"[Miner] Iteration {iteration}: Finished within {round(time.time() - start_time,2)}")
        
        # Track scores for visualization
        score_history['iterations'].append(iteration)
        score_history['avg_scores'].append(avg_score)
        score_history['max_scores'].append(max_score)
        score_history['min_scores'].append(min_score)
        score_history['avg_target_scores'].append(avg_target)
        score_history['avg_antitarget_scores'].append(avg_antitarget)
        
        # plot_scores_history(
        #     score_history['iterations'],
        #     score_history['avg_scores'],
        #     score_history['max_scores'],
        #     score_history['min_scores']
        # )
        
        top_entries = {"molecules": top_pool["name"].tolist()}
        with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    
    config = get_config()
    initialize_models(config)
    main(config)
