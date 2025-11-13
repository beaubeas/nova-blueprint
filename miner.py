import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import time

import bittensor as bt
import pandas as pd
from pathlib import Path
import nova_ph2
from typing import List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from moelcules import generate_valid_random_molecules_batch

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")


psichic_model = PsichicWrapper()

def get_config(input_file: os.path = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


# ---------- scoring helpers (assume psichic_model is a global in the child process) ----------
def target_scores_from_data(data: pd.Series, target_sequence: List[str]):
    global psichic_model
    try:
        # ensure model and protein are initialized for this process
        psichic_model.initialize_model(target_sequence[0])
        bt.logging.info("* Target Protein *")
        scores = psichic_model.score_molecules(data.tolist())
        scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
        return scores["target"]
    except Exception as e:
        bt.logging.error(f"Target scoring error: {e}")
        return pd.Series(dtype=float)


def antitarget_scores_from_data(data: pd.Series, antitarget_sequence: List[str]):
    global psichic_model
    try:
        antitarget_scores = []
        for i, seq in enumerate(antitarget_sequence):
            psichic_model.initialize_model(seq)
            bt.logging.info(f"* Antitarget Protein ({i + 1}) *")
            scores = psichic_model.score_molecules(data.tolist())
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            antitarget_scores.append(scores[f"anti_{i}"])
        # average across antitargets
        anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
        return anti_series
    except Exception as e:
        bt.logging.error(f"Antitarget scoring error: {e}")
        return pd.Series(dtype=float)

def main(config: dict):
    n_samples = config["num_molecules"] * 5
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    iteration = 0
    mutation_prob = 0.1
    elite_frac = 0.5
    seen_inchikeys = set()
    start = time.time()
    
    while True:
        if time.time() - start >=1800:
            break
        iteration += 1
        start_time = time.time()
        bt.logging.info(f"[Miner] Iteration {iteration}: sampling {n_samples} molecules")
        data = generate_valid_random_molecules_batch(rxn_id, n_samples, DB_PATH, config, batch_size=n_samples, elite_names = top_pool['name'].tolist(), 
                                                     elite_frac = elite_frac, mutation_prob = mutation_prob, avoid_inchikeys= seen_inchikeys)
        
        bt.logging.info(f"[Miner] Generation finished within {round(time.time() - start_time,2)}")
        
        if data.empty:
            bt.logging.warning("[Miner] No valid molecules produced; continuing")
            continue

        try:
            filterd_data = data[~data['smiles'].isin(seen_inchikeys)]

            if len(filterd_data) < len(data):
                bt.logging.warning(f"{len(data) - len(filterd_data)} molecules were previously seen; continuing with unseen only")

            dup_ratio = (len(data) - len(filterd_data)) / max(1, len(data))
            if dup_ratio > 0.62:
                mutation_prob = min(0.5, mutation_prob * 1.5)
                elite_frac = max(0.2, elite_frac * 0.8)
            elif dup_ratio < 0.22 and not top_pool.empty:
                mutation_prob = max(0.05, mutation_prob * 0.9)
                elite_frac = min(0.8, elite_frac * 1.1)

            data = filterd_data

        except Exception as e:
            bt.logging.warning(f"[Miner] Pre-score deduplication failed; proceeding unfiltered: {e}")

        bt.logging.info(f"[Miner] Generation finished within {round(time.time() - start_time,2)}")
        data = data.reset_index(drop=True)
        data['Target'] = target_scores_from_data(data['smiles'], config['target_sequences'])
        data = data.sort_values(by='Target', ascending=False)
        if data['Target'].max() - data['Target'].min() > 3.0:
            data = data.iloc[:400]
        elif data['Target'].max() - data['Target'].min()>1.5:
            data = data.iloc[:450]
        bt.logging.info(f"[Miner] After target scoring, {len(data)} molecules selected.")
        data['Anti'] = antitarget_scores_from_data(data['smiles'], config['antitarget_sequences'])
        data['score'] = data['Target'] - (config['antitarget_weight'] * data['Anti'])

        seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])

        total_data = data[["name", "smiles", "InChIKey", "score"]]
        top_pool = pd.concat([top_pool, total_data])
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.sort_values(by="score", ascending=False)
        top_pool = top_pool.head(config["num_molecules"])
        bt.logging.info(f"[Miner] Top pool now has {len(top_pool)} molecules after merging")
        bt.logging.info(f"[Miner] Top molecules: {top_pool[['name', 'score']]}")
        bt.logging.info(f"[Miner] Average top score: {top_pool['score'].mean()}")
        bt.logging.info(f"[Miner] Iteration finished within {round(time.time() - start_time,2)}")
        # format to accepted format
        top_entries = {"molecules": top_pool["name"].tolist()}
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    
    config = get_config()
    main(config)
