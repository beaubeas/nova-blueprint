import os
import json
import bittensor as bt
import pandas as pd
from pathlib import Path
from validator.validity import generate_valid_random_molecules_batch
from validator.scoring import target_scores_from_data, antitarget_scores_from_data

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "combinatorial_db", "molecules.sqlite")

INPUT_PATH = os.path.join(SCRIPT_DIR, "input.json")
OUT_PATH = os.path.join(SCRIPT_DIR, "output.json")
def get_config(input_file: os.path):
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}
    return config

def iterative_sampling_loop(
    db_path: str,
    output_path: str,
    config: dict
) -> None:
    
    n_samples = config["num_molecules"] * 5  # Sample 50x the number of molecules needed to ensure diversity
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    iteration = 0
    while True:
        data = generate_valid_random_molecules_batch(
        rxn_id, n_samples, db_path, config, batch_size=n_samples, seed=iteration + 42
        )

        data = data.reset_index(drop=True)
        data['Target'] = target_scores_from_data(data['smiles'], config['target_sequences'])
        data = data.sort_values(by='Target', ascending=False)
        max_value = data['Target'].max()
        if max_value > 7.0:
            data = data[data['Target']>7.0]
            bt.logging.info(f"[Miner] After target scoring, {len(data)} molecules remain with Target > 7.0.")

        else:
            data = data.iloc[:10]
            bt.logging.info(f"[Miner] After target scoring, {len(data)} molecules selected.")

        if data.empty:
            bt.logging.warning(f"[Miner] No molecules passed the target score threshold. Skipping iteration {iteration}.")
            iteration += 1
            continue

        data['Anti'] = antitarget_scores_from_data(data['smiles'], config['antitarget_sequences'])
        data['score'] = data['Target'] - (config['antitarget_weight'] * data['Anti'])
        data = data[data['score']>0]
        total_data = data[["name", "smiles", "InChIKey", "score"]]
        # Merge, deduplicate, sort and take top x
        top_pool = pd.concat([top_pool, total_data])
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.sort_values(by="score", ascending=False)
        top_pool = top_pool.head(config["num_molecules"])
        bt.logging.info(f"[Miner] Top pool now has {len(top_pool)} molecules after merging, Average top score: {top_pool['score'].mean()}")
        # format to accepted format
        top_entries = {"molecules": top_pool["name"].tolist()}

        # write to file
        with open(output_path, "w") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)
        iteration += 1

def main(config: dict):
    iterative_sampling_loop(
        db_path=DB_PATH,
        output_path= OUT_PATH,
        config=config,
    )
 

if __name__ == "__main__":
    config = get_config(INPUT_PATH)
    main(config)
