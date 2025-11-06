import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import json
import bittensor as bt
import pandas as pd
from pathlib import Path
import time

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(PARENT_DIR)
# Ensure the repo's libs directory is on sys.path so the local `nova_ph2` package can be imported.
REPO_ROOT = os.path.abspath(os.path.join(PARENT_DIR, "."))

from validator.validity import validate_molecules, generate_inchikey
from validator.scoring import target_scores_from_data, antitarget_scores_from_data


DB_PATH = str(Path( "combinatorial_db/molecules.sqlite"))

def get_config(input_file: os.path = "input.json"):
    """
    Get config from input file
    """
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}
    return config

def iterative_sampling_loop(
    db_path: str,
    output_path: str,
    config: dict
) -> None:
    """
    Infinite loop, runs until orchestrator kills it:
      1) Sample n molecules
      2) Score them
      3) Merge with previous top x, deduplicate, sort, select top x
      4) Write top x to file (overwrite) each iteration
    """
    n_samples = config["num_molecules"] * 5  # Sample 50x the number of molecules needed to ensure diversity
    in_data = pd.read_csv('data.csv')

    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    iteration = 0
    while True:
        sampler_data = in_data.sample(n_samples, random_state=iteration + 42)
        data = validate_molecules(sampler_data, config)

        bt.logging.info(f"[Miner] After validation, {len(data)} valid molecules remain out of {n_samples} sampled.")
        if len(data) == 0:
            bt.logging.warning(f"[Miner] No valid molecules found after validation. Skipping iteration {iteration}.")
            iteration += 1
            continue
        else:
            data = data.reset_index(drop=True)
            data['Target'] = target_scores_from_data(data['smiles'], config['target_sequences'])
            print(data)
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
            data['InChIKey'] = data['smiles'].apply(lambda x: generate_inchikey(x))
            total_data = data[["name", "smiles", "InChIKey", "score"]]
            # Merge, deduplicate, sort and take top x
            top_pool = pd.concat([top_pool, total_data])
            top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
            top_pool = top_pool.sort_values(by="score", ascending=False)
            top_pool = top_pool.head(config["num_molecules"])
            bt.logging.info(f"[Miner] Top pool now has {len(top_pool)} molecules after merging")
            bt.logging.info(f"[Miner] Top molecules: {top_pool[['name', 'score']]}")
            bt.logging.info(f"[Miner] Average top score: {top_pool['score'].mean()}")
            # format to accepted format
            top_entries = {"molecules": top_pool["name"].tolist()}

            # write to file
            with open(output_path, "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)
            iteration += 1

def main(config: dict):
    iterative_sampling_loop(
        db_path=DB_PATH,
        output_path= "output.json",
        config=config,
    )
 

if __name__ == "__main__":
    config = get_config()
    main(config)
