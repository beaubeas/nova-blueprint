import os
import json
import time
import bittensor as bt
import pandas as pd
from typing import List
from PSICHIC.wrapper import PsichicWrapper
from validator.validity import generate_valid_random_molecules_batch

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "combinatorial_db", "molecules.sqlite")
INPUT_PATH = os.path.join(SCRIPT_DIR, "input.json")
MODEL_PATH = os.path.join(SCRIPT_DIR, 'PSICHIC/trained_weights/TREAT2/model.pt')

psichic_model = PsichicWrapper()

def target_scores_from_data(data: pd.Series, target_sequence: List[str]) -> pd.Series:
    global psichic_model

    try:
        target_sequence = target_sequence[0]
        psichic_model.initialize_model()
        psichic_model.initialize_protein(target_sequence)
        psichic_model.initialize_smiles(data.tolist())
        bt.logging.info(f"* Target Protein *")
        scores = psichic_model.score_molecules()
        scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
        return scores["target"]
    except Exception as e:
        bt.logging.error(f"Error scoring target {target_sequence}: {e}")
        return pd.Series(dtype=float)

def antitarget_scores_from_data(data: pd.Series, antitarget_sequence: List[str]) -> pd.Series:
    global psichic_model
    antitarget_scores = []
    psichic_model.smiles_list = data.to_list()
    psichic_model.smiles_dict = {k: v for k, v in psichic_model.smiles_dict.items() if k in data.tolist()}
    try:
        for i in range(len(antitarget_sequence)):
            psichic_model.initialize_model()
            psichic_model.initialize_protein(antitarget_sequence[i])
            bt.logging.info(f"* Antitarget Protein ({i + 1}) *")
            scores = psichic_model.score_molecules()
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            antitarget_scores.append(scores[f"anti_{i}"])
        data['Anti'] = pd.DataFrame(antitarget_scores).mean(axis=0).values
        return data['Anti']
        
    except Exception as e:
        bt.logging.error(f"Error scoring antitarget {antitarget_sequence[i]}: {e}")
        return pd.Series(dtype=float)

def download_model_weights(model_path: str, i: int):
    try:
        os.system(f"wget -O {model_path} https://huggingface.co/Metanova/TREAT-2/resolve/main/model.pt")
        bt.logging.info('Downloaded Model Weights Successfully.')
    except Exception as e:
        if  i==5:
            bt.logging.error(f"Failed to download model weights after {i} attempts.")
            return
        bt.logging.error(f"Error downloading model weights, Retrying... Attempt {i+1}/5:")
        time.sleep(2)
        download_model_weights(model_path, i + 1)
    

def get_config(input_file: os.path):
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}
    return config

def iterative_sampling_loop(
    db_path: str,
    config: dict = None,
) -> None:
    
    parent_dir = os.path.dirname(SCRIPT_DIR)
    output_path = os.path.join(parent_dir, "output", "result.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_samples = config["num_molecules"] * 5  # Sample 5x the number of molecules needed to ensure diversity
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])
    specific_pool = []
    iteration = 0
    while True:
        if rxn_id == 5:
            specific_pool = [193892,190177,189908,189662,189449,194399,193228,189448,196793,193506]
        if rxn_id == 4:
            specific_pool = [221193,218682,219061,219046,218998,218681,213053,212649,212650,215170]

        data = generate_valid_random_molecules_batch(
            rxn_id, n_samples, db_path, config, batch_size=n_samples, seed=iteration + 42, specific_pool=specific_pool
        )

        data = data.reset_index(drop=True)
        data['Target'] = target_scores_from_data(data['smiles'], config['target_sequences'])
        data = data.sort_values(by='Target', ascending=False)
        max_value = data['Target'].max()
        if max_value > 6.5:
            data = data[data['Target']>6.5]
            bt.logging.info(f"[Miner] After target scoring, {len(data)} molecules remain with Target > 6.5.")

        else:
            data = data.iloc[:50]
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
        if rxn_id not in [4,5]:
            top_data = list(set([int(i.split(':')[2]) for i in  top_pool.head(5)['name'].tolist()]))
            bt.logging.info(f"[Miner] Top molecule IDs this iteration: {top_data}")
            specific_pool = list(set(top_data + specific_pool) -set(specific_pool))
            bt.logging.info(f"[Miner] Specific pool now has {specific_pool} molecules.")
        bt.logging.info(f"[Miner] Top pool now has {len(top_pool)} molecules after merging, Average top score: {round(top_pool['score'].mean(), 4)}")
        # format to accepted format
        top_entries = {"molecules": top_pool["name"].tolist()}

        # write to file
        with open(output_path, "w") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)
        iteration += 1

def main(config: dict):
    iterative_sampling_loop(
        db_path=DB_PATH,
        config=config,
    )
 

if __name__ == "__main__":
    config = get_config(INPUT_PATH)
    if not os.path.exists(MODEL_PATH):
        download_model_weights(MODEL_PATH, 0)
    else:
        bt.logging.info('Model Weights already exist.') 
    main(config)
