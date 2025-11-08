# minerv1_multiproc.py
import os
import json
import time
import queue                # for queue.Empty
import pandas as pd
import bittensor as bt

# Use torch.multiprocessing to be safe with CUDA
import torch
import torch.multiprocessing as mp

from typing import List
from PSICHIC.wrapper import PsichicWrapper
from utils.molecules import generate_valid_random_molecules_batch

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "combinatorial_db", "molecules.sqlite")
INPUT_PATH = os.path.join(SCRIPT_DIR, "input.json")
MODEL_PATH = os.path.join(SCRIPT_DIR, 'PSICHIC/trained_weights/TREAT2/model.pt')

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

def get_config(input_file: str):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


# ---------- scoring helpers (assume psichic_model is a global in the child process) ----------
def target_scores_from_data(data: pd.Series, target_sequence: List[str]):
    global psichic_model
    try:
        # ensure model and protein are initialized for this process
        seq = target_sequence[0]
        psichic_model.initialize_protein(seq)
        psichic_model.initialize_smiles(data.tolist())
        bt.logging.info("* Target Protein *")
        scores = psichic_model.score_molecules()
        scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
        return scores["target"]
    except Exception as e:
        bt.logging.error(f"Target scoring error: {e}")
        return pd.Series(dtype=float)


def antitarget_scores_from_data(data: pd.Series, antitarget_sequence: List[str]):
    global psichic_model
    try:
        psichic_model.initialize_smiles(data.tolist())
        antitarget_scores = []
        for i, seq in enumerate(antitarget_sequence):
            psichic_model.initialize_protein(seq)
            bt.logging.info(f"* Antitarget Protein ({i + 1}) *")
            scores = psichic_model.score_molecules()
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            antitarget_scores.append(scores[f"anti_{i}"])
        # average across antitargets
        anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
        return anti_series
    except Exception as e:
        bt.logging.error(f"Antitarget scoring error: {e}")
        return pd.Series(dtype=float)


# ---------- producer (CPU) ----------
def cpu_data_producer(mp_queue, config, db_path, stop_event, throttle_batches=5):
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    n_samples = config["num_molecules"] * 5
    iteration = 0

    while not stop_event.is_set():
        # simple throttle: avoid filling the queue too much
        try:
            if mp_queue.qsize() > throttle_batches:
                time.sleep(0.5)
                continue
        except Exception:
            # Some platforms may not support qsize reliably; ignore
            pass

        data = generate_valid_random_molecules_batch(
            rxn_id, n_samples, db_path, config, batch_size=n_samples, seed=iteration + 42
        )
        mp_queue.put((iteration, data))
        bt.logging.info(f"[Producer] Put batch {iteration} (queue size ~ {mp_queue.qsize()})")
        iteration += 1


# ---------- consumer (GPU) ----------
def gpu_ai_consumer(mp_queue, config, stop_event, output_path, gpu_device: int = 0):
    """
    This function runs inside a spawned process. It must create its own PsichicWrapper
    and initialize CUDA there.
    """
    global psichic_model

    # Set which CUDA device this process will use
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_device)
        bt.logging.info(f"[Consumer] CUDA available. Using device {gpu_device}")

    # Create and initialize the model inside this process
    psichic_model = PsichicWrapper()
    try:
        psichic_model.initialize_model()
        bt.logging.info("[Consumer] PsichicWrapper initialized in child process.")
    except Exception as e:
        bt.logging.error(f"[Consumer] Failed to initialize PsichicWrapper in child: {e}")
        # If initialization failed, set stop_event to avoid busy-looping
        stop_event.set()
        return

    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score"])

    while not stop_event.is_set() or not mp_queue.empty():
        try:
            iteration, data = mp_queue.get(timeout=2)
        except queue.Empty:
            continue
        except Exception as e:
            bt.logging.error(f"[Consumer] Error getting from queue: {e}")
            continue

        bt.logging.info(f"[Consumer] Processing batch {iteration}")

        # scoring flow (same as your original logic)
        data = data.reset_index(drop=True)
        data['Target'] = target_scores_from_data(data['smiles'], config['target_sequences'])
        data = data.sort_values(by='Target', ascending=False)

        max_value = data['Target'].max() if not data['Target'].empty else float('-inf')
        if max_value > 6.5:
            data = data[data['Target'] > 6.5]
            bt.logging.info(f"[Consumer] {len(data)} molecules remain with Target > 6.5.")
        else:
            data = data.iloc[:50]
            bt.logging.info(f"[Consumer] Selected {len(data)} molecules (top 50 fallback).")

        if data.empty:
            bt.logging.warning(f"[Consumer] Batch {iteration} empty after target threshold.")
            continue

        data['Anti'] = antitarget_scores_from_data(data['smiles'], config['antitarget_sequences'])
        data['score'] = data['Target'] - (config['antitarget_weight'] * data['Anti'])
        data = data[data['score'] > 0]
        total_data = data[["name", "smiles", "InChIKey", "score"]]

        # update the top pool
        top_pool = pd.concat([top_pool, total_data])
        top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
        top_pool = top_pool.sort_values(by="score", ascending=False)
        top_pool = top_pool.head(config["num_molecules"])

        with open(output_path, "w") as f:
            json.dump({"molecules": top_pool["name"].tolist()}, f, ensure_ascii=False, indent=2)

        bt.logging.info(f"[Consumer] Finished batch {iteration}. Top pool size: {len(top_pool)}")

    bt.logging.info("[Consumer] Exiting consumer loop.")


# ---------- main entry ----------
def main_process():
    config = get_config(INPUT_PATH)

    if not os.path.exists(MODEL_PATH):
        download_model_weights(MODEL_PATH, 0)
    else:
        bt.logging.info('Model Weights already exist.')

    parent_dir = os.path.dirname(SCRIPT_DIR)
    output_path = os.path.join(parent_dir, "output", "result.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create inter-process queue and event
    ctx = mp.get_context('spawn')  # use spawn explicitly
    mp_queue = ctx.Queue(maxsize=10)
    stop_event = ctx.Event()

    # Start processes
    producer = ctx.Process(target=cpu_data_producer, args=(mp_queue, config, DB_PATH, stop_event))
    # Optionally pass a GPU id to the consumer if you have multi-GPU setup
    consumer = ctx.Process(target=gpu_ai_consumer, args=(mp_queue, config, stop_event, output_path, 0))

    producer.start()
    consumer.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        bt.logging.info("KeyboardInterrupt: stopping processes...")
        stop_event.set()

    producer.join()
    consumer.join()
    bt.logging.info("All processes joined. Exiting.")


if __name__ == "__main__":
    # Ensure spawn start method is used before any CUDA imports that initialize driver.
    # If you want to force spawn globally, you can also call:
    # mp.set_start_method('spawn', force=True)
    main_process()
