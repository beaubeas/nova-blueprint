"""
PSICHIC-based molecular scoring functionality
"""

from typing import List
import pandas as pd
import bittensor as bt
from PSICHIC.wrapper import PsichicWrapper
# Global variable to store PSICHIC instance - will be set by validator.py
psichic_model = PsichicWrapper()


def target_scores_from_data(data: pd.Series, target_sequence: List[str]) -> pd.Series:
    global psichic_model

    if psichic_model is None:
        bt.logging.error("PSICHIC model not initialized.")
        return pd.Series(dtype=float)
    try:
        target_sequence = target_sequence[0]
        psichic_model.initialize_model(target_sequence)
        bt.logging.info(f"Protein {target_sequence} started")
        scores = psichic_model.score_molecules(data.tolist())
        scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
        bt.logging.info(f"Protein {target_sequence} processed")
        return scores["target"]
    except Exception as e:
        bt.logging.error(f"Error scoring target {target_sequence}: {e}")
        return pd.Series(dtype=float)

def antitarget_scores_from_data(data: pd.Series, antitarget_sequence: List[str]) -> pd.Series:
    global psichic_model
    antitarget_scores = []

    if psichic_model is None:
        bt.logging.error("PSICHIC model not initialized.")
        return pd.Series(dtype=float)
    try:
        for i in range(len(antitarget_sequence)):
            psichic_model.initialize_model(antitarget_sequence[i])
            bt.logging.info(f"Protein {antitarget_sequence[i]} started")
            scores = psichic_model.score_molecules(data.tolist())
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            bt.logging.info(f"Protein {antitarget_sequence[i]} processed")
            antitarget_scores.append(scores[f"anti_{i}"])
        data['Anti'] = pd.DataFrame(antitarget_scores).mean(axis=0).values
        return data['Anti']
        
    except Exception as e:
        bt.logging.error(f"Error scoring antitarget {antitarget_sequence[i]}: {e}")
        return pd.Series(dtype=float)