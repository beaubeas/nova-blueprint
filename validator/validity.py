import bittensor as bt
from rdkit import Chem
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
from utils.molecules import (
    get_heavy_atom_count, 
    compute_maccs_entropy,
    num_rotatable_bonds
)
import sqlite3
import random
from typing import List, Tuple
from tqdm import tqdm
from combinatorial_db.reactions import get_reaction_info, get_smiles_from_reaction


def validate_molecules( data: pd.DataFrame, config: dict ) -> pd.DataFrame:
    data['smiles'] = data["name"].apply(get_smiles_from_reaction)
    data['heavy_atoms'] = data['smiles'].apply(get_heavy_atom_count)
    data = data[data['heavy_atoms'] >= config['min_heavy_atoms']]
    data['bonds'] = data['smiles'].apply(num_rotatable_bonds)
    data = data[data['bonds'] >= config['min_rotatable_bonds']]
    data = data[data['bonds'] <= config['max_rotatable_bonds']]
    return data

def generate_inchikey(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        inchikey = Chem.MolToInchiKey(mol)
        return inchikey
    except Exception as e:
        bt.logging.error(f"Error generating InChIKey for SMILES {smiles}: {e}")
        return ""


def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?", 
            (role_mask, role_mask)
        )
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return []

def generate_molecules_from_pools(rxn_id: int, n: int, molecules_A: List[Tuple], molecules_B: List[Tuple], 
                                molecules_C: List[Tuple], is_three_component: bool, seed: int = 42) -> List[str]:
    mol_ids = []
    random.seed(seed)
    
    for i in range(n):
        try:
            # Randomly select molecules for each role
            mol_A = random.choice(molecules_A)
            mol_B = random.choice(molecules_B)
            
            mol_id_A, smiles_A, role_mask_A = mol_A
            mol_id_B, smiles_B, role_mask_B = mol_B
            
            if is_three_component:
                mol_C = random.choice(molecules_C)
                mol_id_C, smiles_C, role_mask_C = mol_C
                product_name = f"rxn:{rxn_id}:{mol_id_A}:{mol_id_B}:{mol_id_C}"
            else:
                product_name = f"rxn:{rxn_id}:{mol_id_A}:{mol_id_B}"
            
            mol_ids.append(product_name)
            
        except Exception as e:
            bt.logging.error(f"Error generating molecule {i+1}/{n}: {e}")
            mol_ids.append(None)
    
    return mol_ids

def generate_valid_random_molecules_batch(
    rxn_id: int, n_samples: int, db_path: str, subnet_config: dict, 
    batch_size: int = 200, seed: int = 42
) -> pd.DataFrame:
    """
    Generate a DataFrame of valid, unique molecules for a given reaction using pandas for all batch operations.
    Returns a DataFrame with columns: name, smiles, InChIKey
    """
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    smarts, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0
    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []
    if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
        bt.logging.error(f"No molecules found for roles A={roleA}, B={roleB}, C={roleC}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    valid_rows = []
    seen_keys = set()
    iteration = 0
    progress_bar = tqdm(total=n_samples, desc="Creating valid molecules", unit="molecule")
    while len(valid_rows) < n_samples:
        needed = n_samples - len(valid_rows)
        batch_size_actual = min(batch_size, needed * 10)
        batch_molecules = generate_molecules_from_pools(
            rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, is_three_component, seed + iteration
        )
        batch_df = pd.DataFrame({"name": batch_molecules})
        batch_df = validate_molecules(batch_df, subnet_config)
        if batch_df.empty:
            iteration += 1
            continue
        # Compute InChIKey for deduplication
        batch_df["InChIKey"] = batch_df["smiles"].apply(generate_inchikey)
        # Remove duplicates within batch
        batch_df = batch_df.drop_duplicates(subset=["InChIKey"], keep="first")
        # Remove molecules already seen
        batch_df = batch_df[~batch_df["InChIKey"].isin(seen_keys)]
        
        # Add to results and update seen_keys
        for _, row in batch_df.iterrows():
            seen_keys.add(row["InChIKey"])
            valid_rows.append(row)
            progress_bar.update(1)
            if len(valid_rows) >= n_samples:
                break
        iteration += 1
        
    progress_bar.close()
    if not valid_rows:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    result_df = pd.DataFrame(valid_rows)[["name", "smiles", "InChIKey"]].reset_index(drop=True)
    return result_df.head(n_samples)