import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import bittensor as bt
from rdkit.Chem import Descriptors
from dotenv import load_dotenv

load_dotenv(override=True)


def get_heavy_atom_count(smiles: str) -> int:
    """
    Calculate the number of heavy atoms in a molecule from its SMILES string.
    """
    count = 0
    i = 0
    while i < len(smiles):
        c = smiles[i]
        
        if c.isalpha() and c.isupper():
            elem_symbol = c
            
            # If the next character is a lowercase letter, include it (e.g., 'Cl', 'Br')
            if i + 1 < len(smiles) and smiles[i + 1].islower():
                elem_symbol += smiles[i + 1]
                i += 1 
            
            # If it's not 'H', count it as a heavy atom
            if elem_symbol != 'H':
                count += 1
        
        i += 1
    
    return count


def compute_maccs_entropy(smiles_list: list[str]) -> float:
    """
    Computes fingerprint entropy from MACCS keys for a list of SMILES.

    Parameters:
        smiles_list (list of str): Molecules in SMILES format.

    Returns:
        avg_entropy (float): Average entropy per bit.
    """
    n_bits = 167  # RDKit uses 167 bits (index 0 is always 0)
    bit_counts = np.zeros(n_bits)
    valid_mols = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.array(fp)
            bit_counts += arr
            valid_mols += 1

    if valid_mols == 0:
        raise ValueError("No valid molecules found.")

    probs = bit_counts / valid_mols
    entropy_per_bit = np.array([
        -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
        for p in probs
    ])

    avg_entropy = np.mean(entropy_per_bit)

    return avg_entropy


def find_chemically_identical(smiles_list: list[str]) -> dict:
    """
    Check for identical molecules in a list of SMILES strings by converting to InChIKeys.
    """
    inchikey_to_indices = {}
    
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                inchikey = Chem.MolToInchiKey(mol)
                if inchikey not in inchikey_to_indices:
                    inchikey_to_indices[inchikey] = []
                inchikey_to_indices[inchikey].append(i)
        except Exception as e:
            bt.logging.warning(f"Error processing SMILES {smiles}: {e}")
    
    duplicates = {k: v for k, v in inchikey_to_indices.items() if len(v) > 1}
    
    return duplicates

def num_rotatable_bonds(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        num = Descriptors.NumRotatableBonds(mol)
        return num
    except Exception as e:
        return 0
