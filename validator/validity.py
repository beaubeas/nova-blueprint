import bittensor as bt
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from utils.molecules import (
    get_heavy_atom_count, 
    compute_maccs_entropy,
    find_chemically_identical
)
from combinatorial_db.reactions import get_smiles_from_reaction

# def validate_molecules_and_calculate_entropy(
#     uid_to_data: dict[int, dict[str, list]],
#     score_dict: dict[int, dict[str, list[list[float]]]],
#     config: dict,
#     allowed_reaction: str = None
# ) -> dict[int, dict[str, list[str]]]:
#     """
#     Validates molecules for all UIDs and calculates their MACCS entropy.
#     Updates the score_dict with entropy values.
    
#     Args:
#         uid_to_data: Dictionary mapping UIDs to their data including molecules
#         score_dict: Dictionary to store scores and entropy
#         config: Configuration dictionary containing validation parameters
#         allowed_reaction: Optional allowed reaction filter for this epoch
        
#     Returns:
#         Dictionary mapping UIDs to their list of valid SMILES strings
#     """
#     valid_molecules_by_uid = {}

#     try:
#         identical_molecules = find_chemically_identical(valid_smiles)
#         if identical_molecules:
#             duplicate_names = []
#             for inchikey, indices in identical_molecules.items():
#                 molecule_names = [valid_names[idx] for idx in indices]
#                 duplicate_names.append(f"{', '.join(molecule_names)} (same InChIKey: {inchikey})")
            
#             bt.logging.warning(f"UID={uid} submission contains chemically identical molecules: {'; '.join(duplicate_names)}")
#             score_dict["entropy"] = None
#             score_dict["block_submitted"] = None
                
#     except Exception as e:
#         bt.logging.warning(f"Error checking for chemically identical molecules for UID={uid}: {e}")

    
#     # Calculate entropy if we have valid molecules, or skip if below threshold
#     try:
#         entropy = compute_maccs_entropy(valid_smiles)
#         if entropy > config['entropy_min_threshold']:
#             score_dict["entropy"] = entropy
#             valid_molecules_by_uid = {"smiles": valid_smiles, "names": valid_names}
#         else:
#             bt.logging.warning(f"UID={uid} submission has entropy below threshold: {entropy}")
#             score_dict["entropy"] = None
#             score_dict["block_submitted"] = None
#             valid_smiles = []
#             valid_names = []

#     except Exception as e:
#         bt.logging.error(f"Error calculating entropy for UID={uid}: {e}")
#         score_dict["entropy"] = None
#         score_dict["block_submitted"] = None
#         valid_smiles = []
#         valid_names = []

            
#     return valid_molecules_by_uid

def num_rotatable_bonds(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        num = Descriptors.NumRotatableBonds(mol)
        return num
    except Exception as e:
        return 0

def validate_molecules( data: pd.DataFrame, config: dict ) -> pd.DataFrame:
    data['smiles'] = data["name"].apply(lambda x: get_smiles_from_reaction(x))
    data['bonds'] = data['smiles'].apply(lambda x: num_rotatable_bonds(x))
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