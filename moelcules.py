from rdkit import Chem
import bittensor as bt
from rdkit.Chem import Descriptors
from dotenv import load_dotenv
import pandas as pd
import warnings
import sqlite3
import random
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
load_dotenv(override=True)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info

def get_heavy_atom_counts(smiles: str) -> int:
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


class _MoleculeDescriptor:
    __slots__ = ("mol", "heavy_atoms", "rotatable_bonds", "inchikey")

    def __init__(self, mol: Optional[Chem.Mol]):
        self.mol = mol
        if mol is None:
            self.heavy_atoms = 0
            self.rotatable_bonds = 0
            self.inchikey = ""
        else:
            self.heavy_atoms = get_heavy_atom_counts(self.mol)
            self.rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            self.inchikey = Chem.MolToInchiKey(mol)


class MoleculeDescriptorCache:
    """
    Cache RDKit descriptors to avoid repeatedly parsing SMILES strings.
    This drastically reduces runtime when the same SMILES are evaluated
    multiple times throughout the sampling loops.
    """

    def __init__(self):
        self._cache: Dict[str, _MoleculeDescriptor] = {}

    def describe(self, smiles: str) -> _MoleculeDescriptor:
        if smiles in self._cache:
            return self._cache[smiles]
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        descriptor = _MoleculeDescriptor(mol)
        self._cache[smiles] = descriptor
        return descriptor


_descriptor_cache = MoleculeDescriptorCache()


def get_heavy_atom_count(smiles: str) -> int:
    """
    Calculate the number of heavy atoms in a molecule from its SMILES string.
    """
    return _descriptor_cache.describe(smiles).heavy_atoms

def find_chemically_identical(smiles_list: list[str]) -> dict:
    """
    Check for identical molecules in a list of SMILES strings by converting to InChIKeys.
    """
    inchikey_to_indices = {}
    
    for i, smiles in enumerate(smiles_list):
        try:
            inchikey = _descriptor_cache.describe(smiles).inchikey
            if inchikey:
                inchikey_to_indices.setdefault(inchikey, []).append(i)
        except Exception as e:
            bt.logging.warning(f"Error processing SMILES {smiles}: {e}")
    
    duplicates = {k: v for k, v in inchikey_to_indices.items() if len(v) > 1}
    
    return duplicates

def num_rotatable_bonds(smiles):
    try:
        return _descriptor_cache.describe(smiles).rotatable_bonds
    except Exception as e:
        return 0

def generate_inchikey(smiles: str) -> str:
    try:
        return _descriptor_cache.describe(smiles).inchikey
    except Exception as e:
        bt.logging.error(f"Error generating InChIKey for SMILES {smiles}: {e}")
        return ""


def validate_molecules( data: pd.DataFrame, config: dict ) -> pd.DataFrame:
    data = data.copy()
    data['smiles'] = data["name"].apply(get_smiles_from_reaction)
    data['_descriptor'] = data['smiles'].map(lambda s: _descriptor_cache.describe(s))
    data['heavy_atoms'] = data['_descriptor'].map(lambda d: d.heavy_atoms)
    data = data[data['heavy_atoms'] >= config['min_heavy_atoms']]
    data['bonds'] = data['_descriptor'].map(lambda d: d.rotatable_bonds)
    data = data[data['bonds'] >= config['min_rotatable_bonds']]
    data = data[data['bonds'] <= config['max_rotatable_bonds']]
    data['InChIKey'] = data['_descriptor'].map(lambda d: d.inchikey)
    data = data.drop(columns=['_descriptor'])
    return data


@lru_cache(maxsize=None)
def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
    """
    Get all molecules that have the specified role_mask.
    
    Args:
        role_mask: The role mask to filter by
        db_path: Path to the molecules database
        
    Returns:
        List of tuples (mol_id, smiles, role_mask) for molecules that match the role
    """
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?", 
                (role_mask, role_mask)
            )
            results = cursor.fetchall()
        return results
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return []

def generate_valid_random_molecules_batch(rxn_id: int, n_samples: int, db_path: str, subnet_config: dict, 
                                 batch_size: int = 200, seed: int = None,
                                 elite_names: list[str] = None, elite_frac: float = 0.5, mutation_prob: float = 0.1,
                                 avoid_inchikeys: set[str] = None) -> dict:
    """
    Efficiently generate n_samples valid molecules by generating them in batches and validating.
    """
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
        return {"molecules": [None] * n_samples}
    
    smarts, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0
    
    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []
    
    if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
        bt.logging.error(f"No molecules found for roles A={roleA}, B={roleB}, C={roleC}")
        return {"molecules": [None] * n_samples}

    valid_rows = []
    seen_keys = set()
    iteration = 0
    
    while len(valid_rows) < n_samples:
        iteration += 1
        
        needed = n_samples - len(valid_rows)
        
        batch_size_actual = min(batch_size, needed * 2)
        
        emitted_names = set()
        if elite_names:
            n_elite = max(0, min(batch_size_actual, int(batch_size_actual * elite_frac)))
            n_rand = batch_size_actual - n_elite

            elite_batch = generate_offspring_from_elites(
                rxn_id=rxn_id,
                n=n_elite,
                elite_names=elite_names,
                molecules_A=molecules_A,
                molecules_B=molecules_B,
                molecules_C=molecules_C,
                is_three_component=is_three_component,
                mutation_prob=mutation_prob,
                seed=seed,
                avoid_names=emitted_names,
                avoid_inchikeys=avoid_inchikeys,
                max_tries=10
            )
            emitted_names.update(elite_batch)

            rand_batch = generate_molecules_from_pools(
                rxn_id, n_rand, molecules_A, molecules_B, molecules_C, is_three_component, seed
            )
            rand_batch = [n for n in rand_batch if n and (n not in emitted_names)]
            batch_molecules = elite_batch + rand_batch
        else:
            batch_molecules = generate_molecules_from_pools(
                rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, is_three_component, seed
            )
        
        batch_df = pd.DataFrame({"name": batch_molecules})
        batch_df = batch_df[batch_df["name"].notna()]  # Remove None values
        if batch_df.empty:
            continue
            
        batch_df = validate_molecules(batch_df, subnet_config)
        
        if batch_df.empty:
            continue
            
        # Compute InChIKey for deduplication
        batch_df["InChIKey"] = batch_df["smiles"].apply(generate_inchikey)

        drop_list = find_chemically_identical(batch_df['smiles'].to_list())

        to_drop = []
        for i in drop_list:
            to_drop += drop_list[i][1:]
        
        batch_df = batch_df.drop(batch_df.index[to_drop])

        batch_df = batch_df.drop_duplicates(subset=["InChIKey"], keep="first")
        batch_df = batch_df[~batch_df["InChIKey"].isin(seen_keys)]
        
        for _, row in batch_df.iterrows():
            seen_keys.add(row["InChIKey"])
            valid_rows.append(row)
            if len(valid_rows) >= n_samples:
                break
    
    if not valid_rows:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    result_df = pd.DataFrame(valid_rows)[["name", "smiles", "InChIKey"]].reset_index(drop=True)
    return result_df.head(n_samples)


def generate_molecules_from_pools(rxn_id: int, n: int, molecules_A: List[Tuple], molecules_B: List[Tuple], 
                                molecules_C: List[Tuple], is_three_component: bool, seed: int = None) -> List[str]:
    mol_ids = []

    if seed is not None:
        random.seed(seed)
    
    for i in range(n):
        try:
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

def _parse_components(name: str) -> tuple[int, int, int | None]:
    # name format: "rxn:{rxn_id}:{A}:{B}" or "rxn:{rxn_id}:{A}:{B}:{C}"
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None
    A = int(parts[2]); B = int(parts[3])
    C = int(parts[4]) if len(parts) > 4 else None
    return A, B, C

def _ids_from_pool(pool):
    return [x[0] for x in pool]

def generate_offspring_from_elites(rxn_id: int, n: int, elite_names: list[str],
                                   molecules_A, molecules_B, molecules_C, is_three_component: bool,
                                   mutation_prob: float = 0.1, seed: int | None = None,
                                   avoid_names: set[str] = None,
                                   avoid_inchikeys: set[str] = None,
                                   max_tries: int = 10) -> list[str]:
    if seed is not None:
        random.seed(seed)
    elite_As, elite_Bs, elite_Cs = set(), set(), set()
    for name in elite_names:
        A, B, C = _parse_components(name)
        if A is not None: elite_As.add(A)
        if B is not None: elite_Bs.add(B)
        if C is not None and is_three_component: elite_Cs.add(C)

    pool_A_ids = _ids_from_pool(molecules_A)
    pool_B_ids = _ids_from_pool(molecules_B)
    pool_C_ids = _ids_from_pool(molecules_C) if is_three_component else []

    out = []
    local_names = set()
    for _ in range(n):
        cand = None
        for _try in range(max_tries):
            use_mutA = (not elite_As) or (random.random() < mutation_prob)
            use_mutB = (not elite_Bs) or (random.random() < mutation_prob)
            use_mutC = (not elite_Cs) or (random.random() < mutation_prob)

            A = random.choice(pool_A_ids) if use_mutA else random.choice(list(elite_As))
            B = random.choice(pool_B_ids) if use_mutB else random.choice(list(elite_Bs))
            if is_three_component:
                C = random.choice(pool_C_ids) if use_mutC else random.choice(list(elite_Cs))
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"

            if avoid_names and name in avoid_names:
                continue
            if name in local_names:
                continue

            if avoid_inchikeys:
                try:
                    s = get_smiles_from_reaction(name)
                    if s:
                        mol = Chem.MolFromSmiles(s)
                        if mol:
                            key = Chem.MolToInchiKey(mol)
                            if key in avoid_inchikeys:
                                continue
                except Exception:
                    pass

            cand = name
            break

        if cand is None:
            cand = name
        out.append(cand)
        local_names.add(cand)
        if avoid_names is not None:
            avoid_names.add(cand)
    return out
