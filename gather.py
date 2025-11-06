import asyncio
from types import SimpleNamespace
from typing import cast
import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata
from typing import Dict, Iterable, List, Optional, Tuple, Any
import os
import time
from dataclasses import dataclass
from utils.proteins import (
    get_challenge_params_from_blockhash,
    get_sequence_from_protein_code,
) 
from dotenv import load_dotenv
from config.config_loader import load_config
import re
COMMITMENT_REGEX = re.compile(
    r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)@(?P<branch>[\w./-]+)$"
)

@dataclass
class Miner:
    uid: int
    block_number: int
    raw: str
    owner: str
    repo: str
    branch: str
    hotkey: str
    coldkey: Optional[str] = None

async def get_commitments(subtensor, metagraph, block_hash: str, netuid: int, min_block: int, max_block: int) -> dict:
    commits = await asyncio.gather(*[
        subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=block_hash,
        ) for hotkey in metagraph.hotkeys
    ])

    result = {}
    for uid, hotkey in enumerate(metagraph.hotkeys):
        commit = cast(dict, commits[uid])
        if commit and min_block < commit['block'] < max_block:
            result[hotkey] = SimpleNamespace(
                uid=uid,
                hotkey=hotkey,
                block=commit['block'],
                data=decode_metadata(commit)
            )
    return result

async def fetch_commitments_from_chain(network: Optional[str], netuid: int, min_block: int, max_block: int) -> List[Tuple[int, int, str, str]]:
    """Fetch plaintext commitments within a block window (one per UID)."""
    subtensor = bt.async_subtensor(network=network)
    await subtensor.initialize()
    metagraph = await subtensor.metagraph(netuid)
    block_hash = await subtensor.determine_block_hash(max_block)
    commits = await get_commitments(
        subtensor=subtensor,
        metagraph=metagraph,
        block_hash=block_hash,
        netuid=netuid,
        min_block=min_block,
        max_block=max_block,
    )
    out: List[Tuple[int, int, str, str]] = []
    for c in commits.values():
        out.append((int(c.uid), int(c.block), str(c.data), str(c.hotkey)))
    return out

def to_miners(commitments: Iterable[Miner]) -> List[Miner]:
    return list(commitments)

def parse_commitment(raw: str, uid: int, block_number: int, hotkey: str) -> Optional[Miner]:
    match = COMMITMENT_REGEX.match(raw.strip())
    if not match:
        return None
    owner = match.group("owner")
    repo = match.group("repo")
    branch = match.group("branch")
    if len(owner) == 0 or len(repo) == 0 or len(branch) == 0:
        return None
    return Miner(uid=uid, block_number=block_number, raw=raw, owner=owner, repo=repo, branch=branch, hotkey=hotkey)


def gather_parse_and_schedule(commit_quads: Iterable[Tuple[int, int, str, str]]) -> List[Miner]:
    parsed: List[Miner] = []
    for uid, block_number, raw, hotkey in commit_quads:
        c = parse_commitment(raw, uid, block_number, hotkey)
        if c is not None:
            parsed.append(c)
    miners = to_miners(parsed)
    miners.sort(key=lambda m: (m.block_number, m.uid))
    return miners

def _extract_miner_config(cfg: dict) -> Dict[str, Any]:
    return {
        "antitarget_weight": cfg["antitarget_weight"],
        "entropy_min_threshold": cfg["entropy_min_threshold"],
        "min_heavy_atoms": cfg["min_heavy_atoms"],
        "min_rotatable_bonds": cfg["min_rotatable_bonds"],
        "max_rotatable_bonds": cfg["max_rotatable_bonds"],
        "num_molecules": cfg["num_molecules"],
    }

def build_challenge_params(block_hash: str) -> Dict[str, Any]:
    """
    Build the single challenge_params dict for one orchestrator run.
    - Loads YAML config
    - Derives target/antitarget codes from block hash
    - Fetches sequences only
    - Includes allowed_reaction if enabled
    """
    raw_cfg = load_config()
    miner_cfg = _extract_miner_config(raw_cfg)

    num_antitargets: int = int(raw_cfg["num_antitargets"])
    include_reaction: bool = bool(raw_cfg["random_valid_reaction"])

    params = get_challenge_params_from_blockhash(
        block_hash=block_hash,
        num_antitargets=num_antitargets,
        include_reaction=include_reaction,
    )

    target_code = params.get("target")
    antitarget_codes: List[str] = params.get("antitargets", [])  # type: ignore
    target_seq = get_sequence_from_protein_code(target_code) if isinstance(target_code, str) else None
    antitarget_seqs = [get_sequence_from_protein_code(code) for code in antitarget_codes]

    out: Dict[str, Any] = {
        "config": miner_cfg,
        "challenge": {
            "target_sequences": [target_seq],
            "antitarget_sequences": antitarget_seqs,
        },
    }
    allowed = params.get("allowed_reaction")
    if allowed:
        out["challenge"]["allowed_reaction"] = allowed
    return out


async def main():
    network = os.environ.get("SUBTENSOR_NETWORK")
    netuid = int(os.environ.get("NETUID", "68"))

    subtensor = bt.async_subtensor(network=network)
    await subtensor.initialize()
    current_block = await subtensor.get_current_block()

    cfg_all = load_config()
    interval_seconds = int(cfg_all["competition_interval_seconds"]) 
    now_ts = int(time.time())


    approx_block_time_s = 12
    blocks_window = max(1, interval_seconds // approx_block_time_s)
    min_block = max(0, current_block - blocks_window)
    max_block = current_block
    submissions = await fetch_commitments_from_chain(network=network, netuid=netuid, min_block=min_block, max_block=max_block)
    bt.logging.info(f"Fetched {len(submissions)} commitments from chain in block window {min_block}-{max_block} submissions={submissions}")
    miners = gather_parse_and_schedule(submissions)
    bt.logging.info(f"current_block={current_block} submissions={len(submissions)} miners={len(miners)}")

    block_hash = await subtensor.determine_block_hash(current_block)
    challenge_params = build_challenge_params(str(block_hash))
    bt.logging.info(f"challenge_params={challenge_params}")

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())