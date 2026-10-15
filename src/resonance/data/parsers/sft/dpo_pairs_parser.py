"""Parser for pre-mined DPO (chosen, rejected) pairs.

Reads a dpo_pairs.jsonl file, e.g. the released S3 DPO pairs (docs/DATA.md).
Each record: {prompt: list[dict], chosen: list[dict], rejected: list[dict]}
"""

import json

from ..base import Parser


class DPOPairsParser(Parser):
    """Loads pre-mined DPO pairs from a JSONL file."""

    def __init__(self, data_path: str):
        super().__init__(data_path)

    def load_reactions(self, representation: str = "smiles", max_samples: int = 0, **kwargs) -> list[dict]:
        records = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        if max_samples and max_samples > 0:
            records = records[:max_samples]
        return records

    def parse_reaction(self, task_template: str, reaction: dict) -> str:
        raise NotImplementedError("DPOPairsParser does not use parse_reaction.")
