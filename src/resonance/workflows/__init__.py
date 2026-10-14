"""Hydra entry points: `python -m resonance.workflows.{sft,dpo,benchmark,evaluate}`.

Configs may spell locations as `${run_dir:<key>}/...` (`resonance.paths.run_dir`), so the data
root is set in one place (RESONANCE_DATA_ROOT). Importing this package registers that resolver
before any workflow's Hydra entry point runs.
"""
from omegaconf import OmegaConf

from resonance.paths import run_dir

if not OmegaConf.has_resolver("run_dir"):
    OmegaConf.register_new_resolver("run_dir", lambda key: str(run_dir(key)))
