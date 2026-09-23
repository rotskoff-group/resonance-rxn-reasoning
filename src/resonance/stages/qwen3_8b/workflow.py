"""Run a resonance workflow on a paper config, with the `run_dir` resolver registered.

The paper configs spell every location as `${run_dir:<key>}/...` (see resonance.paths), so
the data root is set in exactly one place. This wrapper registers that resolver and then
executes the workflow module as `__main__`, forwarding the Hydra arguments unchanged.

    python -m resonance.stages.qwen3_8b.workflow {sft,benchmark,dpo,evaluate} \
        --config-path <dir> --config-name <name> [overrides...]
"""
import runpy
import sys

from omegaconf import OmegaConf

from resonance.paths import run_dir

WORKFLOWS = ("sft", "benchmark", "dpo", "evaluate")


def register() -> None:
    if not OmegaConf.has_resolver("run_dir"):
        OmegaConf.register_new_resolver("run_dir", lambda key: str(run_dir(key)))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in WORKFLOWS:
        raise SystemExit(f"usage: python -m resonance.stages.qwen3_8b.workflow "
                         f"{{{','.join(WORKFLOWS)}}} --config-path DIR --config-name NAME ...")
    workflow = sys.argv.pop(1)
    register()
    sys.argv[0] = f"resonance.workflows.{workflow}"
    runpy.run_module(f"resonance.workflows.{workflow}", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
