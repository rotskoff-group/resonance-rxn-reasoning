"""Merge the completed 8B DPO adapter into its recorded S2 base, on CPU.

Ported from run_dir('s3_8b')/scripts/merge_for_benchmark.py.

    python -m resonance.stages.qwen3_8b.merge_lora

Reads outputs/model (adapter) and writes outputs/benchmark_model under run_dir('s3_8b').
The base is the adapter's recorded `base_model_name_or_path`.
"""
import json

from resonance.paths import run_dir
from resonance.stages.common import checkpoint, sha256, write_json

ROOT=run_dir('s3_8b')


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)
    adapter=ROOT/'outputs/model'
    out=ROOT/'outputs/benchmark_model'
    cfg=json.loads((adapter/'adapter_config.json').read_text())
    base=checkpoint(cfg['base_model_name_or_path'])
    state=json.loads((adapter/'checkpoint-3000/trainer_state.json').read_text())
    assert state['global_step']==state['max_steps']==3000
    identity={'base_model':str(base),'adapter':str(adapter),
              'adapter_sha256':sha256(adapter/'adapter_model.safetensors'),
              'adapter_config_sha256':sha256(adapter/'adapter_config.json'),'steps':3000}
    if (out/'MERGE_COMPLETE.json').exists():
        assert json.loads((out/'MERGE_COMPLETE.json').read_text())==identity
        checkpoint(out)
        return
    assert not out.exists(), f'Inspect partial output before retrying: {out}'
    print(json.dumps(identity),flush=True)
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,device_map='cpu')
    model=PeftModel.from_pretrained(model,adapter).merge_and_unload(safe_merge=True)
    model.save_pretrained(out,max_shard_size='5GB')
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    checkpoint(out)
    write_json(out/'MERGE_COMPLETE.json',identity)
    print('Merged checkpoint validated',flush=True)


if __name__=='__main__':
    main()
