#!/usr/bin/env python3
"""Substructure screen over an arbitrary template id list.

resonance.pipelines.synthetic_data.apply_chunk_screen hardcodes NO_MATCH_2RXN_IDS
(18 ids) with no CLI override, because it was written for the no_match subset. This
wrapper rebinds that module global before calling screen_chunk, so the same, tested
code path can screen all 92 two-slot templates without editing the resonance repo.

Ported from run_dir('forward_enum')/scripts/screen_templates.py.
CLI: python -m resonance.pipelines.forward_enum.screen_templates --input --templates
     --template-ids-json --chunk-id --chunk-size --output-dir
"""
import argparse, json
from resonance.pipelines.synthetic_data import apply_chunk_screen as m

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--templates", required=True)
    p.add_argument("--template-ids-json", required=True, help="template_ids.json from template_slots.py")
    p.add_argument("--chunk-id", type=int, required=True)
    p.add_argument("--chunk-size", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    ids = json.load(open(a.template_ids_json))["two_slot"]
    m.NO_MATCH_2RXN_IDS = frozenset(ids)   # screen_chunk reads this at call time
    print(f"screening {len(ids)} two-slot templates", flush=True)
    m.screen_chunk(parquet_path=a.input, templates_path=a.templates,
                   chunk_id=a.chunk_id, chunk_size=a.chunk_size, output_dir=a.output_dir)
