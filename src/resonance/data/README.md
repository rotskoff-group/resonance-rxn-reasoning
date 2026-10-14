# resonance/data

Parsers load raw reactions and fill a prompt's task template; prompts supply the system
prompt and the template. Workflow configs name them by class (`parser_name`,
`prompt_name`), resolved from the package roots.

| Class | Module | Used for |
|---|---|---|
| `ForwardRxnParser` + `ForwardRxnPrompt` | `parsers/benchmark/forward_rxn_parser.py`, `prompts/benchmark/forward_rxn.py` | Forward reaction prediction benchmarks (USPTO-50k test). `get_system_prompt("none")` is the paper's short system prompt, `("rules")` the long one. |
| `ReactionPredictionUSPTO` | `parsers/benchmark/uspto_50k.py` | USPTO-50k train prompts (used to sample the 8B S3 completions) |
| `DPOPairsParser` | `parsers/sft/dpo_pairs_parser.py` | Chosen/rejected pairs for the DPO workflow |

`parsers/base.py` and `prompts/base.py` define the `Parser` and `Prompt`
interfaces (`load_reactions`/`parse_reaction`, `get_system_prompt`/`get_task_template`).
