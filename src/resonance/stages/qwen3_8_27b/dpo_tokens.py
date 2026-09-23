"""TRL adapter for generation-token-exact preference examples.

Ported from run_dir('pipeline_27b')/scripts/dpo_tokens.py.
"""
from trl import DPOTrainer


class TokenDPOTrainer(DPOTrainer):
    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        assert {'prompt_ids', 'chosen_ids', 'rejected_ids'} <= set(dataset.column_names)
        for row in dataset:
            assert row['prompt_ids'] and row['chosen_ids'] and row['rejected_ids']
            assert max(len(row['prompt_ids'])+len(row[k]) for k in ('chosen_ids','rejected_ids')) <= args.max_length
        return dataset
