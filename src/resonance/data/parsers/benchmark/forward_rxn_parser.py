import json

from ..base import Parser


class ForwardRxnParser(Parser):
    def __init__(self, data_path):
        super().__init__(data_path)

    def load_from_json(self, task: dict, representation: str):
        """Extract the SMILES of the reaction components.

        Args:
            task: The full task dictionary
            representation: The representation type (only "smiles" is supported)
        """
        if task is None:
            return None
        if representation != "smiles":
            raise ValueError(f"Invalid representation: {representation}")

        sm = task["original_smiles"]["Starting Material"]
        rgts = (
            task["original_smiles"]["Reagents"]
            if task["original_smiles"]["Reagents"]
            else ""
        )
        rgt_list = [r.strip() for r in rgts.split(".") if r.strip()]
        product = task["original_smiles"]["End Product"]
        return {
            "sm": sm,
            "rgts": rgt_list,
            "product": product,
            "metadata": task.get("metadata", {}),
            "answer": product,
            "answer_type": "smiles",
            "question_type": "forward_rxn",
        }

    def load_reactions(self, representation: str, **kwargs):
        """Load forward reaction tasks from a JSON list of reactions."""
        rxns = []
        print(f"Loading reactions with representation: {representation}")

        data_path = self.data_path

        print(f"Loading reactions from {data_path}")
        with open(data_path) as f:
            data = json.load(f)

            for rxn in data:
                r = self.load_from_json(rxn, representation)
                if r is not None:
                    rxns.append(r)

        print(f"Loaded {len(rxns)} reactions")
        return rxns

    def parse_reaction(self, task_template: str, task: dict):
        """Parse reaction into task template"""

        if task["rgts"]:
            if isinstance(task["rgts"], list):
                reagent_list = ", ".join(task["rgts"])
            else:
                reagent_list = task["rgts"]
        else:
            reagent_list = "No reagents"

        filled_task_template = task_template.format(
            starting_material=task["sm"], reagent_list=reagent_list
        )
        return filled_task_template
