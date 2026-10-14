import json

from ..base import Parser


class ForwardRxnParser(Parser):
    def __init__(self, data_path):
        super().__init__(data_path)

    def load_from_json(
        self, task: dict, representation: str, step_data: dict | None = None
    ):
        """Extract representation of reaction components

        Args:
            task: The full task dictionary
            representation: The representation type (smiles, atom_mapped, iupac)
            step_data: Optional specific step data for multi-step reactions
        """
        if task is None:
            return None

        # checks if there are multiple steps
        data = step_data if step_data is not None else task

        if representation == "smiles":
            sm = data["original_smiles"]["Starting Material"]
            rgts = (
                data["original_smiles"]["Reagents"]
                if data["original_smiles"]["Reagents"]
                else ""
            )
            rgt_list = [r.strip() for r in rgts.split(".") if r.strip()]
            product = data["original_smiles"]["End Product"]
            return {
                "sm": sm,
                "rgts": rgt_list,
                "product": product,
                "metadata": task.get("metadata", {}),
                "answer": product,
                "answer_type": "smiles",
                "question_type": "forward_rxn",
            }

        elif representation == "atom_mapped":
            sm = data["mapped_data"]["mapped"]["starting_material"]
            rgts = (
                data["mapped_data"]["mapped"]["reagents"]
                if data["mapped_data"]["mapped"]["reagents"]
                else ""
            )
            rgt_list = [r.strip() for r in rgts.split(".") if r.strip()]
            product = data["mapped_data"]["mapped"]["end_product"]
            return {
                "sm": sm,
                "rgts": rgt_list,
                "product": product,
                "metadata": task.get("metadata", {}),
                "answer": product,
                "answer_type": "atom_mapped",
                "question_type": "forward_rxn",
            }

        elif representation == "iupac":
            sm = data["iupac"]["Starting Material"]
            rgts = data["iupac"]["Reagents"] if data["iupac"]["Reagents"] else ""
            rgt_list = [r.strip() for r in rgts.split(".") if r.strip()]
            product = data["iupac"]["End Product"]
            return {
                "sm": sm,
                "rgts": rgt_list,
                "product": product,
                "metadata": task.get("metadata", {}),
                "answer": product,
                "answer_type": "iupac",
                "question_type": "forward_rxn",
            }

        else:
            raise ValueError(f"Invalid representation: {representation}")

    def load_reactions(self, representation: str, **kwargs):
        """Load forward reaction tasks from 100_reactions

        Returns a list of reaction dictionaries. For multi-step reactions,
        each step is returned as a separate entry.
        """
        rxns = []
        print(f"Loading reactions with representation: {representation}")

        data_path = self.data_path

        print(f"Loading reactions from {data_path}")
        with open(data_path) as f:
            data = json.load(f)

            for rxn in data:
                if "steps" in rxn.get("mapped_data", {}):
                    for step in rxn["mapped_data"]["steps"]:
                        r = self.load_from_json(rxn, representation, step_data=step)
                        if r is not None:
                            r["metadata"]["step_number"] = step.get("step_number", None)
                            rxns.append(r)
                else:
                    r = self.load_from_json(rxn, representation)
                    if r is not None:
                        rxns.append(r)

        print(
            f"Loaded {len(rxns)} reactions (including individual steps from multi-step reactions)"
        )
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
