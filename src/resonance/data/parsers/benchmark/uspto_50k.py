import json
from ..base import Parser


class ReactionPredictionUSPTO(Parser):
    """Parser for the USPTO-50K reaction prediction benchmark.

    Each reaction in the JSON has:
        id, Starting Material, Reagent, End Product
    """

    def __init__(self, data_path: str):
        super().__init__(data_path)

    def load_reactions(self, representation: str = "smiles", **kwargs) -> list[dict]:
        """Load reactions from the JSON file."""
        with open(self.data_path) as f:
            reactions = json.load(f)

        parsed = []
        for rxn in reactions:
            product = rxn["original_smiles"]["End Product"]
            sm = rxn["original_smiles"]["Starting Material"]
            rgts = rxn["original_smiles"].get("Reagents", "") or ""
            # Combined reactant set (SM + reagents) for the anti-copy reward guard;
            # not shown to the model. "." lets the reward split into molecules.
            reactants = sm if not rgts else f"{sm}.{rgts}"
            parsed.append(
                {
                    "starting_material": sm,
                    "reagent_list": rgts or "No reagents",
                    "reactants_smiles": reactants,  # for ExactOrTanimotoReward anti-copy guard
                    "end_product": product,
                    "answer": product,
                    "true_product_smiles": product,  # for tanimoto_reward
                    "answer_type": representation,
                    "question_type": "forward_rxn",
                }
            )
        return parsed

    def parse_reaction(self, task_template: str, reaction: dict) -> str:
        """Fill the prompt template with starting material and reagents."""
        return task_template.format(
            starting_material=reaction["starting_material"],
            reagent_list=reaction["reagent_list"],
        )
