# prompts/forward_rxn.py

from ..base import Prompt


class ForwardRxnPrompt(Prompt):
    def get_system_prompt(self, system_prompt_option="none"):
        if system_prompt_option == "none":
            text = """
        You are an expert organic chemist specializing in the forward prediction of organic reactions.
        Your task is to predict the major product(s)
        of organic reactions given starting materials and reagents.
        """
        elif system_prompt_option == "rules":
            text = """
        You are an expert organic chemist specializing in the forward prediction of organic reactions.
        Correctly predicting the product of an organic reaction involves understanding of several key concepts.
        Some of the most important concepts to consider include:

        1. Acid-base chemistry: these reactions will proceed from a stronger acid to a weaker one.
        2. Single displacement reactions: Similarly, these reactions will proceed by a more reactive element displacing a less reactive one.
        3. Redox reactions: identify the relationship between reagents and their potential for reducing/oxidizing specific functional groups on a molecule.
        4. Mechanistic analysis: leveraging knowledge of common reaction mechanisms, one can infer key functional groups in a molecule and their potential relationships to reagents and other key intramolecular functional groups

        Your task is to predict the major product(s)
        of organic reactions given starting materials and reagents.
        """
        else:
            raise ValueError(f"Unknown system_prompt_option: {system_prompt_option!r}")
        return text

    def get_task_template(self, representation="smiles"):
        if representation != "smiles":
            raise ValueError(f"Only the smiles representation is supported, got {representation!r}")

        text = """Here are the SMILES of the starting material and reagent(s).
Starting material: {starting_material}
Reagents: {reagent_list}

Predict the single most likely primary product of the chemical reaction in SMILES format.
The reaction should be chemically reasonable, following typical organic reactivity principles.
Instructions:
        1. Identify the functional groups in the starting material
        2. Determine which reagent(s) will react with these functional groups
        3. Apply your knowledge of organic reaction mechanisms to predict the major product
        4. Consider regioselectivity and stereoselectivity where applicable
        If your proposed mechanism or product seems chemically implausible, revise your
        reasoning and try a different mechanistic path.

Return the answer between <answer> and </answer> tags.
"""

        return text
