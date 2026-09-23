"""
Module for package-wide constants
"""

from typing import Final

FUNCTIONAL_GROUPS: Final[dict[str, str]] = {
    "Phosphine": "[PX3;H0]([#6])([#6])[#6]",
    "Phosphate": "[PX4](=[OX1])([OX2H])([OX2H])[OX2]",
    "Phosphonate": "[PX4](=[OX1])([OX2H])[#6]",
    "Isothiocyanate": "[NX2]=[CX2]=[SX1]",
    "Isocyanate": "[NX2]=[CX2]=[OX1]",
    "Peroxide": "[OX2][OX2]",
    "Acyl Halide": "[CX3](=O)[F,Cl,Br,I]",
    "Anhydride": "[CX3](=O)[OX2][CX3](=O)",
    "Thiourea": "[NX3][CX3](=[SX1])[NX3]",
    "Thioamide": "[CX3](=[SX1])[NX3]",
    "Disulfide": "[SX2][SX2]",
    "Azo": "[#6][NX2]=[NX2][#6]",
    "Hydrazone": "[NX3][NX2]=[CX3]",
    "Hydrazine": "[NX3][NX3]",
    "Nitroso": "[NX2](=O)",
    "Imide": "[CX3](=O)[NX3][CX3](=O)",
    "Guanidine": "[NX3][CX3](=[NX2])[NX3]",
    "Amidine": "[CX3](=[NX2])[NX3]",
    "Sulfonic Acid": "[SX4](=[OX1])(=[OX1])[OX2H]",
    "Sulfonate Ester": "[SX4](=[OX1])(=[OX1])[OX2][#6]",
    "Sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "Sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "Sulfoxide": "[SX3](=[OX1])([#6])[#6]",
    "Indole": "c1ccc2[nH]ccc2c1",
    "Morpholine": "[OX2]1CC[NX3]CC1",
    "Piperazine": "[NX3H]1CC[NX3]CC1",
    "Piperidine": "[NX3H]1CCCCC1",
    "Pyrazole": "c1cc[nH]n1",
    "Imidazole": "c1cnc[nH]1",
    "Thiophene": "c1ccsc1",
    "Furan": "c1ccoc1",
    "Pyrimidine": "n1ccncc1",
    "Nitrile": "[NX1]#[CX2]",
    "Nitro": "[NX3+](=O)[OX1-]",
    "Epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "Enol": "[OX2H][#6X3]",
    "Carbonate": "[OX2][CX3](=O)[OX2]",
    "Carbamate": "[NX3][CX3](=O)[OX2]",
    "Urea": "[NX3][CX3](=O)[NX3]",
    "Imine": "[NX2;H1]=[CX3]",
    "Quaternary Ammonium": "[NX4+]",
    "Thiol": "[SX2H]",
    "Thioether": "[SX2]([#6])[#6]",
    "Pyridine": "n1ccccc1",
    "Phenol": "[OX2H][cX3]",
    "Carboxylic Acid": "[CX3](=O)[OX2H1]",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Primary Amine": "[NX3;H2;!$(NC=O)]",
    "Secondary Amine": "[NX3;H1;!$(NC=O)]([#6])[#6]",
    "Tertiary Amine": "[NX3;H0;!$(NC=O)]([#6])([#6])[#6]",
    "Amide": "[CX3](=O)[NX3]",
    "Ester": "[CX3](=O)[OX2][#6;!H0,!H1]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Alcohol": "[OX2H][CX4;!$(C(O)(O))]",
    "Ether": "[OX2]([#6])[#6]",
    "Benzene Ring": "c1ccccc1",
}

# ELEMENTS
TRACKED_ELEMENTS: Final[list[str]] = ["C", "N", "O", "S", "F", "Cl", "Br"]

# HETEROPOOL is a list of common inert alkyl and aryl substructres, some
# of which contain heteroatoms
# The R_CARBON_POOL extends the R_HETERO_POOL by including additional substructures 
# that contain carbon atoms.
R_HETERO_POOL: Final[list[str]] = [
    "C",
    "CC",
    "CCC",
    "CC(C)C",
    "CC(C)(C)C",
    "C1CC1",
    "C1CCCCC1",
    "c1ccccc1",
    "c1ccc(F)cc1",
    "c1ccc(Cl)cc1",
    "c1ccc(C)cc1",
]
R_CARBON_POOL: Final[list[str]] = R_HETERO_POOL + [
    "c1cc(C(F)(F)F)ccc1",
    "c1cccnc1",
    "c1ccncc1",
]
