

def reverse_smarts(smarts: str) -> str | None:
    """Flip a forward reaction SMARTS into retrosynthetic direction.

    Handles both 2-part (reactants>>products) and 3-part
    (reactants>reagents>products) formats. For 3-part templates, reagents
    are dropped — they do not affect substructure matching.
    """
    if ">>" in smarts:
        lhs, rhs = smarts.split(">>", 1)
        return f"{rhs}>>{lhs}"
    parts = smarts.split(">")
    if len(parts) == 3:
        reactants, _reagents, products = parts
        return f"{products}>>{reactants}"
    return None
