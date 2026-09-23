"""Quota allocation and positional splits shared by the synthetic subset builders.

Ported verbatim (these two functions only) from
run_dir('retro_enum_subsets')/bash_scripts/subset_sample.py.
"""


def allocate(counts: dict[int, int], n_total: int, alpha: float) -> dict[int, int]:
    """Water-filled quota: n_total records split ~ n_i^alpha, capped at n_i."""
    quota = {t: 0.0 for t in counts}
    active = {t for t, n in counts.items() if n > 0}
    remaining = float(n_total)

    while active and remaining > 1e-9:
        weights = {t: (counts[t] ** alpha if alpha > 0 else 1.0) for t in active}
        w_total = sum(weights.values())
        if w_total <= 0:
            break
        saturated = [
            t for t in active if remaining * weights[t] / w_total >= counts[t]
        ]
        if saturated:
            for t in saturated:
                quota[t] = float(counts[t])
                active.discard(t)
                remaining -= counts[t]
            continue
        for t in active:
            quota[t] = remaining * weights[t] / w_total
        break

    # Largest-remainder rounding so the total is exactly n_total.
    floors = {t: int(v) for t, v in quota.items()}
    deficit = min(n_total, sum(counts.values())) - sum(floors.values())
    order = sorted(
        (t for t in quota if floors[t] < counts[t]),
        key=lambda t: (quota[t] - floors[t], quota[t]),
        reverse=True,
    )
    for t in order:
        if deficit <= 0:
            break
        floors[t] += 1
        deficit -= 1
    return floors


def split_for_position(i: int, offset: int = 0) -> str:
    """Deterministic 80/10/10 label from position in a permutation.

    The per-template `offset` rotates the pattern so that templates with fewer
    than 10 selected records do not all land in train; over many templates the
    val/test share stays at 10% instead of being systematically lost.
    """
    r = (i + offset) % 10
    return "val" if r == 8 else ("test" if r == 9 else "train")
