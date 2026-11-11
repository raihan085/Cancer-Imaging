"""fit_patch.py -- the one change revision 4 makes to fit.py: optional theta0 (restart
from a previous solution) and lsq_kw (tolerances) on fit(). Defaults reproduce the
old behaviour exactly. Applied by make_v3_nb.py to the notebook's fit.py cell."""
REPS = [
    ('''        trend=True, n_jobs=1, irls=0, verbose=True):
    """irls > 0''',
     '''        trend=True, n_jobs=1, irls=0, verbose=True, theta0=None, lsq_kw=None):
    """theta0 restarts from a previous FitResult.theta (seed_beta is then unused);
    lsq_kw passes tolerances (xtol, ftol, gtol) to least_squares.

    irls > 0'''),
    ('''    if seed_beta is None:
        _, _, _, seed_beta = grid_search(prob, diff, verbose=verbose)
    theta0 = bm.start(*seed_beta)

    kw = {}
''',
     '''    if theta0 is None:
        if seed_beta is None:
            _, _, _, seed_beta = grid_search(prob, diff, verbose=verbose)
        theta0 = bm.start(*seed_beta)
    theta0 = np.asarray(theta0, float)

    kw = dict(lsq_kw or {})
'''),
]


def apply(src):
    for old, new in REPS:
        assert src.count(old) == 1, old[:60]
        src = src.replace(old, new)
    return src


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    src = apply(open(p).read())
    with open(p, "w") as f:
        f.write(src)
    print("patched", p)
