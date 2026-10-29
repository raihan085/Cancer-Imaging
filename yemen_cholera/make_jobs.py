"""Write job lists (one shell command per line) for the extended study."""
import sys, json
CITIES = ['Amanat Al Asimah', "Sana'a", 'Al Mahwit', 'Amran', 'Dhamar', 'Raymah',
          'Hajjah', 'Al Hudaydah', 'Marib', 'Ibb', 'Al Bayda', "Al Dhale'e", "Sa'ada",
          'Al Jawf', 'Taizz', 'Lahj', 'Abyan', 'Aden', 'Shabwah', 'Hadramaut', 'Al Maharah']
# Chosen by total reported cases in the record: the top three, the median
# governorate (11th of 21) and the smallest.
FOCUS = ["Al Hudaydah", "Hajjah", "Dhamar", "Al Bayda", "Hadramaut"]
PINN = ["dpinn", "dpinn_fs", "fdpinn", "ff"]
# The ONE place to change the study design: analysis.py imports these.
N_SEEDS = 10          # seeds per PINN arm, full record and every holdout fold
N_SEEDS_AUX = 3       # seeds for the literature-calibrated recovery runs
slug = lambda c: c.lower().replace(" ", "_").replace("'", "")

def job(city, arm, seed=0, fold=-1, variant="base", truth="", extra=""):
    tag = f"{arm}_s{seed}_f{fold}_{variant}" + (f"_{truth}" if truth else "")
    out = f"results/{slug(city)}/{tag}.pkl"
    return (f'python run_job.py --city "{city}" --arm {arm} --seed {seed} --fold {fold} '
            f'--variant {variant} {"--truth " + truth if truth else ""} {extra} --out {out} '
            f'>> logs/{slug(city)}.log 2>&1')

def cheap():
    J = []
    for c in CITIES:                                  # items 2, 7, 8, 9
        for arm in ("spline", "sir", "det"):
            for f in (-1, 0, 1, 2, 3):
                J.append(job(c, arm, fold=f))
    for c in FOCUS:                                   # item 6
        for v in ("poisson", "trunc", "age_eq", "age_ln", "onset", "nowinsor", "wls", "M2", "M6"):
            J.append(job(c, "det", variant=v))
    for c in FOCUS:                                   # item 4
        for tr in ("lit_share20", "lit_share70"):
            for s in range(N_SEEDS_AUX):
                J.append(job(c, "det", seed=s, truth=tr))
    return J

def pinn():
    """PINN fits in priority order, so the most important results land first."""
    J = []
    for s in range(N_SEEDS):                          # items 1, 3: full data
        for c in FOCUS:
            for arm in PINN:
                J.append(job(c, arm, seed=s))
    for s in range(N_SEEDS):                          # item 2: 4-fold holdout
        for f in range(4):
            for c in FOCUS:
                for arm in ("dpinn", "ff"):
                    J.append(job(c, arm, seed=s, fold=f))
    for s in range(N_SEEDS_AUX):                      # item 4: recovery
        for tr in ("lit_share20", "lit_share70"):
            for c in FOCUS:
                for arm in ("dpinn", "ff"):
                    J.append(job(c, arm, seed=s, truth=tr))
    for v in ("poisson", "wren5", "wren80", "M2", "M6"):   # item 6: PINN sensitivity
        for c in FOCUS:
            for arm in ("dpinn", "ff"):
                J.append(job(c, arm, variant=v))
    for s in range(N_SEEDS):                          # item 2: other two cells
        for f in range(4):
            for c in FOCUS:
                for arm in ("dpinn_fs", "fdpinn"):
                    J.append(job(c, arm, seed=s, fold=f))
    return J


if __name__ == "__main__":
    print("\n".join(globals()[sys.argv[1]]()))
