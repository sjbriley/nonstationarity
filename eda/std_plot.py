import pickle
import numpy as np
from scipy import stats

def main():
    with open("models/wrist/wesad_cp_model.pkl","rb") as f:
        mdl = pickle.load(f)

    durations   = np.array(mdl["duration"])
    mean_changes = np.array(mdl["mean_change"])
    std_changes  = np.array(mdl["std_change"])

    loc, scale = stats.expon.fit(durations, floc=0)   # force loc=0 for a pure Exp(λ)
    lam = 1/scale
    print(f"Duration ~ Exp(rate=λ):  λ = {lam:.4f}   (scale = {scale:.1f}s)")

    mu_mean,   sigma_mean   = stats.norm.fit(mean_changes)
    mu_stdchg, sigma_stdchg = stats.norm.fit(std_changes)
    print(f"Mean-change  ~ N(μ,σ): μ = {mu_mean:.4f} µS, σ = {sigma_mean:.4f} µS")
    print(f"Std-change   ~ N(μ,σ): μ = {mu_stdchg:.4f} µS, σ = {sigma_stdchg:.4f} µS")

if __name__ == '__main__':
    main()