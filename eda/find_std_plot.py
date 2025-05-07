import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main():
    with open("models/wrist/wesad_cp_model.pkl", "rb") as f:
        mdl = pickle.load(f)
    std_changes = np.array(mdl["std_change"])

    mu, sigma = stats.norm.fit(std_changes)
    min_val = std_changes.min()
    shifted = std_changes - min_val + 1e-6
    expon_loc, expon_scale = stats.expon.fit(shifted, floc=0)
    uniform_loc, uniform_scale = stats.uniform.fit(std_changes)

    plt.figure(figsize=(6,4))
    plt.hist(std_changes, bins=20, density=True, alpha=0.6, edgecolor="k")

    x = np.linspace(std_changes.min(), std_changes.max(), 200)
    plt.plot(
        x,
        stats.expon.pdf(x - min_val + 1e-6, expon_loc, expon_scale),
        label=f"Exponential fit",
        lw=2, linestyle=':'
    )
    plt.plot(
        x,
        stats.uniform.pdf(x, uniform_loc, uniform_scale),
        label=f"Uniform fit",
        lw=2, linestyle='--'
    )
    plt.plot(x, stats.norm.pdf(x, mu, sigma),
            label=f"Normal", lw=2)

    plt.xlabel("Change in SD (µS)")
    plt.ylabel("Probability Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("figure3_std.png", dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    main()
