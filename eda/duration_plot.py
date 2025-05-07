import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main():
    with open("models/wrist/wesad_cp_model.pkl","rb") as f:
        mdl = pickle.load(f)
    durations = np.array(mdl["duration"])

    loc, scale = stats.expon.fit(durations, floc=0)
    lam = 1/scale

    norm_par = stats.norm.fit(durations)
    uniform_par = stats.uniform.fit(durations)

    plt.figure(figsize=(6,4))
    plt.hist(durations, bins=20, density=True, alpha=0.6, edgecolor="k")

    x = np.linspace(0, durations.max(), 200)
    plt.plot(x, stats.expon.pdf(x, loc=0, scale=scale),
            label=f"Exp", lw=2)
    plt.plot(x, stats.norm.pdf(x, *norm_par),
            label=f"Normal")
    plt.plot(x, stats.uniform.pdf(x, *uniform_par),
            label="Uniform fit")

    plt.xlabel("Duration (s)")
    plt.ylabel("Probability Density")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig("figure2_duration.png", dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()
