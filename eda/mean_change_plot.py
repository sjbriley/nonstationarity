import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import eda_utils

def main():
    mdl_path = "models/wrist/wesad_cp_model.pkl"
    with open(mdl_path, "rb") as f:
        mdl_dict = pickle.load(f)

    mean_changes = np.array(mdl_dict["mean_change"])
    mu, sigma = stats.norm.fit(mean_changes)
    print(f"Best-fit Normal parameters: mu={mu:.3f}, sigma={sigma:.3f}")

    plt.figure(figsize=(6,4))
    plt.hist(mean_changes, bins=20, density=True, alpha=0.6, edgecolor="k")

    distros = {
        'Normal': stats.norm,
        'Exponential': stats.expon,
        'Uniform': stats.uniform
    }
    x = np.linspace(mean_changes.min(), mean_changes.max(), 200)

    for name, dist in distros.items():
        par = dist.fit(mean_changes)
        pdf = dist.pdf(x, *par)
        plt.plot(x, pdf, label=f"{name} fit")

    # Formatting
    plt.xlabel("Difference in mean (us)", fontsize=12)
    plt.ylabel("Probability Density",   fontsize=12)
    plt.legend(loc="best")
    # plt.tight_layout()
    # plt.show()

    plt.tight_layout()
    plt.savefig("figure.png", dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    main()
