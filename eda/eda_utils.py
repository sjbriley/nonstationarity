import pandas as pd
import numpy as np
import trendet
import matplotlib.pyplot as plt
import tsfel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
from matplotlib.pyplot import figure
from time import time
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from statistics import mean
from sklearn.utils import class_weight
from scipy import stats
from scipy.stats import mannwhitneyu
from datetime import datetime
import string
warnings.filterwarnings("ignore")


def get_dist(data, *_unused, **_kw_unused):
    """
    Fit a few simple distributions and return the best one.

    Returns
    -------
    tuple (dist_name:str, params:tuple)
        dist_name ∈ {'expon', 'norm', 'uniform'}
        params are the `scipy.stats.<dist>.fit()` parameters.
    """
    data = np.asarray(data, dtype=float)
    # skip nan / inf that break the fitters
    data = data[np.isfinite(data)]
    if len(data) == 0:
        # fall back to something harmless
        return ("norm", (0.0, 1.0))

    candidates = {
        "expon":   stats.expon,
        "norm":    stats.norm,
        "uniform": stats.uniform
    }

    best_ll   = -np.inf
    best_name = None
    best_par  = None

    for name, dist in candidates.items():
        try:
            par = dist.fit(data)
            ll  = np.sum(dist.logpdf(data, *par))
            if ll > best_ll:
                best_ll, best_name, best_par = ll, name, par
        except Exception:
            # skip distributions that can’t be fit
            continue

    if best_name is None:
        best_name, best_par = "norm", stats.norm.fit(data)

    return best_name, best_par


def identify_df_trends(df: pd.DataFrame,
                       column: str,
                       window_size: int = 10,
                       identify: str = "both") -> dict:
    """
    Detect monotonic up/down trends and return them as contiguous segments
    in the format expected by `annotate_cp_duration`.

    Example output
    --------------
    {
      "Up Trend":   [ {"from": 123, "to": 157, "index": 123, "duration": 34}, ... ],
      "Down Trend": [ {"from": 310, "to": 350, "index": 310, "duration": 40}, ... ]
    }
    """
    # ── 1. Sanity checks ───────────────────────────────────────────────
    if column not in df.columns:
        raise ValueError(f"{column!r} not found in DataFrame")
    if identify not in {"both", "up", "down"}:
        raise ValueError("identify must be 'both', 'up', or 'down'")

    # ── 2. Ensure a DateTimeIndex for Trendet (spacing only matters) ───
    df_copy = df.copy()
    if not np.issubdtype(df_copy.index.dtype, np.datetime64):
        df_copy.index = pd.date_range("2000-01-01",
                                      periods=len(df_copy),
                                      freq="S")     # 1-second grid

    # ── 3. Run Trendet and reset to positional index ───────────────────
    trend_df = trendet.identify_df_trends(df=df_copy,
                                          column=column,
                                          window_size=window_size,
                                          identify=identify).reset_index(drop=True)

    # ── 4. Convert row-wise flags → list-of-segments ───────────────────
    results = {"Up Trend": [], "Down Trend": []}

    def collect(series: pd.Series, key: str):
        in_seg, start = False, None
        for i, flag in series.items():
            if pd.notna(flag) and not in_seg:
                in_seg, start = True, i
            elif pd.isna(flag) and in_seg:
                results[key].append({
                    "from": start,
                    "to": i - 1,
                    "index": start,
                    "duration": (i - 1) - start
                })
                in_seg = False
        if in_seg:  # tail segment
            results[key].append({
                "from": start,
                "to": len(series) - 1,
                "index": start,
                "duration": (len(series) - 1) - start
            })

    if identify in ("both", "up") and "Up Trend" in trend_df.columns:
        collect(trend_df["Up Trend"], "Up Trend")
    if identify in ("both", "down") and "Down Trend" in trend_df.columns:
        collect(trend_df["Down Trend"], "Down Trend")

    # ── 5. Return according to `identify` parameter ────────────────────
    if identify == "up":
        return {"Up Trend": results["Up Trend"]}
    if identify == "down":
        return {"Down Trend": results["Down Trend"]}
    return results


def annotate_cp_duration(df: pd.DataFrame,
                         trends: dict) -> pd.DataFrame:
    """
    Add two columns to *df*:

        • 'cpd'      – 1 at the first sample of every trend, else 0
        • 'duration' – length (in samples) of that trend, NaN elsewhere

    The *trends* argument must be the dict returned by
    `identify_df_trends`, i.e.:

        {
          "Up Trend":   [ {"from": .., "to": .., "index": .., "duration": ..}, ... ],
          "Down Trend": [ {...}, ... ]
        }
    """
    out = df.copy()
    if 'cpd' not in out.columns:
        out['cpd'] = 0
    if 'duration' not in out.columns:
        out['duration_samples'] = np.nan

    for direction in ("Up Trend", "Down Trend"):
        for seg in trends[direction]:
            # guard: only process well-formed dicts
            idx = seg['index']
            out.at[idx, 'cpd'] = 1
            out.at[idx, 'duration_samples'] = seg['duration']

    return out


# def annotate_cp_duration(df, trends):
#     df = df.copy()
#     df['cpd'] = 0
#     df['duration_samples'] = np.nan
#     for t in trends:
#         df.at[t['index'], 'cpd'] = 1
#         df.at[t['index'], 'duration_samples'] = t['duration']
#     return df


def plot_changepoints(data, start_date, end_date, ptID, col_name):
    figure(figsize=(12, 4), dpi=80)
    data['dates'] = pd.to_datetime(data['dates'])
    data_small = data[(data['dates'] >= start_date) & (data['dates'] <= end_date)]
    plt.plot(data_small['dates'], data_small['glucose_level'])

    #get the breakpoints index from the data
    breakpoints = data_small[data_small[col_name] == 1].index.tolist()
    print(breakpoints)
    breakpoints = data_small[data_small[col_name] == 1]['dates'].tolist()
    #for i in range (len(breakpoints)-1):
    for i in (breakpoints):
        #changepoint = data_small.iloc[breakpoints[i]]['dates']
        print(i)
        #changepoint = data_small.iloc[i]['dates']
        plt.axvline(x = i, color="red", linestyle="--", label='changepoint')
    plt.title(f"Patient {ptID} from {start_date} to {end_date}.")
    plt.ylabel('CGM (mg/dl)', fontsize = 12)
    plt.xlabel('Timestamp',fontsize = 12)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')
    plt.show()


def plot_cpd(data, start_date, end_date, ptID, col_name, show_ckp=False):
    '''
    Plot the cpd in the data. Set new_glucose to True if you want to plot the transformed glucose as well
    col_name: the col name of the cpd indicator...most times cpd_pelt
    '''
    figure(figsize=(12, 4), dpi=80)
    data['dates'] = pd.to_datetime(data['dates'])
    data_small = data[(data['dates'] >= start_date) & (data['dates'] <= end_date)]
    plt.plot(data_small['dates'], data_small['glucose_level'], label='sim_glucose')
    plt.plot(data_small['dates'], data_small['glucose_level_new'], label='sim_glucose_new')

    #get the breakpoints index from the data
    if(show_ckp == True):
        breakpoints = data_small[data_small[col_name] == 1].index.tolist()
        print(breakpoints)
        #for i in range (len(breakpoints)-1):
        for i in (breakpoints):
            #changepoint = data_small.iloc[breakpoints[i]]['dates']
            changepoint = data_small['dates'][i]
            plt.axvline(x = changepoint, color="red", linestyle="-", label='changepoint')
    plt.title(f"Patient {ptID} from {start_date} to {end_date}.")
    plt.ylabel('CGM (mg/dl)', fontsize = 12)
    plt.xlabel('Timestamp',fontsize = 12)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    plt.legend(handles, labels, loc='best')
    plt.show()


def get_features(df, *_args):
    if len(_args) == 2:   # old call
        freq, window = _args
    elif len(_args) == 3: # new call
        _, freq, window = _args
    else:
        raise TypeError("get_features expects 3-4 args")
    win = window * freq
    X = tsfel.time_series_features_extractor(
            tsfel.get_features_by_domain(),
            df['eda_signal'], fs=freq, window_size=win)
    # y = df['cpd'].fillna(0).astype(int).values

    labels = []
    for start in range(0, len(df), win):
        window = df.iloc[start:start+win]
        if len(window) == win:               # only full windows
            labels.append(int(window['cpd'].any()))
    y = np.array(labels, dtype=int)

    return X.reset_index(drop=True), y, df

# def check_sig_diff(
#     sample1: "pd.Series | np.ndarray",
#     sample2: "pd.Series | np.ndarray",
#     *,
#     alpha: float = 0.05,
# ) -> bool:
#     """
#     Return True if `sample1` and `sample2` differ significantly.

#     By default we use a two-sided Mann-Whitney-U test because it is
#     non-parametric and robust to non-normal EDA distributions.
#     """
#     s1 = np.asarray(sample1, dtype=float)
#     s2 = np.asarray(sample2, dtype=float)
#     if len(s1) < 3 or len(s2) < 3:          # not enough data to test
#         return False

#     stat, p = mannwhitneyu(s1, s2, alternative="two-sided")

#     return p < alpha

def check_sig_diff(x, y):
    sig_diff = False
    if (len(x) > 0 and len(y) > 0):
        #print("YESS diff dey")
        stat, p_value = mannwhitneyu(x, y)
        #print(p_value)
        alpha = 0.05
        if(p_value < alpha):
            sig_diff = True #reject H0: x and y are the same
    return sig_diff

def draw_sample(mu, sigma):
    s = np.random.normal(mu, sigma, 1)
    #print(f"Sample:{s}")
    return s

def save_to_pkl(filepath, file_to_save):
    with open(filepath, 'wb') as fp:
        pickle.dump(file_to_save, fp)
