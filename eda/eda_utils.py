import numpy as np
import pandas as pd
import tsfel
import pickle
from scipy import stats

# def check_sig_diff(ts: np.ndarray, threshold: float) -> np.ndarray:
#     diffs = np.abs(np.diff(ts, prepend=ts[0]))
#     return (diffs > threshold).astype(int)

def check_sig_diff(pre_ts: np.ndarray,
                   post_ts: np.ndarray,
                   alpha: float = 0.05) -> bool:
    """
    Return True if the two sample windows are *significantly* different
    (Welch’s two‑sample t‑test, p < alpha).  If either window has fewer than
    two samples, return False.
    """
    if len(pre_ts) < 2 or len(post_ts) < 2:
        return False
    _, p = stats.ttest_ind(pre_ts, post_ts, equal_var=False, nan_policy="omit")
    return p < alpha


# def annotate_cp_duration(trends: list) -> pd.DataFrame:
#     return pd.DataFrame([{'CPD_Index': t['index'], 'duration': t['duration']} for t in trends])

def annotate_cp_duration(df, trends):
    df = df.copy()
    df['cpd'] = 0
    df['duration_samples'] = np.nan
    for t in trends:
        df.at[t['index'], 'cpd'] = 1
        df.at[t['index'], 'duration_samples'] = t['duration']
    return df

def identify_df_trends(df: pd.DataFrame, col: str, freq: int, window_size: int):
    """
    Very basic changepoint detector: when diff > 2*std within a rolling window.
    Returns list of {'index', 'duration', 'mean_change', 'std_change'} dicts.
    """
    ts = df[col].values
    # trivial detection: find where absolute diff > rolling std
    rolls = pd.Series(ts).rolling(window=window_size*freq, min_periods=1)
    stds  = rolls.std().fillna(0).values
    diffs = np.abs(np.diff(ts, prepend=ts[0]))
    cps   = np.where(diffs > 2*stds)[0].tolist()
    trends = []
    for cp in cps:
        dur = window_size  # placeholder: 1 window
        pre = ts[max(0, cp - window_size*freq):cp]
        post= ts[cp:cp + window_size*freq]
        m    = post.mean() - pre.mean() if len(pre) and len(post) else 0.0
        s    = post.std()  - pre.std()  if len(pre) and len(post) else 0.0
        trends.append({'index': cp, 'duration': dur, 'mean_change': m, 'std_change': s})
    return trends

# def get_features(df: pd.DataFrame, window: int, freq: int) -> pd.DataFrame:
#     cfg = tsfel.get_features_by_domain()
#     feats = tsfel.time_series_features_extractor(cfg, df['eda_signal'], fs=freq)
#     return feats

# def get_dist(data: list, dist_names: list):
#     dists = {}
#     for name in dist_names:
#         if name == 'normal':
#             dists['normal'] = stats.norm.fit(data)
#         elif name == 'exponential':
#             dists['exponential'] = stats.expon.fit(data)
#         # add more as needed
#     return dists

from scipy import stats
import numpy as np

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

def save_to_pkl(path: str, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def get_features(df, *_args):
    if len(_args) == 2:   # old call
        freq, window = _args
    elif len(_args) == 3: # new call
        _, freq, window = _args
    else:
        raise TypeError("get_features expects 3‒4 args")
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

    # return X, y, df
