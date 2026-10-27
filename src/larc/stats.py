"""RDI, Friedman and Nemenyi, exactly as defined in the COR review (Section 'Comparative performance').
RDI = (Zbest - ZA) / (Zbest - Zworst) * 100, pool-relative, lower is better."""
import numpy as np
import pandas as pd
from scipy import stats

# Nemenyi q_0.05 (Demsar 2006), k = 2..11
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219}


def rdi(obj: pd.DataFrame) -> pd.DataFrame:
    best, worst = obj.max(axis=1), obj.min(axis=1)
    span = (best - worst).to_numpy()[:, None]
    val = (best.to_numpy()[:, None] - obj.to_numpy()) / np.where(span == 0, np.nan, span) * 100
    return pd.DataFrame(np.nan_to_num(val), index=obj.index, columns=obj.columns)


def mean_ranks(obj: pd.DataFrame) -> pd.Series:
    return obj.rank(axis=1, ascending=False, method="average").mean()


def friedman(obj: pd.DataFrame):
    return stats.friedmanchisquare(*[obj[c].values for c in obj.columns])


def nemenyi_cd(k: int, n: int) -> float:
    return Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
