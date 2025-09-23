# stride_xai/data/preprocess.py

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

def get_binary_or_proba(s: pd.Series) -> pd.Series:
    """Converts a pandas Series to a binary (0 or 1) format.

    Handles boolean, numeric, and common string representations like 'yes'/'no'.

    Args:
        s (pd.Series): The input Series to convert.

    Returns:
        pd.Series: A Series of integers (0 or 1).
    """
    if s.dtype == bool:
        return s.astype(int)
    t = s.astype(str)
    t = t.str.strip().str.lower()
    t = t.str.replace(r"[.\s]+$", "", regex=True)
    t = t.str.replace(r"['\"]", "", regex=True)
    mapping = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1, ">50k": 1, ">50k.": 1,
        "no": 0,  "n": 0, "false": 0, "f": 0, "0": 0, "<=50k": 0, "<=50k.": 0
    }
    mapped = t.map(mapping)
    if mapped.isna().any():
        as_num = pd.to_numeric(t, errors="coerce")
        idx = mapped.isna() & (~as_num.isna())
        mapped[idx] = (as_num[idx] > 0).astype(int)
    if mapped.isna().any():
        vals = t.unique().tolist()
        if len(vals) == 2:
            c = t.value_counts()
            majority = c.index[0]
            minority = c.index[1]
            mapped = t.map({majority: 0, minority: 1})
        else:
            mapped = mapped.fillna(0).astype(int)
    mapped = mapped.fillna(0).astype(int)
    if mapped.nunique() < 2:
        vc = mapped.value_counts()
        if len(vc) == 1:
            tc = t.value_counts()
            if len(tc) >= 2:
                rare = tc.index[-1]
                mapped = np.where(t == rare, 1, 0).astype(int)
            else:
                idxs = np.arange(len(mapped))
                mapped = (idxs % 2).astype(int)
    return pd.Series(mapped, index=s.index).astype(int)

def _onehot_df(df: pd.DataFrame, target: str,
               categorical: Optional[list]=None,
               numeric: Optional[list]=None,
               drop: Optional[list]=None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Performs one-hot encoding for categorical features and prepares X, y matrices.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str): The name of the target column.
        categorical (Optional[list]): A list of categorical column names. If None, inferred from dtype.
        numeric (Optional[list]): A list of numeric column names. If None, inferred from dtype.
        drop (Optional[list]): A list of columns to drop before processing.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: A tuple containing the feature matrix (X), 
                                                   the target vector (y), and the list of feature names.
    """
    if drop:
        df = df.drop(columns=[c for c in drop if c in df.columns])
    if categorical is None:
        categorical = [c for c in df.columns if df[c].dtype == "object" and c != target]
    if numeric is None:
        numeric = [c for c in df.columns if c not in categorical + [target]]
    X_num = df[numeric].apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(X_num.median())
    X_cat = pd.get_dummies(df[categorical], dummy_na=False) if categorical else pd.DataFrame(index=df.index)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df[target].values
    return X.values.astype(float), y, list(X.columns)

def merge_rare_categories(df: pd.DataFrame, categorical: List[str], min_freq: float = 0.02) -> pd.DataFrame:
    """Merges rare categorical values into a single '__OTHER__' category.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical (List[str]): A list of categorical column names to process.
        min_freq (float): The minimum frequency threshold for a category to be kept.

    Returns:
        pd.DataFrame: The DataFrame with rare categories merged.
    """
    df = df.copy()
    for c in categorical:
        vc = df[c].value_counts(normalize=True, dropna=False)
        rare = vc[vc < min_freq].index
        df[c] = df[c].where(~df[c].isin(rare), other="__OTHER__")
    return df