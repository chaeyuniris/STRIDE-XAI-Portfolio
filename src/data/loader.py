# stride_xai/data/loader.py

import io
import pandas as pd
import numpy as np
import urllib.request
from zipfile import ZipFile
from pathlib import Path
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer
from ucimlrepo import fetch_ucirepo
from src.data.preprocess import _onehot_df, merge_rare_categories

def load_adult_income():
    url_tr = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    url_te = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    cols = ['age','workclass','fnlwgt','education','education-num','marital-status',
            'occupation','relationship','race','sex','capital-gain','capital-loss',
            'hours-per-week','native-country','income']
    df = pd.concat([
        pd.read_csv(url_tr, header=None, names=cols, na_values=' ?', skipinitialspace=True),
        pd.read_csv(url_te, header=0, names=cols, na_values=' ?', skipinitialspace=True)
    ], ignore_index=True).dropna()
    df['income'] = df['income'].str.contains('>50K').astype(int)
    y = df['income'].values
    X_df = pd.get_dummies(df.drop('income', axis=1))
    X_df = X_df.loc[:, X_df.std(axis=0) > 1e-8]
    X = X_df.values.astype(float)
    feature_names = list(X_df.columns)
    return X, y, feature_names

def load_bank_marketing():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
    with urllib.request.urlopen(url) as response:
        with ZipFile(io.BytesIO(response.read())) as z:
            with z.open('bank-additional/bank-additional-full.csv') as f:
                df = pd.read_csv(f, sep=';')
    df = df.dropna()
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    y = df['y'].values
    X_df = pd.get_dummies(df.drop('y', axis=1))
    X_df = X_df.loc[:, X_df.std(axis=0) > 1e-8]
    X = X_df.values.astype(float)
    feature_names = list(X_df.columns)
    return X, y, feature_names

def load_dataset(DATASET: str, data_dir: str | None = None):
    """Loads and preprocesses a variety of common benchmark datasets.

    Fetches data from sklearn, UCI repository, or local files, and performs
    necessary preprocessing like one-hot encoding.

    Args:
        DATASET (str): The name of the dataset to load (e.g., 'california', 'adult').
        data_dir (str | None): Optional path to a local data directory.

    Returns:
        Tuple: A tuple containing (X, y, feature_names, metadata_dict).
    """
    name = DATASET.lower()

    def _to_binary_series(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s.astype(int)
        try:
            arr = pd.to_numeric(s, errors="coerce")
            u = pd.unique(arr.dropna())
            if set(pd.Series(u).astype(int).tolist()) <= {0, 1}:
                return arr.fillna(0).astype(int)
        except Exception:
            pass
        t = s.astype(str).str.strip().str.lower()
        t = t.str.replace(".", "", regex=False).str.replace(",", "", regex=False)
        mapping = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1, ">50k": 1, "yes.": 1,
        "no": 0,  "n": 0, "false": 0, "f": 0, "0": 0, "<=50k": 0, "no.": 0
        }
        mapped = t.map(mapping)
        if mapped.isna().any():
            num = pd.to_numeric(t, errors="coerce")
            mapped = np.where((~num.isna()) & (num > 0), 1, np.where((~num.isna()) & (num <= 0), 0, mapped))
            mapped = pd.Series(mapped, index=s.index)
        mapped = mapped.fillna(0).astype(int)
        if len(np.unique(mapped)) < 2:
            pos_like = t.isin([">50k", "greaterthan50k"])
            if pos_like.any():
                mapped = np.where(pos_like, 1, mapped).astype(int)
        if len(np.unique(mapped)) < 2:
            raise ValueError("Binary target mapping failed to produce 2 classes. Please inspect raw labels.")
        return mapped.astype(int)

    # === sklearn built-ins ===
    if name == "california":
        data = fetch_california_housing()
        X, y = data.data, data.target
        feat = list(data.feature_names)
        return X, y, feat, {"problem": "reg", "name": "california"}

    if name == "diabetes":
        data = load_diabetes()
        X, y = data.data, data.target
        feat = list(data.feature_names)
        return X, y, feat, {"problem": "reg", "name": "diabetes"}

    if name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        feat = list(data.feature_names)
        return X, y.astype(float), feat, {"problem": "clf", "name": "breast_cancer"}

    # === UCI via ucimlrepo ===
    if name == "heart_disease":
        ds = fetch_ucirepo(name="heart disease")
        Xdf = ds.data.features.copy()
        ydf = ds.data.targets.copy()
        if "num" in ydf.columns:
            y_bin = (ydf["num"] > 0).astype(int)
        else:
            col = ydf.columns[0]
            y_bin = ydf[col]
            if set(np.unique(y_bin)) == {1, 2}:
                y_bin = (y_bin == 2).astype(int)
            else:
                y_bin = (y_bin.astype(float) > 0).astype(int)
        df = Xdf.join(y_bin.rename("target"))
        X, y, feat = _onehot_df(df, target="target")
        return X, y.astype(int), feat, {"problem": "clf", "name": "heart_disease"}

    if name == "german_credit":
        min_freq = 0.05
        ds = fetch_ucirepo(name="statlog (german credit data)")
        Xdf = ds.data.features.copy()
        ydf = ds.data.targets.copy()
        col = ydf.columns[0]
        y_bin = ydf[col].map({1: 0, 2: 1}).fillna(ydf[col]).astype(int)
        df = Xdf.join(y_bin.rename("target"))
        categorical = [c for c in Xdf.columns if Xdf[c].dtype == "object"]
        df2 = merge_rare_categories(df, categorical, min_freq)
        X, y, feat = _onehot_df(df2, target="target")
        return X, y.astype(int), feat, {"problem": "clf", "name": "german_credit"}

    if name == "online_shoppers":
        ds = fetch_ucirepo(name="Online Shoppers Purchasing Intention Dataset")
        Xdf = ds.data.features.copy()
        ydf = ds.data.targets.copy()
        col = ydf.columns[0]
        y_bin = _to_binary_series(ydf[col])
        df = Xdf.join(y_bin.rename("target"))
        X, y, feat = _onehot_df(df, target="target")
        return X, y.astype(int), feat, {"problem": "clf", "name": "online_shoppers"}

    if name in ("adult", "adult_income"):
        X, y, feat = load_adult_income()
        return X, y.astype(int), feat, {"problem": "clf", "name": "adult"}

    if name in ("bank_marketing", "bank"):
        X, y, feat = load_bank_marketing()
        return X, y.astype(int), feat, {"problem": "clf", "name": "bank_marketing"}

    if name in ("credit_default", "taiwan_credit_default"):
        ds = fetch_ucirepo(name="default of credit card clients")
        Xdf = ds.data.features.copy()
        ydf = ds.data.targets.copy()
        col = ydf.columns[0]
        y_bin = _to_binary_series(ydf[col])
        df = Xdf.join(y_bin.rename("target"))
        categorical = [c for c in Xdf.columns if Xdf[c].dtype == "object"]
        if categorical:
            df = merge_rare_categories(df, categorical, min_freq=0.01)
        X, y, feat = _onehot_df(df, target="target")
        return X, y.astype(int), feat, {"problem": "clf", "name": "credit_default"}

    if name in ("wine_quality_red", "wine_red"):
        ds = fetch_ucirepo(name="wine quality")
        Xdf = ds.data.features.copy()
        ydf = ds.data.targets.copy()
        df_full = Xdf.join(ydf.rename(columns={ydf.columns[0]: "quality"}))
        if "type" in df_full.columns:
            df_red = df_full[df_full["type"].astype(str).str.lower().str.contains("red")]
            X = df_red.drop(columns=["quality", "type"]).values.astype(float)
            y = df_red["quality"].values.astype(float)
            feat = [c for c in df_red.columns if c not in ("quality", "type")]
        else:
            X = Xdf.values.astype(float)
            y = ydf[ydf.columns[0]].values.astype(float)
            feat = list(Xdf.columns)
        return X, y, feat, {"problem": "reg", "name": "wine_quality_red"}

    if name in ("wine_quality_white", "wine_white"):
        ds = fetch_ucirepo(name="wine quality")
        Xdf = ds.data.features.copy()
        ydf = ds.data.targets.copy()
        df_full = Xdf.join(ydf.rename(columns={ydf.columns[0]: "quality"}))
        if "type" in df_full.columns:
            df_white = df_full[df_full["type"].astype(str).str.lower().str.contains("white")]
            X = df_white.drop(columns=["quality", "type"]).values.astype(float)
            y = df_white["quality"].values.astype(float)
            feat = [c for c in df_white.columns if c not in ("quality", "type")]
        else:
            X = Xdf.values.astype(float)
            y = ydf[ydf.columns[0]].values.astype(float)
            feat = list(Xdf.columns)
        return X, y, feat, {"problem": "reg", "name": "wine_quality_white"}

    if name == "abalone":
        ds = fetch_ucirepo(name="abalone")
        Xdf = ds.data.features.copy()
        ydf = ds.data.targets.copy()
        target_col = "Rings" if "Rings" in ydf.columns else ydf.columns[0]
        df = Xdf.join(ydf[target_col].rename("target"))
        categorical = [c for c in Xdf.columns if Xdf[c].dtype == "object"]
        if categorical:
            df = merge_rare_categories(df, categorical, min_freq=0.01)
        X, y, feat = _onehot_df(df, target="target")
        return X, y.astype(float), feat, {"problem": "reg", "name": "abalone"}

    if name == "year_prediction_msd":
        import urllib.request
        import zipfile
        import io

        # 데이터셋 URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"

        print("Downloading and extracting YearPredictionMSD dataset...")
        # URL에서 직접 데이터를 다운로드하고 압축을 해제.
        r = urllib.request.urlopen(url).read()
        file = zipfile.ZipFile(io.BytesIO(r), 'r')
        csv_path = file.namelist()[0] # 압축 파일 내의 CSV 파일 경로

        with file.open(csv_path) as f:
            df = pd.read_csv(f, header=None)
        print("Dataset loaded.")

        y = df.iloc[:, 0].values.astype(float)
        X = df.iloc[:, 1:].values.astype(float)

        num_features = X.shape[1]
        feat = [f'feature_{i+1}' for i in range(num_features)]

        meta = {"problem": "reg", "name": "year_prediction_msd"}

        return X, y, feat, meta

    if name == "custom_test_data":
        target = "MedHouseVal"
        csv = pd.read_csv(Path(__file__).parent / "scripts" / "data" / "custom_test_data.csv")
        X = csv.drop(columns=[target]).values
        y = csv[target].values.astype(float)
        feat = [c for c in csv.columns if c != target]

        def _infer_problem_from_y(y: np.ndarray, max_classes_for_clf: int = 20) -> str:
            y = np.asarray(y)
            if y.ndim != 1:
                return "reg"
            u = np.unique(y[~np.isnan(y)]) if np.issubdtype(y.dtype, np.floating) else np.unique(y)
            k = len(u)
            n = len(y)

            # 이진이면 분류
            if k == 2:
                return "clf"

            # 클래스가 너무 많으면 회귀로 간주
            # (예: 정수형이지만 사실상 연속형 라벨)
            if k > max_classes_for_clf:
                return "reg"

            # 클래스 수가 3~20이면: 현재 STRIDE 구현은 이진 분류만 안정적이므로
            # 1) 모델이 멀티클래스 proba를 주면 pos_class를 하나 고르거나
            # 2) 회귀로 취급
            # 기본은 "reg"로 돌리는 게 안정적
            return "reg"

        def _resolve_problem(y: np.ndarray | None) -> str:
            if y is not None:
                return _infer_problem_from_y(y)
            # 마지막 안전망: 회귀로
            return "reg"

        problem = _resolve_problem(y)
        return X, y, feat, {"problem": problem, "name": "custom_test_data"}

    raise ValueError(f"Unknown DATASET: {DATASET}")