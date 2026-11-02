"""Age-normative modelling of biomarkers.

EEG slows with normal ageing, and age alone already predicts the CAUEEG label at ~55%.
A raw theta/alpha ratio therefore mixes "old" with "impaired". Here every biomarker is
regressed on age using ONLY the Normal recordings of the TRAIN split, and replaced by its
deviation from the age-expected value, in robust SD units:

    z = (x - f_normal(age)) / (1.4826 * MAD of the Normal residuals)

so the model sees "how abnormal is this EEG for someone this age". Age itself is passed
to the network separately, which keeps the two contributions apart (and ablatable).
"""
import numpy as np

META_PREFIX = "_"
INDICATORS = ("has_eo", "has_photic")


def feature_columns(df):
    return [c for c in df.columns if c != "serial" and not c.startswith(META_PREFIX)]


class NormativeModel:
    """Per-feature polynomial age fit on train Normals (normative=True) or plain train standardisation."""

    def __init__(self, normative=True, degree=1, z_clip=6.0):
        self.normative, self.degree, self.z_clip = normative, degree, z_clip

    def _design(self, age):
        a = (np.asarray(age, np.float64) - self.age_mu) / self.age_sd
        return np.stack([a ** k for k in range(self.degree + 1)], axis=1)

    def fit(self, X, age, labels, names, normal_label=0):
        X = np.asarray(X, np.float64)
        self.names = list(names)
        self.age_mu, self.age_sd = float(np.mean(age)), float(np.std(age) + 1e-9)
        ref = (np.asarray(labels) == normal_label) if self.normative else np.ones(len(X), bool)
        A = self._design(np.asarray(age)[ref])
        self.coef = np.zeros((X.shape[1], self.degree + 1))
        self.scale = np.ones(X.shape[1])
        for j, n in enumerate(self.names):
            col = X[ref, j]
            ok = np.isfinite(col)
            if n in INDICATORS or ok.sum() < 10:
                continue
            if self.normative:
                self.coef[j] = np.linalg.lstsq(A[ok], col[ok], rcond=None)[0]
            else:
                self.coef[j, 0] = col[ok].mean()
            res = col[ok] - A[ok] @ self.coef[j]
            mad = 1.4826 * np.median(np.abs(res - np.median(res)))
            self.scale[j] = mad if mad > 1e-9 else (res.std() + 1e-9)
        return self

    def transform(self, X, age):
        X = np.asarray(X, np.float64)
        Z = (X - self._design(age) @ self.coef.T) / self.scale
        for j, n in enumerate(self.names):
            if n in INDICATORS:
                Z[:, j] = X[:, j]
        Z = np.clip(Z, -self.z_clip, self.z_clip)
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)  # missing state -> "typical for age"

    def age_slopes(self):
        """Per-feature normal-ageing slope (in feature units per SD of age) - for the report."""
        return {n: float(self.coef[j, 1]) if self.degree >= 1 else 0.0 for j, n in enumerate(self.names)}


def build_feature_matrices(features_df, splits, fcfg):
    """Fit the normative model on train and return {split: (Z, age, y, serials)} + model."""
    names = feature_columns(features_df)
    table = features_df.set_index("serial")

    def raw(split):
        rows = splits[split]
        s = [r["serial"] for r in rows]
        return (table.loc[s, names].to_numpy(np.float64), np.array([r["age"] for r in rows]),
                np.array([r["label"] for r in rows]), s)

    Xtr, atr, ytr, _ = raw("train")
    model = NormativeModel(fcfg["normative"], fcfg["normative_degree"], fcfg["z_clip"]).fit(Xtr, atr, ytr, names)
    out = {}
    for split in splits:
        X, a, y, s = raw(split)
        out[split] = {"Z": model.transform(X, a), "age": a, "y": y, "serials": s}
    return out, names, model
