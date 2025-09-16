import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_cols, errors="ignore")


class EncodeAndScale(BaseEstimator, TransformerMixin):
    def __init__(self, encoder, scaler):
        self.encoder = encoder
        self.scaler = scaler
        self.expected_cols = None  # set later in fit()

    def fit(self, X, y=None):
        # set expected columns after encoder + scaler are fitted
        cat_encoded = self.encoder.transform(X[["Geography", "Gender"]])
        if hasattr(cat_encoded, "toarray"):
            cat_encoded = cat_encoded.toarray()
        cat_encoded = pd.DataFrame(
            cat_encoded,
            columns=self.encoder.get_feature_names_out(["Geography", "Gender"]),
            index=X.index,
        )

        numeric_cols = [c for c in X.columns if c not in ["Geography", "Gender"]]
        X_numeric = X[numeric_cols]

        X_combined = pd.concat([X_numeric, cat_encoded], axis=1)
        self.expected_cols = X_combined.columns.tolist()
        return self

    def transform(self, X):
        # encode categorical
        X_encoded = self.encoder.transform(X[["Geography", "Gender"]])
        if hasattr(X_encoded, "toarray"):
            X_encoded = X_encoded.toarray()
        X_encoded = pd.DataFrame(
            X_encoded,
            columns=self.encoder.get_feature_names_out(["Geography", "Gender"]),
            index=X.index,
        )

        numeric_cols = [c for c in X.columns if c not in ["Geography", "Gender"]]
        X_numeric = X[numeric_cols]

        X_combined = pd.concat([X_numeric, X_encoded], axis=1)

        # ensure correct column order
        X_final = X_combined.reindex(columns=self.expected_cols, fill_value=0)

        return self.scaler.transform(X_final)


class KerasWrapper(BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self  # model already trained

    def predict(self, X):
        preds = self.model.predict(X)
        return (preds > 0.5).astype(int).ravel()
