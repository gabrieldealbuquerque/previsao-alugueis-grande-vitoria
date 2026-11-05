import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from xgboost import XGBRegressor

from src.preprocessamento import GroupAwareTargetPreprocessor, _model_area


XGBOOST_DEVICE = os.getenv("XGBOOST_DEVICE", "cpu")

def build_xgboost_regressor():
    """Única configuração de XGBoost usada pelo projeto."""
    return XGBRegressor(
        n_estimators=800,
        learning_rate=0.04,
        max_depth=7,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=4,
        reg_alpha=0.1,
        gamma=0,
        objective="reg:absoluteerror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=2,
        tree_method="hist",
        device=XGBOOST_DEVICE,
    )


class AreaScaledWeightedLogTargetRegressor(BaseEstimator, RegressorMixin):
    """XGBoost em log(preço/área^0,45), ponderado para aproximar o MAPE."""

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        frame = pd.DataFrame(X)
        target = np.asarray(y, dtype=float)
        area = _model_area(frame["area_m2"])
        scaled_target = np.log1p(target / np.maximum(area, 1) ** 0.45)
        self.preprocess_ = GroupAwareTargetPreprocessor()
        transformed = self.preprocess_.fit_transform(frame, scaled_target)
        self.regressor_ = clone(self.regressor)
        sample_weight = 1 / np.maximum(target, 1) ** 0.5
        sample_weight /= sample_weight.mean()
        self.regressor_.fit(transformed, scaled_target, sample_weight=sample_weight)
        return self

    def predict(self, X):
        frame = pd.DataFrame(X)
        area = _model_area(frame["area_m2"])
        scaled = np.expm1(self.regressor_.predict(self.preprocess_.transform(frame)))
        return np.maximum(scaled * np.maximum(area, 1) ** 0.45, 1)


class HouseComparableBlendRegressor(BaseEstimator, RegressorMixin):
    """Fallback global com comparáveis de casas, sem criar modelos por cidade."""

    def __init__(self, global_model):
        self.global_model = global_model

    @staticmethod
    def _weighted_median(values, weights):
        order = np.argsort(values)
        values = np.asarray(values)[order]
        weights = np.asarray(weights)[order]
        return values[np.searchsorted(np.cumsum(weights), weights.sum() / 2)]

    def fit(self, X, y):
        frame = pd.DataFrame(X).copy().reset_index(drop=True)
        target = np.asarray(y, dtype=float)
        self.global_model_ = clone(self.global_model).fit(frame, target)
        house = frame["tipo_base"].astype(str).eq("Casa").to_numpy()
        columns = ["cidade", "bairro_norm", "quartos", "banheiros", "vagas", "area_m2", "area_construida_m2"]
        self.house_frame_ = frame.loc[house, columns].reset_index(drop=True)
        self.house_target_ = target[house]
        return self

    def _comparable_prediction(self, row):
        candidates = self.house_frame_
        target = self.house_target_
        query_built = pd.notna(row.get("area_construida_m2"))
        candidate_built = pd.to_numeric(candidates["area_construida_m2"], errors="coerce").notna().to_numpy()
        branch = candidate_built if query_built else ~candidate_built
        if int(branch.sum()) >= 5:
            candidates = candidates.loc[branch].reset_index(drop=True)
            target = target[branch]
        distance = candidates["cidade"].astype(str).ne(str(row["cidade"])).to_numpy(float) * 20
        distance += candidates["bairro_norm"].astype(str).ne(str(row["bairro_norm"])).to_numpy(float) * 10
        rooms = pd.to_numeric(candidates["quartos"], errors="coerce").fillna(-1).to_numpy(float)
        query_rooms = -1 if pd.isna(row.get("quartos")) else float(row["quartos"])
        distance += np.abs(rooms - query_rooms) * 2
        bathrooms = pd.to_numeric(candidates["banheiros"], errors="coerce").fillna(-1).to_numpy(float)
        query_bathrooms = -1 if pd.isna(row.get("banheiros")) else float(row["banheiros"])
        distance += np.abs(bathrooms - query_bathrooms)
        parking = pd.to_numeric(candidates["vagas"], errors="coerce").fillna(-1).to_numpy(float)
        query_parking = -1 if pd.isna(row.get("vagas")) else float(row["vagas"])
        distance += np.abs(parking - query_parking)
        areas = np.log1p(pd.to_numeric(candidates["area_m2"], errors="coerce").fillna(0).to_numpy(float))
        raw_query_area = pd.to_numeric(pd.Series([row.get("area_m2")]), errors="coerce").iloc[0]
        query_area = np.log1p(float(raw_query_area) if pd.notna(raw_query_area) and raw_query_area > 0 else 0.0)
        distance += np.abs(areas - query_area) * 2
        selected = np.argsort(distance)[: min(3, len(distance))]
        weights = 1 / (0.2 + distance[selected])
        selected_areas = np.maximum(
            pd.to_numeric(candidates.iloc[selected]["area_m2"], errors="coerce").fillna(1).to_numpy(float),
            1,
        )
        query_area_value = float(raw_query_area) if pd.notna(raw_query_area) and raw_query_area > 0 else 1.0
        adjusted_target = target[selected] * (query_area_value / selected_areas) ** 0.2
        return self._weighted_median(adjusted_target, weights)

    def predict(self, X):
        frame = pd.DataFrame(X).copy().reset_index(drop=True)
        global_prediction = np.maximum(self.global_model_.predict(frame), 1)
        prediction = global_prediction.copy()
        house_indexes = np.flatnonzero(frame["tipo_base"].astype(str).eq("Casa").to_numpy())
        for index in house_indexes:
            comparable = max(float(self._comparable_prediction(frame.iloc[index])), 1)
            prediction[index] = np.exp(
                0.4 * np.log(global_prediction[index]) + 0.6 * np.log(comparable)
            )
        return prediction


class ApartmentComparableRegressor(BaseEstimator, RegressorMixin):
    """Mediana ponderada dos apartamentos mais parecidos do treino."""

    def fit(self, X, y):
        frame = pd.DataFrame(X).copy().reset_index(drop=True)
        target = np.asarray(y, dtype=float)
        mask = frame["tipo_base"].astype(str).eq("Apartamento").to_numpy()
        columns = ["cidade", "bairro_norm", "rua_norm", "quartos", "banheiros", "vagas", "area_m2", "condominio"]
        self.frame_ = frame.loc[mask, columns].reset_index(drop=True)
        self.target_ = target[mask]
        self.fallback_ = float(np.median(target))
        return self

    def _predict_one(self, row):
        candidates = self.frame_
        distance = candidates["cidade"].astype(str).ne(str(row["cidade"])).to_numpy(float) * 20
        distance += candidates["bairro_norm"].astype(str).ne(str(row["bairro_norm"])).to_numpy(float) * 3
        query_street = str(row.get("rua_norm", "Desconhecido"))
        if query_street != "Desconhecido":
            candidate_street = candidates["rua_norm"].astype(str)
            distance += (
                candidate_street.ne(query_street) & candidate_street.ne("Desconhecido")
            ).to_numpy(float)
        rooms = pd.to_numeric(candidates["quartos"], errors="coerce").fillna(-1).to_numpy(float)
        query_rooms = -1 if pd.isna(row.get("quartos")) else float(row["quartos"])
        distance += np.abs(rooms - query_rooms) * 0.5
        bathrooms = pd.to_numeric(candidates["banheiros"], errors="coerce").fillna(-1).to_numpy(float)
        query_bathrooms = -1 if pd.isna(row.get("banheiros")) else float(row["banheiros"])
        distance += np.abs(bathrooms - query_bathrooms)
        parking = pd.to_numeric(candidates["vagas"], errors="coerce").fillna(-1).to_numpy(float)
        query_parking = -1 if pd.isna(row.get("vagas")) else float(row["vagas"])
        distance += np.abs(parking - query_parking) * 2
        areas = np.maximum(pd.to_numeric(candidates["area_m2"], errors="coerce").fillna(1).to_numpy(float), 1)
        query_area = pd.to_numeric(pd.Series([row.get("area_m2")]), errors="coerce").fillna(1).iloc[0]
        query_area = max(float(query_area), 1)
        distance += np.abs(np.log1p(areas) - np.log1p(query_area)) * 10
        condos = pd.to_numeric(candidates["condominio"], errors="coerce").fillna(0).to_numpy(float)
        query_condo = pd.to_numeric(pd.Series([row.get("condominio")]), errors="coerce").fillna(0).iloc[0]
        distance += np.abs(np.log1p(condos) - np.log1p(max(float(query_condo), 0))) * 2
        selected = np.argsort(distance)[: min(5, len(distance))]
        weights = 1 / (0.2 + distance[selected])
        return HouseComparableBlendRegressor._weighted_median(self.target_[selected], weights)

    def predict(self, X):
        frame = pd.DataFrame(X).copy().reset_index(drop=True)
        result = np.full(len(frame), self.fallback_, dtype=float)
        for index in np.flatnonzero(frame["tipo_base"].astype(str).eq("Apartamento").to_numpy()):
            result[index] = self._predict_one(frame.iloc[index])
        return result


class ApartmentComparableBlendRegressor(BaseEstimator, RegressorMixin):
    """Média simples entre o modelo-base e os comparáveis de apartamentos."""

    def __init__(self, base_model):
        self.base_model = base_model

    def fit(self, X, y):
        self.base_model_ = clone(self.base_model).fit(X, y)
        self.apartment_model_ = ApartmentComparableRegressor().fit(X, y)
        return self

    def predict(self, X):
        frame = pd.DataFrame(X).copy().reset_index(drop=True)
        prediction = np.maximum(self.base_model_.predict(frame), 1)
        apartment = frame["tipo_base"].astype(str).eq("Apartamento").to_numpy()
        comparable = np.maximum(self.apartment_model_.predict(frame), 1)
        prediction[apartment] = (
            prediction[apartment] + comparable[apartment]
        ) / 2
        return prediction
