"""Oracle actual-time linear extrapolation and label-free time-only regression."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from training.common import TrainStandardizer, clip_predictions


class LinearGrowth:
    condition = "Oracle-phenotype"

    def predict(self, history, history_times, forecast_delta_days):
        times = np.asarray(history_times, dtype=np.float64)
        centered_times = times - times.mean(axis=1,keepdims=True)
        centered_history = history - history.mean(axis=1,keepdims=True)
        denominator = np.sum(centered_times ** 2,axis=1)
        if np.any(denominator <= 0):
            raise ValueError("History must span positive elapsed time")
        slope = np.sum(centered_history * centered_times[:,:,None],axis=1) / denominator[:,None]
        future_time = times[:, -1] + np.asarray(forecast_delta_days)
        predicted = history.mean(axis=1) + slope * (future_time-times.mean(axis=1))[:,None]
        return clip_predictions(predicted)


class TimeOnly:
    condition = "Time-only; elapsed observation age, not sowing age"

    def __init__(self, alpha: float = 10):
        self.alpha = alpha
        self.polynomial = PolynomialFeatures(degree=2,include_bias=False)
        self.regression = Ridge(alpha=alpha,solver="svd")

    def fit(self, times, targets, rows):
        self.time_scaler = TrainStandardizer.fit(times,rows)
        self.target_scaler = TrainStandardizer.fit(targets,rows)
        design = self.polynomial.fit_transform(self.time_scaler.transform(times).astype(np.float64))
        counts = rows.groupby("plant_uid").window_id.transform("count").to_numpy()
        sample_weight = len(rows) / (rows.plant_uid.nunique() * counts)
        self.regression.fit(design,self.target_scaler.transform(targets).astype(np.float64),sample_weight=sample_weight)
        self.regression.coef_ = np.ascontiguousarray(self.regression.coef_)
        self.regression.intercept_ = np.ascontiguousarray(self.regression.intercept_)
        return self

    def predict(self, times):
        design = self.polynomial.transform(self.time_scaler.transform(times).astype(np.float64))
        return clip_predictions(self.target_scaler.inverse(self.regression.predict(design)))

    def state(self):
        return {"model":"B1b","alpha":self.alpha,"degree":2,"solver":"svd","parameter_dtype":str(self.regression.coef_.dtype),
                "coefficients":self.regression.coef_.tolist(),"intercept":self.regression.intercept_.tolist(),
                "polynomial_powers":self.polynomial.powers_.tolist(),
                "time_scaler":self.time_scaler.state(),"target_scaler":self.target_scaler.state()}

    @classmethod
    def from_state(cls, state):
        model = cls(state["alpha"])
        model.time_scaler = TrainStandardizer.from_state(state["time_scaler"])
        model.target_scaler = TrainStandardizer.from_state(state["target_scaler"])
        model.polynomial.fit(np.zeros((1,len(model.time_scaler.mean))))
        if model.polynomial.powers_.tolist() != state["polynomial_powers"]:
            raise ValueError("Polynomial checkpoint compatibility mismatch")
        model.regression.coef_ = np.asarray(state["coefficients"],dtype=state["parameter_dtype"])
        model.regression.intercept_ = np.asarray(state["intercept"],dtype=state["parameter_dtype"])
        return model
