"""B0: oracle-mask historical phenotype persistence."""

import numpy as np

from training.common import clip_predictions


class Persistence:
    condition = "Oracle-phenotype"

    def predict(self, history: np.ndarray, **kwargs):
        if history.ndim != 3 or history.shape[1:] != (3,2):
            raise ValueError("Expected batch of three two-target historical phenotypes")
        return clip_predictions(history[:, -1])
