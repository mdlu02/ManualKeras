
import numpy as np

from .core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        """Categorical accuracy forward pass"""
        super().__init__()
        try:
            total, count = probs.shape[0], 0
        except:
            total, count = len(probs), 0
        for i in range(total):
            if np.argmax(probs[i], axis=0) == np.argmax(labels[i], axis=0):
                count += 1
        return count / total