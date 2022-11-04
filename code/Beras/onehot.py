import numpy as np

from .core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        self.uniq = np.unique(data)
        self.vecs = np.eye(len(self.uniq))
        self.uniq2oh = {e : self.vecs[i] for i, e in enumerate(self.uniq)}

    def forward(self, data):
        if not hasattr(self, "uniq2oh"):
            self.fit(data)
        return np.array([self.uniq2oh[x] for x in data])

    def inverse(self, data):
        assert hasattr(self, "uniq"), \
            "forward() or fit() must be called before attempting to invert"
        return np.array([self.uniq[x == 1][0] for x in data])
