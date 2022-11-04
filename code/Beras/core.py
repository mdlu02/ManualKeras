from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


## DO NOT MODIFY THIS CLASS
class Callable(ABC):
    """
    Callable Sub-classes:
     - CategoricalAccuracy (./metrics.py)
     - OneHotEncoder       (./preprocess.py)
     - Diffable            (.)
    """

    def __call__(self, *args, **kwargs) -> np.array:
        """Lets `self()` and `self.forward()` be the same"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.array:
        """Pass inputs through function. Can store inputs and outputs as instance variables"""
        pass


class Diffable(Callable):
    """
    Diffable Sub-classes:
     - Dense            (./layers.py)
     - LeakyReLU, ReLU  (./activations.py)
     - Softmax          (./activations.py)
     - MeanSquaredError (./losses.py)
    """

    """Stores whether the operation being used is inside a gradient tape scope"""
    gradient_tape = None  ## All-instance-shared variable

    def __init__(self):
        """Is the layer trainable"""
        super().__init__()
        self.trainable = True  ## self-only instance variable

    def __call__(self, *args, **kwargs) -> np.array:
        """
        If there is a gradient tape scope in effect, perform and record the operation.
        Otherwise... just perform the operation and don't let the gradient tape know.
        """
        if Diffable.gradient_tape is not None:
            Diffable.gradient_tape.operations += [self]
        return self.forward(*args, **kwargs)

    @abstractmethod
    def input_gradients(self: np.array) -> np.array:
        """Returns gradient for input (this part gets specified for all diffables)"""
        pass

    def weight_gradients(self: np.array) -> Tuple[np.array, np.array]:
        """Returns gradient for weights (this part gets specified for SOME diffables)"""
        return ()

    def compose_to_input(self, J: np.array) -> np.array:
        """
        Compose the inputted cumulative jacobian with the input jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `input_gradients` to provide either batched or overall jacobian.
        Assumes input/cumulative jacobians are matrix multiplied
        """
        ig = self.input_gradients()
        batch_size = J.shape[0]
        n_out, n_in = ig.shape[-2:]
        j_new = np.zeros((batch_size, n_out), dtype=ig.dtype)
        for b in range(batch_size):
            ig_b = ig[b] if len(ig.shape) == 3 else ig
            j_new[b] = ig_b @ J[b]
        return j_new

    def compose_to_weight(self, J: np.array) -> list:
        """
        Compose the inputted cumulative jacobian with the weight jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `weight_gradients` to provide either batched or overall jacobian.
        Assumes weight/cumulative jacobians are element-wise multiplied (w/ broadcasting)
        and the resulting per-batch statistics are averaged together for avg per-param gradient.
        """
        # print(f'Composing to weight in {self.__class__.__name__}')
        assert hasattr(
            self, "weights"
        ), f"Layer {self.__class__.__name__} cannot compose along weight path"
        J_out = []
        ## For every weight/weight-gradient pair...
        for w, wg in zip(self.weights, self.weight_gradients()):
            batch_size = J.shape[0]
            ## Make a cumulative jacobian which will contribute to the final jacobian
            j_new = np.zeros((batch_size, *w.shape), dtype=wg.dtype)
            ## For every element in the batch (for a single batch-level gradient updates)
            for b in range(batch_size):
                ## If the weight gradient is a batch of transform matrices, get the right entry.
                ## Allows gradient methods to give either batched or non-batched matrices
                wg_b = wg[b] if len(wg.shape) == 3 else wg
                ## Update the batch's Jacobian update contribution
                j_new[b] = wg_b * J[b]
            ## The final jacobian for this weight is the average gradient update for the batch
            J_out += [np.mean(j_new, axis=0)]
        ## After new jacobian is computed for each weight set, return the list of gradient updatates
        return J_out


class GradientTape:

    def __init__(self):
        ## Log of operations that were performed inside tape scope
        self.operations = []

    def __enter__(self):
        # When tape scope is entered, let Diffable start recording to self.operation
        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, stop letting Diffable record
        Diffable.gradient_tape = None

    def gradient(self) -> list:
        """Get the gradient from first to last recorded operation"""
        grads = []
        flipped_ops = np.flip(self.operations)
        grad = flipped_ops[0].input_gradients()
        for i in range(1, flipped_ops.shape[0]):
            if flipped_ops[i].trainable and hasattr(flipped_ops[i], "weights"):
                temp = flipped_ops[i].compose_to_weight(grad)
                grads = temp + grads
            grad = flipped_ops[i].compose_to_input(grad)
        # print(np.max(grads[0]))
        # d
        return grads
