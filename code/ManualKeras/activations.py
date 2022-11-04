import numpy as np

from .core import Diffable


class LeakyReLU(Diffable):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(self.inputs * self.alpha, self.inputs)
        return self.outputs

    def input_gradients(self):
        grad = self.outputs
        for i in range(grad.shape[0]):
            if grad[i].shape:
                for j in range(grad[i].shape[0]):
                    if grad[i][j] > 0:
                        grad[i][j] = 1
                    else:
                        grad[i][j] = self.alpha
            else:
                if grad[i] > 0:
                    grad[i] = 1
                else:
                    grad[i] = self.alpha
        return grad

    def compose_to_input(self, J):
        return self.input_gradients() * J


class ReLU(LeakyReLU):
    def __init__(self):
        super().__init__(alpha=0)


class Softmax(Diffable):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        """Softmax forward pass!"""
        self.inputs = inputs
        e = np.exp(inputs - np.amax(inputs, axis=-1, keepdims=True))
        self.outputs = e / np.sum(e, axis=-1, keepdims=True)
        return self.outputs

    def input_gradients(self):
        """Softmax backprop!"""
        batch_size, out = self.inputs.shape
        # Resize based on batch size, use np.eye to accound for special case when
        # row == col
        grad = np.resize(np.eye(out), (batch_size, out, out))
        # broadcasts and subtracts outputs from np.eye to account for both piecewise cases
        grad -= np.expand_dims(self.outputs, -1)
        # multiply to get final
        grad *= np.expand_dims(self.outputs, 1)
        return grad