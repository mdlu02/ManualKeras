import numpy as np
from .core import Diffable


class Dense(Diffable):


    def __init__(self, input_size, output_size, initializer="kaiming"):
        super().__init__()
        self.w, self.b = self.__class__._initialize_weight(
            initializer, input_size, output_size
        )
        self.weights = [self.w, self.b]
        self.inputs  = None
        self.outputs = None

    def forward(self, inputs):
        """Forward pass for a dense layer"""
        self.inputs = inputs
        self.outputs = np.matmul(self.inputs, self.w) + self.b
        return self.outputs

    def weight_gradients(self):
        """Calculating the gradients wrt weights and biases!"""
        x = np.ones_like(self.weights[0]) * np.expand_dims(self.inputs, axis=-1)
        b = np.ones_like(self.weights[1])
        return x, b

    def input_gradients(self):
        """Calculating the gradients wrt inputs!"""
        return self.weights[0]

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size):
        """
        Initializes the values of the weights and biases.
        """
        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"
        io_size = (input_size, output_size)
        bias = np.zeros(io_size[1])
        
        if initializer == "zero":
            weights = np.zeros(io_size)

        elif initializer == "normal":
            weights = np.random.normal(0.0, 1, size=io_size)
        elif initializer == "xavier":
            weights = np.random.normal(0, np.sqrt(2/(io_size[0]+io_size[1])), size=io_size)
        elif initializer == "kaiming":
            weights = np.random.normal(0, np.sqrt(2/io_size[0]), size=io_size)

        return weights, bias
