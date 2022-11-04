from types import SimpleNamespace
import numpy as np
import ManualKeras


class SequentialModel(ManualKeras.Model):
    """
    Implemented in ManualKeras/model.py
    """

    def call(self, inputs):
        """
        Forward pass in sequential model. It's helpful to note that layers are initialized in ManualKeras.Model, and
        you can refer to them with self.layers. You can call a layer by doing var = layer(input).
        """
        x = np.copy(inputs)
        for layer in self.layers:
                x = layer(x)
        return x

    def batch_step(self, x, y, training=True):
        """
        Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model! 
        Most of this method (forward, loss, applying gradients)
        will take place within the scope of Beras.GradientTape()
        """
        # If training, then also update the gradients according to the optimizer
        with ManualKeras.GradientTape() as tape:
            preds = self.call(x)
            loss = self.compiled_loss(preds, y)
            if training == True:
                self.optimizer.apply_gradients(self.trainable_variables, tape.gradient())
        accuracy = self.compiled_acc(preds, y)
        return {"loss": loss, "acc": accuracy}


def get_simple_model_components():
    """
    Returns a simple single-layer model.
    """
    from ManualKeras.activations import Softmax
    from ManualKeras.layers import Dense
    from ManualKeras.metrics import CategoricalAccuracy
    from ManualKeras.losses import MeanSquaredError
    from ManualKeras.optimizers import Adam

    model = SequentialModel([Dense(784, 10, initializer='kaiming'), Softmax()])
    model.compile(
        optimizer=Adam(0.01),
        loss_fn=MeanSquaredError(),
        acc_fn=CategoricalAccuracy(),
    )
    return SimpleNamespace(model=model, epochs=10, batch_size=100)


def get_advanced_model_components():
    """
    Returns a multi-layered model with more involved components.
    """
    from ManualKeras.activations import Softmax, LeakyReLU
    from ManualKeras.layers import Dense
    from ManualKeras.losses import MeanSquaredError
    from ManualKeras.metrics import CategoricalAccuracy
    from ManualKeras.optimizers import Adam

    model = SequentialModel([Dense(784, 120), LeakyReLU(), Dense(120, 120), LeakyReLU(0.0025), Dense(120, 10), Softmax()])
    model.compile(
        optimizer=Adam(0.005),
        loss_fn=MeanSquaredError(),
        acc_fn=CategoricalAccuracy(),
    )

    return SimpleNamespace(model=model, epochs=8, batch_size=500)


if __name__ == "__main__":
    """
    Read in MNIST data and initialize/train/test your model.
    """
    from ManualKeras.onehot import OneHotEncoder
    import preprocess

    ## Read in MNIST data,
    train_inputs, train_labels = preprocess.get_data_MNIST("train", "../data")
    test_inputs,  test_labels  = preprocess.get_data_MNIST("test",  "../data")

    ohe = OneHotEncoder()  ## placeholder function: returns zero for a given input
    ohe.fit(np.concatenate([train_labels, test_labels], axis=-1))

    ## Get your model to train and test
    simple = False
    args = get_simple_model_components() if simple else get_advanced_model_components()
    model = args.model

    train_agg_metrics = model.fit(
        train_inputs, 
        ohe(train_labels), 
        epochs     = args.epochs, 
        batch_size = args.batch_size
    )

    test_agg_metrics = model.evaluate(test_inputs, ohe(test_labels), batch_size=100)
    print('Testing Performance:', test_agg_metrics)
