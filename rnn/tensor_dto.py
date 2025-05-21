from torch import Tensor


class TensorDto:
    x_train_tensor: Tensor
    y_train_tensor: Tensor
    x_test_tensor: Tensor
    y_test_tensor: Tensor

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train_tensor = x_train
        self.y_train_tensor = y_train
        self.x_test_tensor = x_test
        self.y_test_tensor = y_test
