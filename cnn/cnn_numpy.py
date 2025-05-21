import numpy as np


def conv(
    matrix: np.ndarray,
    kernel: np.ndarray,
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 1,
):  # only 1 channel
    if padding >= kernel_size:
        return None

    padding_matrix = np.zeros(
        (matrix.shape[0] + 2 * padding, matrix.shape[1] + 2 * padding)
    )
    conv_h = int(matrix.shape[0] + 2 * padding - kernel_size) // stride + 1
    conv_w = int(matrix.shape[1] + 2 * padding - kernel_size) // stride + 1
    conv_matrix = np.zeros((conv_h, conv_w))

    if padding > 0:
        padding_matrix[padding:-padding, padding:-padding] = matrix
    else:
        padding_matrix = matrix.copy()

    width, height = padding_matrix.shape

    for x_idx in range(0, width - kernel_size + 1, stride):
        for y_idx in range(0, height - kernel_size + 1, stride):
            sum_num = np.sum(
                padding_matrix[x_idx : x_idx + kernel_size, y_idx : y_idx + kernel_size]
                * kernel
            )
            relu_data = ReLU(sum_num)
            conv_matrix[x_idx - stride + 1, y_idx - stride + 1] = relu_data

    return conv_matrix


def max_pooling(matrix: np.ndarray, pooling_size: int = 2, stride: int = 2):
    pooling_h = (matrix.shape[0] - pooling_size) // stride + 1
    pooling_w = (matrix.shape[1] - pooling_size) // stride + 1
    pooling_matrix = np.zeros((pooling_h, pooling_w))

    for x_idx in range(0, matrix.shape[0] - pooling_size + 1, stride):
        for y_idx in range(0, matrix.shape[1] - pooling_size + 1, stride):
            pooling_matrix[x_idx // stride, y_idx // stride] = np.max(
                matrix[x_idx : x_idx + pooling_size, y_idx : y_idx + pooling_size]
            )

    return pooling_matrix


def ReLU(num: float):
    return max(0, num)


def mnist_predict_matrix_test():
    origin_matrix = np.random.random(size=(28, 28))
    print(f"origin_matrix shape: {origin_matrix.shape}")

    tmp_kernel_list = []
    for _ in range(32):
        tmp_kernel = np.random.random(size=(3, 3))
        tmp_kernel_list.append(tmp_kernel)

    conv_matrix_list = []
    for kernel in tmp_kernel_list:
        tmp_conv_matrix = conv(
            matrix=origin_matrix, kernel=kernel, kernel_size=3, padding=1, stride=1
        )
        conv_matrix_list.append(tmp_conv_matrix)
    conv_matrix = np.stack(conv_matrix_list, axis=2)
    print(f"conv_matrix shape: {conv_matrix.shape}")

    pooling_matrix_list = []
    for idx in range(conv_matrix.shape[2]):
        tmp_pooling_matrix = max_pooling(
            matrix=conv_matrix[:, :, idx], pooling_size=2, stride=2
        )
        pooling_matrix_list.append(tmp_pooling_matrix)

    pooling_matrix = np.stack(pooling_matrix_list, axis=2)
    print(f"pooling_matrix shape: {pooling_matrix.shape}")

    tmp_kernel_list.clear()
    conv_matrix_list.clear()
    for _ in range(2):
        tmp_kernel = np.random.random(size=(3, 3))
        tmp_kernel_list.append(tmp_kernel)

    for idx in range(pooling_matrix.shape[2]):
        for kernel in tmp_kernel_list:
            tmp_conv_matrix = conv(
                matrix=pooling_matrix[:, :, idx],
                kernel=kernel,
                kernel_size=3,
                padding=1,
                stride=1,
            )
            conv_matrix_list.append(tmp_conv_matrix)

    conv_matrix = np.stack(conv_matrix_list, axis=2)
    print(f"conv_matrix shape: {conv_matrix.shape}")

    pooling_matrix_list.clear()
    for idx in range(conv_matrix.shape[2]):
        tmp_pooling_matrix = max_pooling(
            matrix=conv_matrix[:, :, idx], pooling_size=2, stride=2
        )
        pooling_matrix_list.append(tmp_pooling_matrix)

    pooling_matrix = np.stack(pooling_matrix_list, axis=2)
    print(f"pooling_matrix shape: {pooling_matrix.shape}")

    flatten_matrix = pooling_matrix.flatten()
    print(f"flatten_matrix shape: {flatten_matrix.shape}")

    tmp_flatten_kernel = np.random.random(size=(flatten_matrix.shape[0], 128))
    print(f"tmp_flatten_kernel shape: {tmp_flatten_kernel.shape}")

    dot_matrix: np.ndarray = np.dot(flatten_matrix, tmp_flatten_kernel)
    print(f"dot_matrix shape: {dot_matrix.shape}")

    tmp_softmax_kernel = np.random.random(size=(128, 10))
    print(f"tmp_softmax_kernel shape: {tmp_softmax_kernel.shape}")

    softmax_matrix: np.ndarray = np.dot(dot_matrix, tmp_softmax_kernel)
    print(f"softmax_matrix shape: {softmax_matrix.shape}")

    print(softmax_matrix)


if __name__ == "__main__":
    mnist_predict_matrix_test()
