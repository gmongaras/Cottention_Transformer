import numpy as np

def read_tensor_from_file(filename):
    with open(filename, 'rb') as file:
        dimensions = np.fromfile(file, dtype=np.int32, count=1)[0]
        shape = np.fromfile(file, dtype=np.int32, count=dimensions)
        remaining_size = np.product(shape)
        data = np.fromfile(file, dtype=np.float32, count=remaining_size)  # Specify count to ensure reading the expected number of elements
        tensor = data.reshape(shape)
    return tensor


# Load the tensors
# K = read_tensor_from_file('K.bin')
# V = read_tensor_from_file('V.bin')
Q = read_tensor_from_file('Q.bin')
KV = read_tensor_from_file('KV.bin')
output = read_tensor_from_file('output.bin')

print()