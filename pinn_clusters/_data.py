from scipy.io import savemat
import numpy as np

class ArrayRecorder:
    """
    A class to store a series of NumPy arrays in a dictionary and save them to disk.
    
    Attributes:
        arrays (dict): A dictionary to store the arrays with their associated keys.
    """

    def __init__(self):
        """
        Initializes an ArrayRecorder instance with an empty dictionary for storing arrays.
        """
        self.arrays = {}

    def add(self, key: str, array: np.array):
        """
        Adds a new array to the dictionary with a specified key.

        Args:
            key (str): The key associated with the array to be added.
            array (np.array): The NumPy array to be stored.

        Raises:
            RuntimeError: If the specified key already exists in the dictionary.
        """
        if key in self.arrays:
            msg = f"{self.__class__.__name__} add key: '{key}' already has a record."
            raise RuntimeError(msg)

        self.arrays[key] = np.array(array)

    def to_mat(self, out_f: str):
        """
        Saves the stored arrays to a MATLAB .mat file.

        Args:
            out_f (str): The path of the output .mat file.
        """
        savemat(out_f, self.arrays, oned_as="row")
import numpy as np

def add_noise(data: np.array, ratio: float) -> np.array:
    """
    Adds Gaussian noise to the input data.

    Args:
        data (np.array): The input data array to which noise is to be added.
        ratio (float): The standard deviation of the Gaussian noise as a fraction of the data value.

    Returns:
        np.array: The data array with added noise.

    Raises:
        ValueError: If the noise ratio is not within the range [0, 1].
    """
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"Noise ratio must be within [0, 1], got {ratio}")

    noise = np.random.normal(0, ratio, data.shape)
    return data + data * noise

def random_sample(n: int, x_star, *args):
    """
    Randomly samples data points from the given datasets.

    Args:
        n (int): The number of data points to sample.
        x_star: The x-location of training data to be sampled.
        *args: Additional arrays of training data to be sampled alongside x_star.

    Returns:
        A tuple containing sampled points from x_star and each array in args.

    Raises:
        ValueError: If the number of samples requested is less than 1 or greater than the size of data.
        AssertionError: If the shape of x_star or any of the args does not match the expected format.

    Note:
        It is assumed that x_star and all arrays in args have a size of 401 in the relevant dimension.
    """
    assert x_star.shape[0] == 401, "x_star should have 401 data points in the first dimension."
    for arg in args:
        assert arg.shape[-1] == 401, "Each arg should have 401 data points in the last dimension."

    result = []
    if n < 1:
        raise ValueError(f"Invalid value for sampling data: try sample {n} from 401 data poitns")
    elif n == 1:
        for arg in args:
            result.append(arg[:, [0]])
        return x_star[[0]], *result
    else:
        sample_n = n - 1
        sample_range = np.arange(1, 401)
        np.random.shuffle(sample_range)
        sampled = [0] + list(sample_range[:sample_n])
        sampled.sort()

    for arg in args:
        result.append(arg[..., sampled])
    return x_star[sampled], *result

