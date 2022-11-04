import gzip
import pickle

import numpy as np
from tqdm import tqdm


def get_data_MNIST(subset, data_path="../data"):
    """
    Takes in a subset of data ("train" or "test"), unzips the inputs and labels files,
    normalizes the inputs, and returns (NumPy array of inputs, NumPy array of labels).

    :param subset: string to indicate which subset of data to get ("train" or "test")
    :param data_path: folder containing the MNIST data
    :return:
        inputs (NumPy array of float32)
        labels (NumPy array of uint8)
    """
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    inputs_file_path, labels_file_path, num_examples = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
    }[subset]
    inputs_file_path = f"{data_path}/mnist/{inputs_file_path}"
    labels_file_path = f"{data_path}/mnist/{labels_file_path}"

    
    # Defines number to read for training and testing sets.
    if subset == 'train':
        num_read = 60000
    else:
        num_read = 10000

    # unzips and reads data file, making sure to skip the 16 byte header
    with open(inputs_file_path,'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        buffer = bytestream.read(num_read * 784)

    # convert to np.asarray, change dtype to np.float32, normalize, and reshape for inputs
    image = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
    image /= 255.0
    image = image.reshape(num_read, 784)

    # unzips and reads labels file, making sure to skip the 8 byte header
    with open(labels_file_path,'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        buffer = bytestream.read(num_read)

    # converts to np.asarray with dtype np.float32
    label = np.frombuffer(buffer, dtype=np.uint8)

    return image, label



def shuffle_data(image_full, label_full, seed):
    """
    Shuffles the full dataset with the given random seed.

    :param: the dataset before shuffling
    :return: the dataset after shuffling
    """
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(np.arange(len(image_full)))
    image_full = image_full[shuffled_index]
    label_full = label_full[shuffled_index]
    return image_full, label_full


def get_specific_class(image_full, label_full, specific_class=0, num=None):
    """
    The MNIST dataset includes all ten digits, but they are not sorted,
        and it does not have the same number of images for each digits.

    :param image_full: the image array returned by the get_data function
    :param label_full: the label array returned by the get_data function
    :param specific_class: the specific class you want
    :param num: number of the images and labels to return
    :return image: Numpy array of inputs (float32)
    :return label: Numpy array of labels
                   (either uint8 or string, whichever type it was originally)
    """

    full_data = image_full
    full_labels = label_full

    # blank for new data
    image = np.asarray([])
    label = np.asarray([])

    # while loop to gather num data for speicifc_class
    i = 0
    while image.shape[0] < num:
        if full_labels[i] == specific_class:
            label = np.append(label, full_labels[i]) if label.size else np.asarray([full_labels[i]])
            image = np.concatenate([image, [full_data[i]]]) if image.size else np.asarray([full_data[i]])
        i += 1
        
    return image, label


def get_subset(image_full, label_full, class_list=list(range(10)), num=100):
    """
    The MNIST dataset includes all ten digits, but they are not sorted,
        and it does not have the same number of images for each digits.


    :param image: the image array returned by the get_data function
    :param label: the label array returned by the get_data function
    :param class_list: the list of specific classes you want
    :param num: number of the images and labels to return for each class
    :return image: Numpy array of inputs (float32)
    :return label: Numpy array of labels
                   (either uint8 or string, whichever type it was originally)
    """
    image = np.asarray([])
    label = np.asarray([])

    for c in tqdm(class_list):
        i, l = get_specific_class(image_full, label_full, c, num)
        image = np.vstack((image, i)) if image.size else i
        label = np.append(label, l) if label.size else np.asarray([l])

    return image, label
