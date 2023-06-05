import random
from os import path

import numpy as np


def load_dataset(dataset_dir):
    xs_name = "uspsdata.txt"
    ys_name = "uspscl.txt"
    xs = read_images(path.join(dataset_dir, xs_name))
    ys = read_labels((path.join(dataset_dir, ys_name)))
    return xs, ys


def read_images(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    images = []
    for line in lines:
        image = [float(p) for p in line.split("\t")]
        image = np.array(image).reshape((16, 16))
        image = normalize(image)
        images.append(image)

    return np.array(images)


def read_labels(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    labels = [int(line.strip()) for line in lines]
    return np.array(labels)


def shuffled_dataset(dataset):
    x = list(zip(*dataset))
    random.shuffle(x)
    images, labels = zip(*x)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def shuffle_split_dataset(dataset):
    m = len(dataset[0])
    xs, ys = shuffled_dataset(dataset)
    train_len = int(m * 0.6)
    test_lim = train_len + int(m * 0.2)
    train_set = xs[:train_len], ys[:train_len]
    test_set = xs[train_len: test_lim], ys[train_len: test_lim]
    valid_set = xs[test_lim:], ys[test_lim:]
    return train_set, test_set, valid_set


def normalize(image: np.ndarray):
    return (image - np.min(image)) / (np.max(image) - np.min(image))
