import matplotlib.pyplot as plt
import numpy as np

from dataset_utils import load_dataset, shuffle_split_dataset
import knn


def plot_images(images: np.ndarray):
    if len(images) == 0: return

    figure = plt.figure()
    for i in range(len(images)):
        ax = figure.add_subplot(1, len(images), i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(images[i])
    plt.show()


def get_missed_prediction_xs(dataset, predictions):
    xs, ys = dataset
    missed_mask = predictions != ys
    return xs[missed_mask]


def main():
    dataset = load_dataset("data")
    train, test, valid = shuffle_split_dataset(dataset)
    xs_test, ys_test = test

    print("(iv)")
    predictions = knn.predict(train, xs_test)
    print(f"Calculated error for 1 neighbor: {knn.calculate_error(ys_test, predictions)}")
    xs_missed = get_missed_prediction_xs(test, predictions)
    plot_images(xs_missed)

    print("(v)")
    for k in range(1, 14, 2):
        predictions = knn.predict(train, xs_test, k)
        print(f"Calculated error for {k} neighbor(s): {knn.calculate_error(ys_test, predictions)}")


if __name__ == '__main__':
    main()
