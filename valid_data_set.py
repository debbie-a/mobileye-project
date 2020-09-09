import os
import numpy as np
import matplotlib.pyplot as plt


def validate_data_set():

    root = "data_dir\\"
    data_path = os.path.join(root, "train\\data.bin")
    label_path = os.path.join(root, "train\\labels.bin")

    size = 81 * 81 * 3
    index = 0
    image = np.memmap(data_path, dtype='uint8', mode='r', offset=index * size, shape=(81, 81, 3))
    is_tfl = np.memmap(label_path, dtype='uint8', mode='r', offset=index, shape=(1))

    if is_tfl:
        print("yes tfl")
    else:
        print("not tfl")

    plt.imshow(image)
    plt.show()


