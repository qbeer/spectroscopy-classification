import pandas as pd
import numpy as np
import os
from multiprocessing import sharedctypes
from multiprocessing import Pool
import matplotlib.pyplot as plt

shared_array = None


def multi_run_wrapper(args):
    return read_images(*args)


def read_images(start, end, width, height, df):
    imgs = np.ctypeslib.as_array(shared_array)
    for n in range(start, end):
        for _x in range(width):
            for _y in range(height):
                imgs[n, _x, _y] = df.values[_y * width + _x, n + 2]


def extract(processors=4):
    global shared_array
    files = os.listdir(os.getcwd() + "/src/data/")
    images = []

    for file_name in files:
        if file_name.startswith('.'): continue
        df = pd.read_csv(os.getcwd() + "/src/data/" + file_name)
        image_per_spectrum = df.columns.size - 2
        n_spectrum = df.values.shape[0]
        df.columns = ['x', 'y'] + np.arange(image_per_spectrum).tolist()
        n_image = image_per_spectrum * n_spectrum
        width = df['x'].unique().shape[0]
        height = df['y'].unique().shape[0]

        result = np.ctypeslib.as_ctypes(np.zeros((n_spectrum, width, height)))
        shared_array = sharedctypes.RawArray(result._type_, result)

        batch_size = image_per_spectrum // processors
        items = [
            (i * batch_size, (i + 1) * batch_size if
             (i + 1) * batch_size < image_per_spectrum else image_per_spectrum,
             width, height, df) for i in range(processors)
        ]

        with Pool(processes=processors) as pool:
            pool.map(multi_run_wrapper, items)

        result = np.ctypeslib.as_array(shared_array)

        for ind, res in enumerate(result):
            plt.imsave(os.getcwd() + "/src/data/" +
                       file_name.replace(".txt", "_%d_.png" % ind), res)
