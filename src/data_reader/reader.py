import pandas as pd
import numpy as np
import os
from multiprocessing import sharedctypes
from multiprocessing import Pool


class DataReader:
    def __init__(self, data_folder_path):
        self.files = [os.listdir(os.getcwd() + data_folder_path)]
        self.images = []

    def extract(self, processors=4):
        for file_path in files:
            self.df = pd.read_csv(file_path)
            image_per_spectrum = self.df.columns.size - 2
            n_spectrum = self.df.values.shape[0]
            self.df.columns = ['x', 'y'
                               ] + np.arange(image_per_spectrum).tolist()
            n_image = image_per_spectrum * n_spectrum
            width = len(set(self.df['x'].values))
            height = len(set(self.df['y'].values))
            self.x = {
                _x: ind
                for ind, _x in enumerate(sorted(list(set(
                    self.df['x'].values))))
            }
            self.y = {
                _y: ind
                for ind, _y in enumerate(sorted(list(set(
                    self.df['y'].values))))
            }

            X = np.random.random((n_spectrum, width, height))
            result = np.ctypeslib.as_ctypes(
                np.zeros((n_spectrum, width, height)))
            self.shared_array = sharedctypes.RawArray(result._type_, result)

            batch_size = image_per_spectrum // processors
            items = [(i * batch_size, (i + 1) * batch_size if
                      (i + 1) * batch_size < image_per_spectrum else
                      image_per_spectrum) for i in range(processors)]
            print(items)

            with Pool(processes=processors) as pool:
                pool.map(self._multi_run_wrapper, items)

            result = np.ctypeslib.as_array(shared_array)

    def _multi_run_wrapper(self, args):
        return _read_images(*args)

    def _read_images(self, start, end):
        imgs = np.ctypeslib.as_array(self.shared_array)
        for n in range(start, end):
            for _x in self.x:
                for _y in self.y:
                    imgs[n, self.x[_x], self.y[_y]] = self.df.loc[
                        self.df['x'] == _x].loc[self.df['y'] ==
                                                _y].values[0, n + 2]