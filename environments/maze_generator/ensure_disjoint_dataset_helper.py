# Novelty Search Library.
# Copyright (C) 2020 Sorbonne University
# Maintainer: Achkan Salehi (salehi@isir.upmc.fr)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import sys
import os
import pdb
import scipy.spatial.distance


class dataset:
    """
    dataset of pbm images
    """

    def __init__(self, dir_name, as_vectors=True):

        self.fns = os.listdir(dir_name)
        self.fns = [dir_name + "/" + x for x in self.fns if ".pbm" in x]
        self.ims = []
        self.loaded = False
        self.as_vectors = as_vectors

    def load(self):

        for fn in self.fns:

            self.ims.append(
                cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype("bool"))
            self.ims[-1] = self.ims[-1].flatten(
            ) if self.as_vectors else self.ims[-1]

        self.loaded = True

    def compare_to_other_dataset(self, other):

        assert other.loaded and self.loaded, "empty dataset"
        assert self.as_vectors and other.as_vectors, "as_vectors should be True"

        c1 = np.concatenate([x.reshape(1, -1) for x in self.ims], 0)
        c2 = np.concatenate([x.reshape(1, -1) for x in other.ims], 0)
        dists = scipy.spatial.distance.cdist(c1, c2)

        shared = []
        for i in range(dists.shape[0]):
            if (dists[i] == 0).any():
                shared.append(i)

        return dists, shared

    def shuffle(self):
        """
        just for testing
        """
        np.random.shuffle(self.ims)


if __name__ == "__main__":

    dat1 = dataset(sys.argv[1])
    dat2 = dataset(sys.argv[2])
    dat1.load()
    dat2.load()

    dat2.shuffle()
    dd, shared = dat1.compare_to_other_dataset(dat2)
    plt.imshow(dd)
    plt.title(f"min value == {dd.min()}")
    print(
        f"{len(shared)} elements out of {dd.shape[0]} were commong between the two datasets"
    )
    plt.show()
