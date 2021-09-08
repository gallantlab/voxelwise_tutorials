"""
================================
Download the data set from CRCNS
================================

In this script, we download the data set from CRCNS. A (free) account is
required.

.. Warning:: The data has not been publicly released yet on CRCNS, so this
    notebook will not work ! You can download the data manually from `Google
    Drive
    <https://drive.google.com/drive/folders/1NuxO5_GHgDvjrL2FX5ohzAsvWpZuepIA?usp=sharing>`_,
    or run the tutorials from `Colab
    <https://colab.research.google.com/github/gallantlab/voxelwise_tutorials/blob/main/tutorials/notebooks/movies_3T/merged_for_colab.ipynb>`_

Cite this data set
------------------

This tutorial is based on publicly available data `published on CRCNS
<https://crcns.org/data-sets/vc/TBD>`_. If you publish any work using this data
set, please cite the original publication [1]_, and the data set [2]_.
"""
# sphinx_gallery_thumbnail_path = "static/crcns.png"

raise RuntimeError("The data has not been publicly released yet, so "
                   "this script/notebook will not work !")

###############################################################################
# Download
# --------

# path of the data directory
import os
from voxelwise_tutorials.io import get_data_home
directory = os.path.join(get_data_home(), "vim-5")
print(directory)

###############################################################################
# We will only use the first subject in this tutorial, but you can run the same
# analysis on the four other subjects. Uncomment the lines in ``DATAFILES`` to
# download more subjects.
#
# We also skip the stimuli files, since the dataset provides two preprocessed
# feature spaces to perform voxelwise modeling without requiring the original
# stimuli.

import getpass

from voxelwise_tutorials.io import download_crcns

DATAFILES = [
    "TBD/features/motion_energy.hdf",
    "TBD/features/wordnet.hdf",
    "TBD/mappers/S01_mappers.hdf",
    # "TBD/mappers/S02_mappers.hdf",
    # "TBD/mappers/S03_mappers.hdf",
    # "TBD/mappers/S04_mappers.hdf",
    # "TBD/mappers/S05_mappers.hdf",
    "TBD/responses/S01_responses.hdf",
    # "TBD/responses/S02_responses.hdf",
    # "TBD/responses/S03_responses.hdf",
    # "TBD/responses/S04_responses.hdf",
    # "TBD/responses/S05_responses.hdf",
    # "TBD/stimuli/test.hdf",
    # "TBD/stimuli/train_00.hdf",
    # "TBD/stimuli/train_01.hdf",
    # "TBD/stimuli/train_02.hdf",
    # "TBD/stimuli/train_03.hdf",
    # "TBD/stimuli/train_04.hdf",
    # "TBD/stimuli/train_05.hdf",
    # "TBD/stimuli/train_06.hdf",
    # "TBD/stimuli/train_07.hdf",
    # "TBD/stimuli/train_08.hdf",
    # "TBD/stimuli/train_09.hdf",
    # "TBD/stimuli/train_10.hdf",
    # "TBD/stimuli/train_11.hdf",
    "TBD/utils/wordnet_categories.txt",
    "TBD/utils/wordnet_graph.dot",
]

###############################################################################
username = input("CRCNS username: ")
password = getpass.getpass("CRCNS password: ")

for datafile in DATAFILES:
    local_filename = download_crcns(datafile, username, password,
                                    destination=directory)

###############################################################################
# References
# ----------
#
# .. [1] Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2012). A
#     continuous semantic space describes the representation of thousands of
#     object and action categories across the human brain. Neuron, 76(6),
#     1210-1224.
#
# .. [2] Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2020):
#     Gallant Lab Natural Movie 3T fMRI Data. CRCNS.org.
#     http://dx.doi.org/10.6080/TBD
