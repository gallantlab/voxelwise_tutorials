"""
================================
Download the data set from CRCNS
================================

This script describes how to download the data from CRCNS.

This tutorial is based on publicly available data
[published on CRCNS](https://crcns.org/data-sets/TBD).
"""
# sphinx_gallery_thumbnail_path = "_static/crcns.png"

###############################################################################
# Update the directory variable to link to the directory containing the data.

directory = '/data1/tutorials/vim-4/'

###############################################################################
# We run the download from CRCNS. A (free) account is required.
#
# We will only use the first subject in this tutorial, but you can run the same
# analysis on the four other subjects.
#
# We also skip the stimuli files, since the dataset provides two processed
# feature spaces to perform voxelwise modeling without requiring the original
# stimuli.

import getpass

from voxelwise.io import download_crcns

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
]

if __name__ == "__main__":

    username = input("CRCNS username: ")
    password = getpass.getpass("CRCNS password: ")

    for datafile in DATAFILES:
        local_filename = download_crcns(datafile, username, password,
                                        destination=directory)
