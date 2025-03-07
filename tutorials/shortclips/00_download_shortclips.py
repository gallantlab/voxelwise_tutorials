"""
=====================
Download the data set
=====================

In this script, we download the data set from Wasabi or GIN. No account is
required.

Cite this data set
------------------

This tutorial is based on publicly available data `published on GIN
<https://gin.g-node.org/gallantlab/shortclips>`_. If you publish any work using
this data set, please cite the original publication {cite}`huth2012`, and the data set
{cite}`huth2022data`.
"""
# sphinx_gallery_thumbnail_path = "static/download.png"
###############################################################################
# Download
# --------

# path of the data directory
from voxelwise_tutorials.io import get_data_home

directory = get_data_home(dataset="shortclips")
print(directory)

###############################################################################
# We will only use the first subject in this tutorial, but you can run the same
# analysis on the four other subjects. Uncomment the lines in ``DATAFILES`` to
# download more subjects.
#
# We also skip the stimuli files, since the dataset provides two preprocessed
# feature spaces to perform voxelwise modeling without requiring the original
# stimuli.

from voxelwise_tutorials.io import download_datalad

DATAFILES = [
    "features/motion_energy.hdf",
    "features/wordnet.hdf",
    "mappers/S01_mappers.hdf",
    # "mappers/S02_mappers.hdf",
    # "mappers/S03_mappers.hdf",
    # "mappers/S04_mappers.hdf",
    # "mappers/S05_mappers.hdf",
    "responses/S01_responses.hdf",
    # "responses/S02_responses.hdf",
    # "responses/S03_responses.hdf",
    # "responses/S04_responses.hdf",
    # "responses/S05_responses.hdf",
    # "stimuli/test.hdf",
    # "stimuli/train_00.hdf",
    # "stimuli/train_01.hdf",
    # "stimuli/train_02.hdf",
    # "stimuli/train_03.hdf",
    # "stimuli/train_04.hdf",
    # "stimuli/train_05.hdf",
    # "stimuli/train_06.hdf",
    # "stimuli/train_07.hdf",
    # "stimuli/train_08.hdf",
    # "stimuli/train_09.hdf",
    # "stimuli/train_10.hdf",
    # "stimuli/train_11.hdf",
]

source = "https://gin.g-node.org/gallantlab/shortclips"

for datafile in DATAFILES:
    local_filename = download_datalad(datafile, destination=directory,
                                      source=source)

###############################################################################
# References
# ----------
#
# ```{bibliography}
# :filter: docname in docnames
# ```
