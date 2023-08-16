"""
==================
Setup Google Colab
==================

In this script, we setup a Google Colab environment. This script will only work
when run from `Google Colab <https://colab.research.google.com/>`_). You can
skip it if you run the tutorials on your machine.
"""
# sphinx_gallery_thumbnail_path = "static/colab.png"
###############################################################################
# Change runtime to use a GPU
# ---------------------------
#
# This tutorial is much faster when a GPU is available to run the computations.
# In Google Colab you can request access to a GPU by changing the runtime type.
# To do so, click the following menu options in Google Colab:
#
# (Menu) "Runtime" -> "Change runtime type" -> "Hardware accelerator" -> "GPU".

###############################################################################
# Download the data and install all required dependencies
# -------------------------------------------------------
#
# Uncomment and run the following cell to download the required packages.

#!git config --global user.email "you@example.com" && git config --global user.name "Your Name"
#!wget -O- http://neuro.debian.net/lists/jammy.us-ca.libre | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
#!apt-key adv --recv-keys --keyserver hkps://keyserver.ubuntu.com 0xA5D32F012649A5A9 > /dev/null
#!apt-get -qq update > /dev/null
#!apt-get install -qq inkscape git-annex-standalone > /dev/null
#!pip install -q voxelwise_tutorials

###############################################################################
# For the record, here is what each command does:

# - Set up an email and username to use git, git-annex, and datalad (required to download the data)
# - Add NeuroDebian to the package sources
# - Update the gpg keys to use NeuroDebian
# - Update the list of available packages
# - Install Inkscape to use more features from Pycortex, and install git-annex to download the data
# - Install the tutorial helper package, and all the required dependencies

###############################################################################
try:
    import google.colab # noqa
    in_colab = True
except ImportError:
    in_colab = False
if not in_colab:
    raise RuntimeError("This script is only meant to be run from Google "
                       "Colab. You can skip it if you run the tutorials "
                       "on your machine.")

###############################################################################
# Now run the following cell to download the data for the tutorials.

from voxelwise_tutorials.io import download_datalad

DATAFILES = [
    "features/motion_energy.hdf",
    "features/wordnet.hdf",
    "mappers/S01_mappers.hdf",
    "responses/S01_responses.hdf",
]

source = "https://gin.g-node.org/gallantlab/shortclips"
destination = "/content/shortclips"

for datafile in DATAFILES:
    local_filename = download_datalad(
        datafile,
        destination=destination,
        source=source
    )

###############################################################################
# Now run the following cell to set up the environment variables for the
# tutorials and pycortex.

import os
os.environ['VOXELWISE_TUTORIALS_DATA'] = "/content"

import sklearn
sklearn.set_config(assume_finite=True)

###############################################################################
# Your Google Colab environment is now set up for the voxelwise tutorials.
