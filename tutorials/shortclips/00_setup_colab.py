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
# Uncomment and run the following cell to download the tutorial data and
# install the required dependencies.

# !pip install -U --no-cache-dir gdown --pre
# ![ -f "vim-5-for-ccn.tar.gz" ] || gdown --id 1b0I0Ytj06m6GCmfxfNrZuyF97fDo3NZb
# ![ -d "vim-5" ] || tar xzf vim-5-for-ccn.tar.gz
# !ln -s vim-5 shortclips
# !apt-get install -qq inkscape > /dev/null
# !pip install -q voxelwise_tutorials>=0.1.2
# ![ -f "ngrok-stable-linux-amd64.zip" ] || wget -q https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
# ![ -f "ngrok" ] || unzip ngrok-stable-linux-amd64.zip

###############################################################################
# If downloading the data failed, you can try to download it manually, turning
# `if False` into `if True`, and running the following cell.

if False:
    from voxelwise_tutorials.io import get_data_home
    from voxelwise_tutorials.io import download_datalad

    for datafile in [
        "features/motion_energy.hdf", "features/wordnet.hdf",
        "mappers/S01_mappers.hdf", "responses/S01_responses.hdf",
    ]:
        local_filename = download_datalad(
            datafile, destination=get_data_home(dataset="shortclips"),
            source="https://gin.g-node.org/gallantlab/shortclips")

###############################################################################
# For the record, here is what each command does:

# - update gdown to get the latest fixes
# - Download the dataset archive
# - Extract the dataset archive
# - Create a symlink to the extracted dataset
# - Install Inkscape, to use more features from Pycortex
# - Install the tutorial helper package, and all the required dependencies
# - Download ngrok to create a tunnel for pycortex 3D brain viewer
# - Extract the ngrok archive

###############################################################################
# Now run the following cell to set up the environment variables for the
# tutorials and pycortex.

import os
os.environ['VOXELWISE_TUTORIALS_DATA'] = "/content"

import sklearn
sklearn.set_config(assume_finite=True)

###############################################################################
# Your Google Colab environment is now set up for the voxelwise tutorials.
