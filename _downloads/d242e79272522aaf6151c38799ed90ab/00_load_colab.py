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
# This tutorial is much faster when a GPU is available to run the computations:
#
# (Menu) "Runtime" -> "Change runtime type" -> "Hardware accelerator" -> "GPU".

###############################################################################
# Mount data with Google Drive
# ----------------------------
#
# First, open the following Google Drive link:
# https://drive.google.com/drive/folders/1NuxO5_GHgDvjrL2FX5ohzAsvWpZuepIA
#
# Then, clic on the directory name ("vim-5"), and add a shortcut to your Drive
# ("Add shortcut to Drive").
#
# Then, mount Google Drive in Google Colab. To do so, run the following cell,
# and follow the instructions from Google to copy/paste the authorization code.

from google.colab import drive
drive.mount("/content/drive")

###############################################################################
# Uncomment and run the following command to check that Google Drive was
# correctly mounted.

# !ls 'drive' && echo SUCCESS || echo FAILURE

###############################################################################
# Then, tell the voxelwise_tutorials package where the data is.

import os
os.environ['VOXELWISE_TUTORIALS_DATA'] = "drive/MyDrive/"

###############################################################################
# Install package helper
# ----------------------
#
# Finally, install the tutorial helper package, by uncommenting and running
# the install command.

# !pip install voxelwise_tutorials

###############################################################################
# When using Colab, the install of pycortex might fail to locate the default
# pycortex filestore. You can fix it by uncommenting and running the following:

# !git clone https://github.com/gallantlab/pycortex

###############################################################################
# and changing the filestore path.

import cortex
filestore = "/content/pycortex/filestore/"
cortex.options.config['basic']['filestore'] = filestore
cortex.options.config['webgl']['colormaps'] = "/content/pycortex/filestore/colormaps"
cortex.database.db = cortex.database.Database(filestore)
cortex.db = cortex.database.db
cortex.utils.db = cortex.database.db
cortex.dataset.braindata.db = cortex.database.db
