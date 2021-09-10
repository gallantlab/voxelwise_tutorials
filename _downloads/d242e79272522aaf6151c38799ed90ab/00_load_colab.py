"""
==================
Setup Google Colab
==================

In this script, we setup a Google Colab environment. This script will only work
when run from `Google Colab <https://colab.research.google.com/>`_). You can
skip it if you run the tutorials on your machine.
"""
# sphinx_gallery_thumbnail_path = "static/colab.png"
#
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

# ![ -f "vim-5-for-ccn.tar.gz" ] || gdown --id 1b0I0Ytj06m6GCmfxfNrZuyF97fDo3NZb
# ![ -d "vim-5" ] || tar xzf vim-5-for-ccn.tar.gz
# ![ -d "pycortex" ] || git clone https://github.com/gallantlab/pycortex
# !pip install -q voxelwise_tutorials


###############################################################################
# Now run the following cell to set up the environment variables for the
# tutorials and pycortex.

import os
os.environ['VOXELWISE_TUTORIALS_DATA'] = "/content"

import cortex
filestore = "/content/pycortex/filestore/"
cortex.options.config['basic']['filestore'] = filestore
cortex.options.config['webgl']['colormaps'] = "/content/pycortex/filestore/colormaps"
cortex.database.db = cortex.database.Database(filestore)
cortex.db = cortex.database.db
cortex.utils.db = cortex.database.db
cortex.dataset.braindata.db = cortex.database.db

import sklearn
sklearn.set_config(assume_finite=True)

###############################################################################
# Your Google Colab environment is now set up for the voxelwise tutorials.
