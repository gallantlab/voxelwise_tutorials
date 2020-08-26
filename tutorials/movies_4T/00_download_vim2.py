"""
================================
Download the data set from CRCNS
================================

In this script, we download the data set from CRCNS.
A (free) account is required.

Cite this data set
------------------

This tutorial is based on publicly available data
`published on CRCNS <https://crcns.org/data-sets/vc/vim-2/about-vim-2>`_.
If you publish any work using this data set, please cite the original
publication [1]_, and the data set [2]_.
"""
# sphinx_gallery_thumbnail_path = "static/crcns.png"
###############################################################################
# Download
# --------

# path of the data directory
import os
from voxelwise_tutorials.io import get_data_home
directory = os.path.join(get_data_home(), "vim-2")
print(directory)

###############################################################################
# We will only use the first subject in this tutorial, but you can run the same
# analysis on the two other subjects. Uncomment the lines in ``DATAFILES`` to
# download more subjects, or to download the anatomy files.

import getpass

from voxelwise_tutorials.io import download_crcns

DATAFILES = [
    'vim-2/Stimuli.tar.gz',
    'vim-2/VoxelResponses_subject1.tar.gz',
    # 'vim-2/VoxelResponses_subject2.tar.gz',
    # 'vim-2/VoxelResponses_subject3.tar.gz',
    # 'vim-2/anatomy.zip',
    # 'vim-2/checksums.md5',
    # 'vim-2/filelist.txt',
    # 'vim-2/docs/crcns-vim-2-data-description.pdf',
]

###############################################################################
username = input("CRCNS username: ")
password = getpass.getpass("CRCNS password: ")

for datafile in DATAFILES:
    local_filename = download_crcns(datafile, username, password,
                                    destination=directory, unpack=True)
###############################################################################
# References
# ----------
#
# .. [1] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu, B., &
#     Gallant, J. L. (2011). Reconstructing visual experiences from brain
#     activity evoked by natural movies. Current Biology, 21(19), 1641-1646.
#
# .. [2] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu, B., &
#     Gallant, J. L. (2014): Gallant Lab Natural Movie 4T fMRI Data. CRCNS.org.
#     http://dx.doi.org/10.6080/K00Z715X
