"""
================================
Download the data set from CRCNS
================================

This script describes how to download the data from CRCNS.

This tutorial is based on publicly available data
[published on CRCNS](https://crcns.org/data-sets/vc/vim-2/about-vim-2).
"""
# sphinx_gallery_thumbnail_path = "_static/crcns.png"
###############################################################################
# Update the directory variable to link to the directory containing the data.

directory = '/data1/tutorials/vim-2/'

###############################################################################
# We run the download from CRCNS. A (free) account is required.
#
# We will only use the first subject in this tutorial, but you can run the same
# analysis on the two other subjects.

import getpass

from voxelwise.io import download_crcns

DATAFILES = [
    'vim-2/Stimuli.tar.gz',
    'vim-2/VoxelResponses_subject1.tar.gz',
    # 'vim-2/VoxelResponses_subject2.tar.gz',
    # 'vim-2/VoxelResponses_subject3.tar.gz',
    # 'vim-2/anatomy.zip',
    'vim-2/checksums.md5',
    'vim-2/filelist.txt',
    'vim-2/docs/crcns-vim-2-data-description.pdf',
]

if __name__ == "__main__":

    username = input("CRCNS username: ")
    password = getpass.getpass("CRCNS password: ")

    for datafile in DATAFILES:
        local_filename = download_crcns(datafile, username, password,
                                        destination=directory, unpack=True)
