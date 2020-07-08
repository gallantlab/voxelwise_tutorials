"""
This tutorial is based on publicly available data
[published on CRCNS](https://crcns.org/data-sets/vc/vim-2/about-vim-2).

This script describes how to download the data from CRCNS.
"""

###############################################################################
# First, let's define some downloading function, with a progress bar.

import os.path as op
import requests
import getpass
import shutil

from himalaya.progress_bar import ProgressBar

URL = 'https://portal.nersc.gov/project/crcns/download/index.php'


def download(datafile, username, password, path, chunk_size=2 ** 20):
    """Download a file from CRCNS, with a progress bar."""

    login_data = dict(username=username, password=password, fn=datafile,
                      submit='Login')

    with requests.Session() as s:
        response = s.post(URL, data=login_data, stream=True)

        # get content length for error checking and progress bar
        content_length = int(response.headers['Content-Length'])

        # check errors if the content is small
        if content_length < 1000:
            if "Error" in response.text:
                raise RuntimeError(response.text)

        filename = login_data['fn'].split('/')[-1]
        local_filename = op.join(path, filename)
        if op.exists(local_filename):
            print("%s already exists." % local_filename)
        else:
            bar = ProgressBar(title=filename, max_value=content_length)
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    bar.update_with_increment_value(chunk_size)
                    if chunk:
                        f.write(chunk)

    return local_filename


def uncompress(local_filename):
    """Uncompress an archive, saving files on the same directory."""
    print('\tUncompressing')
    extract_dir = op.dirname(local_filename)
    shutil.unpack_archive(local_filename, extract_dir=extract_dir)


###############################################################################
# Then, we run the download from CRCNS. A (free) account is required.
#
# We will only use the first subject in this tutorial, but you can run the same
# analysis on the two other subjects.

DATAFILES = [
    'vim-2/Stimuli.tar.gz',
    'vim-2/VoxelResponses_subject1.tar.gz',
    # 'vim-2/VoxelResponses_subject2.tar.gz',
    # 'vim-2/VoxelResponses_subject3.tar.gz',
    'vim-2/anatomy.zip',
    'vim-2/checksums.md5',
    'vim-2/docs',
    'vim-2/filelist.txt',
    'vim-2/docs/crcns-vim-2-data-description.pdf',
]

if __name__ == "__main__":

    path = "/data1/tutorials/vim2"

    username = input("CRCNS username: ")
    password = getpass.getpass("CRCNS password: ")

    for datafile in DATAFILES:
        local_filename = download(datafile, username, password, path)

        if op.splitext(local_filename)[1] in [".zip", ".gz"]:
            uncompress(local_filename)
