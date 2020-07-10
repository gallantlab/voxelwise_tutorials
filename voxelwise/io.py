import os
import os.path as op
import requests
import shutil

from .progress_bar import ProgressBar

URL_CRCNS = 'https://portal.nersc.gov/project/crcns/download/index.php'


def download_crcns(datafile, username, password, destination,
                   chunk_size=2 ** 20, unpack=True):
    """Download a file from CRCNS, with a progress bar.

    Parameters
    ----------
    datafile : str
        Name of the file on CRCNS.
    username : str
        Username on CRCNS.
    password : str
        Password on CRCNS.
    destination: str
        Directory where the data will be saved.
    chunk_size : int
        Size of the data downloaded at each iteration.
    unpack : bool
        If True, archives will be uncompress locally after the download.

    Returns
    -------
    local_filename : str
        Local name of the downloaded file.
    """

    login_data = dict(username=username, password=password, fn=datafile,
                      submit='Login')

    with requests.Session() as s:
        response = s.post(URL_CRCNS, data=login_data, stream=True)

        # get content length for error checking and progress bar
        content_length = int(response.headers['Content-Length'])

        # check errors if the content is small
        if content_length < 1000:
            if "Error" in response.text:
                raise RuntimeError(response.text)

        # remove the dataset name
        filename = op.join(*login_data['fn'].split('/')[1:])
        local_filename = op.join(destination, filename)

        # create subdirectory if necessary
        local_directory = op.dirname(local_filename)
        if not op.exists(local_directory) or not op.isdir(local_directory):
            os.makedirs(local_directory)

        # download the file if it does not already exist
        if op.exists(local_filename):
            print("%s already exists." % local_filename)
        else:
            bar = ProgressBar(title=filename, max_value=content_length)
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    bar.update_with_increment_value(chunk_size)
                    if chunk:
                        f.write(chunk)

    # uncompress archives
    if unpack and op.splitext(local_filename)[1] in [".zip", ".gz"]:
        unpack_archive(local_filename)

    return local_filename


def unpack_archive(archive_name):
    """Unpack an archive, saving files on the same directory.

    Parameters
    ----------
    archive_name : str
        Local name of the archive.
    """
    print('\tUnpacking')
    extract_dir = op.dirname(archive_name)
    shutil.unpack_archive(archive_name, extract_dir=extract_dir)
