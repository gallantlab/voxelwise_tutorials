import os
import requests
import shutil

import h5py
import scipy.sparse
from himalaya.progress_bar import ProgressBar


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
        filename = os.path.join(*login_data['fn'].split('/')[1:])
        local_filename = os.path.join(destination, filename)

        # create subdirectory if necessary
        local_directory = os.path.dirname(local_filename)
        if not os.path.exists(local_directory) or not os.path.isdir(
                local_directory):
            os.makedirs(local_directory)

        # download the file if it does not already exist
        if os.path.exists(local_filename):
            print("%s already exists." % local_filename)
        else:
            bar = ProgressBar(title=filename, max_value=content_length)
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    bar.update_with_increment_value(chunk_size)
                    if chunk:
                        f.write(chunk)
            print('%s downloaded.' % local_filename)

    # uncompress archives
    if unpack and os.path.splitext(local_filename)[1] in [".zip", ".gz"]:
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
    extract_dir = os.path.dirname(archive_name)
    shutil.unpack_archive(archive_name, extract_dir=extract_dir)


def download_datalad(datafile, destination, source,
                     siblings=["wasabi", "origin"]):
    """
    Parameters
    ----------
    datafile : str
        Name of the file in the dataset.
    destination: str
        Directory where the data will be saved.
    source : str
        URL of the dataset.
    siblings : list of str
        List of sibling/remote on which the download will be attempted.

    Returns
    -------
    local_filename : str
        Local name of the downloaded file.

    Examples
    --------
    >>> from voxelwise_tutorials.io import get_data_home
    >>> from voxelwise_tutorials.io import download_datalad
    >>> destination = get_data_home(dataset="shortclips")
    >>> download_datalad("features/wordnet.hdf", destination=destination,
                         source="https://gin.g-node.org/gallantlab/shortclips")
    """
    import datalad.api
    dataset = datalad.api.Dataset(path=destination)
    if not dataset.is_installed():
        dataset = datalad.api.install(path=destination, source=source)

    def has_content():
        """Double check that the file is actually present."""
        repo = datalad.support.annexrepo.AnnexRepo(destination)
        return repo.file_has_content(datafile)

    for sibling in siblings:  # Try the download successively on each sibling
        result = dataset.get(datafile, source=sibling)
        if has_content():
            return result[0]["path"]

    raise RuntimeError("Failed to download %s." % datafile)


def load_hdf5_array(file_name, key=None, slice=slice(0, None)):
    """Function to load data from an hdf file.

    Parameters
    ----------
    file_name: string
        hdf5 file name.
    key: string
        Key name to load. If not provided, all keys will be loaded.
    slice: slice, or tuple of slices
        Load only a slice of the hdf5 array. It will load `array[slice]`.
        Use a tuple of slices to get a slice in multiple dimensions.

    Returns
    -------
    result : array or dictionary
        Array, or dictionary of arrays (if `key` is None).
    """
    with h5py.File(file_name, mode='r') as hf:
        if key is None:
            data = dict()
            for k in hf.keys():
                data[k] = hf[k][slice]
            return data
        else:
            # Some keys have been renamed. Use old key on KeyError.
            if key not in hf.keys() and key in OLD_KEYS:
                key = OLD_KEYS[key]

            return hf[key][slice]


def load_hdf5_sparse_array(file_name, key):
    """Load a scipy sparse array from an hdf file

    Parameters
    ----------
    file_name : string
        File name containing array to be loaded.
    key : string
        Name of variable to be loaded.

    Notes
    -----
    This function relies on variables being stored with specific naming
    conventions, so cannot be used to load arbitrary sparse arrays.
    """
    with h5py.File(file_name, mode='r') as hf:

        # Some keys have been renamed. Use old key on KeyError.
        if '%s_data' % key not in hf.keys() and key in OLD_KEYS:
            key = OLD_KEYS[key]

        # The voxel_to_fsaverage mapper is sometimes split between left/right.
        if (key == "voxel_to_fsaverage" and '%s_data' % key not in hf.keys()
                and "vox_to_fsavg_left_data" in hf.keys()):
            left = load_hdf5_sparse_array(file_name, "vox_to_fsavg_left")
            right = load_hdf5_sparse_array(file_name, "vox_to_fsavg_right")
            return scipy.sparse.vstack([left, right])

        data = (hf['%s_data' % key], hf['%s_indices' % key],
                hf['%s_indptr' % key])
        sparsemat = scipy.sparse.csr_matrix(data, shape=hf['%s_shape' % key])
    return sparsemat


# Correspondence between old keys and new keys. {new_key: old_key}
OLD_KEYS = {
    "flatmap_mask": "pixmask",
    "voxel_to_flatmap": "pixmap",
}


def save_hdf5_dataset(file_name, dataset, mode='w'):
    """Save a dataset of arrays and sparse arrays.

    Parameters
    ----------
    file_name : str
        Full name of the file.
    dataset : dict of arrays
        Mappers to save.
    mode : str
        File opening model.
        Use 'w' to write from scratch, 'a' to add to existing file.
    """
    print("Saving... ", end="", flush=True)

    with h5py.File(file_name, mode=mode) as hf:
        for name, array in dataset.items():

            if scipy.sparse.issparse(array):  # sparse array
                array = array.tocsr()
                hf.create_dataset(name + '_indices', data=array.indices,
                                  compression='gzip')
                hf.create_dataset(name + '_data', data=array.data,
                                  compression='gzip')
                hf.create_dataset(name + '_indptr', data=array.indptr,
                                  compression='gzip')
                hf.create_dataset(name + '_shape', data=array.shape,
                                  compression='gzip')
            else:  # dense array
                hf.create_dataset(name, data=array, compression='gzip')

    print("Saved %s" % file_name)


def get_data_home(dataset=None, data_home=None) -> str:
    """Return the path of the voxelwise tutorials data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times. By default the data dir is set to a folder named
    'voxelwise_tutorials' in the user home folder. Alternatively, it can be set
    by the 'VOXELWISE_TUTORIALS_DATA' environment variable or programmatically
    by giving an explicit folder path. The '~' symbol is expanded to the user
    home folder. If the folder does not already exist, it is automatically
    created.

    Parameters
    ----------
    dataset : str | None
        Optional name of a particular dataset subdirectory, to append to the
        data_home path.
    data_home : str | None
        Optional path to voxelwise tutorials data dir, to use instead of the
        default one.

    Returns
    -------
    data_home : str
        The path to voxelwise tutorials data directory.
    """
    if data_home is None:
        data_home = os.environ.get(
            'VOXELWISE_TUTORIALS_DATA',
            os.path.join('~', 'voxelwise_tutorials_data'))

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    if dataset is not None:
        data_home = os.path.join(data_home, dataset)

    return data_home


def clear_data_home(dataset=None, data_home=None):
    """Delete all the content of the data home cache.

    Parameters
    ----------
    dataset : str | None
        Optional name of a particular dataset subdirectory, to append to the
        data_home path.
    data_home : str | None
        Optional path to voxelwise tutorials data dir, to use instead of the
        default one.
    """
    data_home = get_data_home(dataset=dataset, data_home=data_home)
    shutil.rmtree(data_home)
