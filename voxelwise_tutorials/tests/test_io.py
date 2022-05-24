from tempfile import NamedTemporaryFile

import numpy as np
import scipy.sparse as sp

from voxelwise_tutorials.io import save_hdf5_dataset
from voxelwise_tutorials.io import load_hdf5_array
from voxelwise_tutorials.io import load_hdf5_sparse_array


def test_save_dataset_to_hdf5():
    tmp_file = NamedTemporaryFile(suffix='.hdf5')
    file_name = tmp_file.name

    dataset = {
        'pixmap': sp.csr_matrix(np.random.rand(10, 3)),
        'array': np.random.rand(10, 3),
    }
    save_hdf5_dataset(file_name, dataset)

    # test loading sparse arrays
    pixmap = load_hdf5_sparse_array(file_name, 'pixmap')
    np.testing.assert_array_equal(pixmap.toarray(), dataset['pixmap'].toarray())
    # test new name
    pixmap = load_hdf5_sparse_array(file_name, 'voxel_to_flatmap')
    np.testing.assert_array_equal(pixmap.toarray(), dataset['pixmap'].toarray())
    # test loading dense arrays
    array = load_hdf5_array(file_name, 'array')
    np.testing.assert_array_equal(array, dataset['array'])
