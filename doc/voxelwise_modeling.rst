The voxelwise modeling framework
================================

VM Framework
------------

Voxelwise modeling (VM) is a framework to perform functional magnetic resonance
imaging (fMRI) data analysis.
Over the years, VM has led to many high profile publications 
[1]_ [2]_ [3]_ [4]_  [5]_ [6]_ [7]_ [8]_ [9]_ [10]_ [11]_.

[...]

Critical improvements
---------------------

VM provides multiple critical improvements over other approaches to fMRI data
analysis:

#.
    Most methods for analyzing fMRI data rely on simple contrasts
    between a small number of conditions. In contrast, VM can efficiently analyze
    many different stimulus and task features simultaneously. This framework
    enables the analysis of complex naturalistic stimuli and tasks which contain
    a large number of features; for example, VM has been used with naturalistic images
    [1]_ [2]_, movies [3]_, and stories [8]_.

#.
    Unlike the traditional null hypothesis testing framework, VM is not prone
    to overfitting and type I error and generalizes to new subjects and stimuli .
    VM is a predictive modeling framework that
    evaluates model performance on a separate test data set not used during fitting.

#.
    VM performs an analysis in each subject’s native brain space instead of lossily
    transforming subjects into a common group space. This allows VM to produce
    results with maximal spatial resolution. Each subject provides their own fit
    and test data, so every subject provides a complete replication of all
    hypothesis tests.

#.
    VM produces high-dimensional functional maps rather than simple contrast 
    maps or correlation matrices. These maps reflect the
    selectivity of each voxel to thousands of stimulus and task features spread
    across dozens of feature spaces. These functional maps are much more
    detailed than those produced using statistical parametric mapping (SPM),
    multivariate pattern analysis (MVPA), or representational similarity
    analysis (RSA).

#.
    VM recovers stable and interpretable functional parcellations, which
    respect individual variability in anatomy [8]_. 


References
----------

.. [1] Kay, K. N., Naselaris, T., Prenger, R. J., & Gallant, J. L. (2008).
    Identifying natural images from human brain activity.
    Nature, 452(7185), 352-355.

.. [2] Naselaris, T., Prenger, R. J., Kay, K. N., Oliver, M., & Gallant, J. L. (2009).
    Bayesian reconstruction of natural images from human brain activity.
    Neuron, 63(6), 902-915.

.. [3] Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu, B., & Gallant, J. L. (2011).
    Reconstructing visual experiences from brain activity evoked by natural movies.
    Current Biology, 21(19), 1641-1646.

.. [4] Huth, A. G., Nishimoto, S., Vu, A. T., & Gallant, J. L. (2012).
    A continuous semantic space describes the representation of thousands of
    object and action categories across the human brain.
    Neuron, 76(6), 1210-1224.

.. [5] Çukur, T., Nishimoto, S., Huth, A. G., & Gallant, J. L. (2013).
    Attention during natural vision warps semantic representation across the human brain.
    Nature neuroscience, 16(6), 763-770.

.. [6] Çukur, T., Huth, A. G., Nishimoto, S., & Gallant, J. L. (2013).
    Functional subdomains within human FFA.
    Journal of Neuroscience, 33(42), 16748-16766.

.. [7] Stansbury, D. E., Naselaris, T., & Gallant, J. L. (2013).
    Natural scene statistics account for the representation of scene categories
    in human visual cortex.
    Neuron, 79(5), 1025-1034

.. [8] Huth, A. G., De Heer, W. A., Griffiths, T. L., Theunissen, F. E., & Gallant, J. L. (2016).
    Natural speech reveals the semantic maps that tile human cerebral cortex.
    Nature, 532(7600), 453-458.

.. [9] de Heer, W. A., Huth, A. G., Griffiths, T. L., Gallant, J. L., & Theunissen, F. E. (2017).
    The hierarchical cortical organization of human speech processing.
    Journal of Neuroscience, 37(27), 6539-6557.

.. [10] Lescroart, M. D., & Gallant, J. L. (2019).
    Human scene-selective areas represent 3D configurations of surfaces.
    Neuron, 101(1), 178-192.

.. [11] Deniz, F., Nunez-Elizalde, A. O., Huth, A. G., & Gallant, J. L. (2019).
    The representation of semantic information across human cerebral cortex
    during listening versus reading is invariant to stimulus modality.
    Journal of Neuroscience, 39(39), 7722-7736.