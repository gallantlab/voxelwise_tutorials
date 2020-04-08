import re
from setuptools import find_packages, setup

# get version from voxelwise/__init__.py
with open('voxelwise/__init__.py') as f:
    infos = f.readlines()
for line in infos:
    if "__version__" in line:
        match = re.search(r"__version__ = '([^']*)'", line)
        __version__ = match.groups()[0]

requirements = [
    "numpy",
    "scipy",
    "h5py",
    "scikit-learn",
    "scikit-image",
    "himalaya",
]

if __name__ == "__main__":
    setup(
        name='voxelwise',
        maintainer="Tom Dupre la Tour",
        maintainer_email="tom.dupre-la-tour@m4x.org",
        description="Tools and tutorials for voxelwise modeling",
        license='BSD (3-clause)',
        version=__version__,
        packages=find_packages(),
        install_requires=requirements,
    )
