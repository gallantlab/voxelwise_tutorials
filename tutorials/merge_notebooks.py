#!/usr/bin/env python
# Inspired from https://gist.github.com/fperez/e2bbc0a208e82e450f69
"""
usage:
python merge_notebooks.py A.ipynb B.ipynb C.ipynb > merged.ipynb
"""
import io
import os.path
import sys

import nbformat


def merge_notebooks(filenames):
    merged = None
    for fname in filenames:
        with io.open(fname, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        if merged is None:
            merged = nb
            # add a Markdown cell with the file name, then all cells
            # merged.cells = [cell_with_title(title=os.path.basename(fname))
            #                 ] + nb.cells
            # Do not add the title cell, use the title in each notebook
            merged.cells = nb.cells
        else:
            # add a code cell resetting all variables
            merged.cells.append(cell_with_reset())
            # add a Markdown cell with the file name
            # merged.cells.append(cell_with_title(title=os.path.basename(fname)))
            # add all cells from current notebook
            merged.cells.extend(nb.cells)

    if not hasattr(merged.metadata, 'name'):
        merged.metadata.name = ''
    merged.metadata.name += "_merged"
    print(nbformat.writes(merged))


def cell_with_title(title):
    """Returns a Markdown cell with a title."""
    return nbformat.from_dict({
        'cell_type': 'markdown',
        'metadata': {},
        'source': f'\n# {title}\n',
    })


def cell_with_reset():
    """Returns a code cell with magic command to reset all variables."""
    return nbformat.from_dict({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {'collapsed': False},
        'outputs': [],
        'source': '%reset -f',
    })


if __name__ == '__main__':
    notebooks = sys.argv[1:]
    if not notebooks:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    merge_notebooks(notebooks)
