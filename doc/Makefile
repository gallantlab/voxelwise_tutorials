# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = Tutorials
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/
	rm -rf _auto_examples/

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished (noplot). The HTML pages are in $(BUILDDIR)/html."

notebooks:
	rm -f ../tutorials/notebooks/*/*.ipynb
	python create_notebooks.py
	$(MAKE) merge-notebooks

NBDIR = ../tutorials/notebooks/movies_3T

merge-notebooks:
	python merge_notebooks.py \
		$(NBDIR)/00_load_colab.ipynb \
		$(NBDIR)/01_plot_explainable_variance.ipynb \
		$(NBDIR)/02_plot_wordnet_model.ipynb \
		$(NBDIR)/03_plot_hemodynamic_response.ipynb \
		$(NBDIR)/04_plot_motion_energy_model.ipynb \
		$(NBDIR)/05_plot_banded_ridge_model.ipynb \
		> $(NBDIR)/merged_for_colab.ipynb
	echo "Saved in $(NBDIR)/merged_for_colab.ipynb"
