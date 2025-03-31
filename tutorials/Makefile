# Makefile to use with jupyter-book

# You can set these variables from the command line.
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Available commands:"
	@echo "  help            Show this help message"
	@echo "  clean           Remove build directory"
	@echo "  merge-notebooks Merge notebook files into single files for Colab"
	@echo "  build           Build the jupyter book"
	@echo "  push-pages      Push built book to GitHub Pages"
	@echo "  preview         Preview the book locally"

.PHONY: help Makefile clean merge-notebooks build push-pages preview

clean:
	rm -rf $(BUILDDIR)/

NBDIR = notebooks/shortclips

merge-notebooks:
	python merge_notebooks.py \
		$(NBDIR)/01_setup_colab.ipynb \
		$(NBDIR)/02_download_shortclips.ipynb \
		$(NBDIR)/03_compute_explainable_variance.ipynb \
		$(NBDIR)/04_understand_ridge_regression.ipynb \
		$(NBDIR)/05_fit_wordnet_model.ipynb \
		$(NBDIR)/06_visualize_hemodynamic_response.ipynb \
		$(NBDIR)/08_fit_motion_energy_model.ipynb \
		$(NBDIR)/09_fit_banded_ridge_model.ipynb \
		> $(NBDIR)/vem_tutorials_merged_for_colab.ipynb
	echo "Saved in $(NBDIR)/vem_tutorials_merged_for_colab.ipynb"

	python merge_notebooks.py \
		$(NBDIR)/01_setup_colab.ipynb \
		$(NBDIR)/02_download_shortclips.ipynb \
		$(NBDIR)/03_compute_explainable_variance.ipynb \
		$(NBDIR)/05_fit_wordnet_model.ipynb \
		$(NBDIR)/08_fit_motion_energy_model.ipynb \
		$(NBDIR)/09_fit_banded_ridge_model.ipynb \
		> $(NBDIR)/vem_tutorials_merged_for_colab_model_fitting.ipynb
	echo "Saved in $(NBDIR)/vem_tutorials_merged_for_colab_model_fitting.ipynb"

build:
	jupyter book build --all $(SOURCEDIR)

preview:
	python -m http.server --directory _build/html 8000
	@echo "Preview the book at http://localhost:8000"
	@echo "Press Ctrl+C to stop the server"

# -b gh_pages --single-branch (to clone only one branch)
# --no-checkout (just fetches the root folder without content)
# --depth 1 (since we don't need the history prior to the last commit)
push-pages:
	rm -rf _build/gh_pages
	git clone -b gh-pages --single-branch --no-checkout --depth 1 \
		https://github.com/gallantlab/voxelwise_tutorials _build/gh_pages

	cd _build/ && \
		cp -r html/* gh_pages && \
		cd gh_pages && \
		touch .nojekyll && \
		echo "*.ipynb -diff" > .gitattributes && \
		git add * && \
		git add .nojekyll .gitattributes && \
		git commit -a -m 'Make push-pages' && \
		git push
