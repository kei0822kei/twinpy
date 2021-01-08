# Makefile for Sphinx documentation

SPHINXBUILD   = sphinx-build
BUILDDIR      = build
ALLSPHINXOPTS = -n -d $(BUILDDIR)/doctrees source

clean:
	rm -r $(BUILDDIR)

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
