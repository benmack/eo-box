# Makefile for creating a new release of the package and uploading it to PyPI

PYTHON = python3
PACKAGES = sampledata, raster

.PHONY: $(PACKAGES:test)

help:
	@echo "Use 'make upload-<package>' to upload the package to PyPi"

.ONESHELL:
build-sampledata:
	cd sampledata
	rm -r dist
	$(PYTHON) setup.py sdist

.ONESHELL:
build-raster:
	cd raster
	rm -r dist
	$(PYTHON) setup.py sdist

.ONESHELL:
build-abstract-package:
	rm -r dist
	$(PYTHON) setup.py sdist

upload-sampledata: build-sampledata
	twine upload sampledata/dist/*

upload-raster: build-raster
	twine upload raster/dist/*

upload-abstract-package: build-abstract-package
	twine upload dist/*

upload-all: \
 	upload-sampledata \
 	upload-raster \
	upload-abstract-package

# For testing:

test-upload-sampledata: build-sampledata
	twine upload --repository testpypi sampledata/dist/*

test-upload-raster: build-raster
	twine upload --repository testpypi raster/dist/*

test-upload-abstract-package: build-abstract-package
	twine upload --repository testpypi dist/*

test-upload-all: \
 	test-upload-sampledata \
 	test-upload-raster \
	test-upload-abstract-package
