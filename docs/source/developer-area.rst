## Contributer

TODO: Create best-practice ressource for contribution in a open source GitHub context. 

## Maintainer

### Release a new version

#### Prerequisites:

* Owner rights for the package on : https://test.pypi.org/manage/projects/

* Owner rights for the package on : https://pypi.org/manage/projects/

#### Steps

##### Develop branch

* Update the changelog and set a new version number.

* Make sure you have updated / synchronized the *README.md* and the info in *docs/source/index.rst* and *docs/source/install.rst* until the last two are not automatically taken from the *README.md*.

* Follow the [guide for generating and uploading distribution archives](https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives) for testing from the develop branch.

* Install the new version and try it out:

  `python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps eobox==0.3.2`

##### Master branch

* Merge the develop branch into master

* Create a new tag `git tag -a v0.3.2 -m "new release v0.3.2"` & `git push origin --tags`

* Create a new release on GitHub and upload files

* Upload to PyPi from master without `python3 -m twine upload dist/*`.

## Docker

Steps

* set version
* push
* pull clean
* docker image build -t benmack/eobox-notebook:latest -t benmack/eobox-notebook:v0.0.2 -f docker/eobox-notebook.dockerfile .
* push