import os
from setuptools import setup
import subprocess


def get_long_description():
    this_directory = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    return long_description

subprocess.check_call('python install_all.py', shell=True)

setup(name='eo-box',
      python_requires='>=3.5,<3.7',
      version='0.1.0',
      description='A toolbox for processing earth observation data with Python.',
      long_description=get_long_description(),
      long_description_content_type='text/markdown',
      url='https://github.com/benmack/eo-box',
      author='Benjamin Mack',
      author_email='ben8mack@gmail.com',
      license='MIT',
      packages=[],
      install_requires=[
          'eo-box-sampledata',
          'eo-box-raster',
          ],
      zip_safe=False)
