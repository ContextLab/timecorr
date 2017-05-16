from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(
  name = 'helper',
  include_dirs=[numpy.get_include()],
  ext_modules = cythonize("helpers.pyx"),
)
