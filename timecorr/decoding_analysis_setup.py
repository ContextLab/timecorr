from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(
  name = 'decoding analysis',
  include_dirs=[numpy.get_include()],
  ext_modules = cythonize("decoding_analysis.pyx"),
)
