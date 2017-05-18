from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
extensions = [
    Extension("helpers", ["helpers.pyx"])
]

setup(
  name = 'helper',
  include_dirs=[numpy.get_include()],
  ext_modules = cythonize(extensions),
)
