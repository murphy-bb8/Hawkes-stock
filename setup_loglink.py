"""
setup_loglink.py — 编译 hawkes_cy_loglink.pyx
==============================================
运行：
  python setup_loglink.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "hawkes_cy_loglink",
        ["hawkes_cy_loglink.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["/O2"] if __import__("sys").platform == "win32" else ["-O3", "-march=native"],
    )
]

setup(
    name="hawkes_cy_loglink",
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'initializedcheck': False,
    }),
    include_dirs=[np.get_include()],
)
