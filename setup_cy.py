"""编译 Cython 加速模块: python setup_cy.py build_ext --inplace"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "_hawkes_cy",
        ["_hawkes_cy.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="_hawkes_cy",
    ext_modules=cythonize(extensions, compiler_directives={
        "boundscheck": False, "wraparound": False, "cdivision": True,
    }),
)
