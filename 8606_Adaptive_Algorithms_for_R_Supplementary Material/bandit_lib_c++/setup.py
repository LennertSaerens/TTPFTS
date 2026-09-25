import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extra_compile_args = ["-std=c++17", "-march=native", ] #"-ffast-math"
language = "c++"
setup(name="bandit_cxx",
      ext_modules=cythonize([
          Extension(
              name="lib.bandits",
              sources=["lib/bandits.pyx",
                       "src/cpp/utils.cxx"],
              language=language,
              extra_compile_args=extra_compile_args
          ),
          Extension(
              name="lib.policies",
              sources=["lib/policies.pyx",
                       "src/cpp/utils.cxx"],
              language=language,
              extra_compile_args=extra_compile_args
          )
      ]),
      language_level=3,
      include_dirs=[np.get_include()]
      )
