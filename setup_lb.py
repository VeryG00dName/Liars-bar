# setup_lb.py
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob
import os

PROJ_ROOT = os.path.dirname(__file__)
SRC_DIR = os.path.join(PROJ_ROOT, "src", "cpp_build")
sources = sorted(glob(os.path.join(SRC_DIR, "*.cpp")))

ext = Pybind11Extension(
    name="src.misc.lb",          # import path
    sources=sources,             # all your .cpp files
    include_dirs=[SRC_DIR],      # headers in src/cpp_build
    cxx_std=17,
    extra_compile_args=["-O3", "-fPIC", "-fvisibility=hidden"],
)

setup(
    name="src.misc.lb",
    version="0.0.1",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
