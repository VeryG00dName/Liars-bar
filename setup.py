# Build configuration for the Liars Bar C++ extension.
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob
import os

# --- Define project structure ---
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
# Core C++ library headers
CPP_INCLUDE_DIR = os.path.join(PROJ_ROOT, "src", "cpp", "include")
# Core C++ library source files
CPP_SRC_DIR = os.path.join(PROJ_ROOT, "src", "cpp", "src")
# Pybind11 binding source files
BINDINGS_DIR = os.path.join(PROJ_ROOT, "src", "cpp", "bindings")

# --- Collect all C++ source files ---
# This will find all .cpp files in both the library source and bindings directories
sources = sorted(
    glob(os.path.join(CPP_SRC_DIR, "*.cpp")) +
    glob(os.path.join(BINDINGS_DIR, "*.cpp"))
)

# --- Platform-specific compiler arguments ---
if sys.platform == "win32":
    # MSVC compiler flags
    extra_compile_args = ["/O2", "/std:c++17"]
else:
    # GCC/Clang compiler flags
    extra_compile_args = ["-O3", "-fPIC", "-fvisibility=hidden", "-std=c++17"]

ext = Pybind11Extension(
    # The name defines the import path in Python: from src.misc import lb
    name="src.misc.lb",
    sources=sources,
    # Tell the compiler where to find header files (.h)
    include_dirs=[CPP_INCLUDE_DIR],
    # Use C++17 standard
    cxx_std=17,
    # Pass the platform-specific flags
    extra_compile_args=extra_compile_args,
)

setup(
    name="LiarsBarCore",  # A more descriptive name for the package
    version="1.0.0",
    author="<Your Name>",
    author_email="<your_email@example.com>",
    description="Core C++ components for the Liar's Bar project",
    ext_modules=[ext],
    # Tell setuptools how to run the build
    cmdclass={"build_ext": build_ext},
    # Add zip_safe=False for good measure with C++ extensions
    zip_safe=False,
)
