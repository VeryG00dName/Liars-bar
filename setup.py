# setup.py
import sys
from setuptools import setup
from glob import glob
import os
# --- NEW IMPORTS ---
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

# --- Define project structure (unchanged) ---
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
CPP_INCLUDE_DIR = os.path.join(PROJ_ROOT, "src", "cpp", "include")
CPP_SRC_DIR = os.path.join(PROJ_ROOT, "src", "cpp", "src")
BINDINGS_DIR = os.path.join(PROJ_ROOT, "src", "cpp", "bindings")

# --- Collect all C++ source files (unchanged) ---
sources = sorted(
    glob(os.path.join(CPP_SRC_DIR, "*.cpp")) +
    glob(os.path.join(BINDINGS_DIR, "*.cpp"))
)

# --- Platform-specific compiler arguments (can be simplified) ---
# The CppExtension will handle many of these flags for us
extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ['/O2', '/std:c++17']
else:
    # We can keep optimization flags, but CppExtension will add the crucial ones.
    extra_compile_args = ['-O3', '-std=c++17']

# --- NEW EXTENSION DEFINITION ---
ext = CppExtension(
    # The name defines the import path
    name="src.misc.lb",
    sources=sources,
    # CppExtension automatically finds Pybind11 headers if it's installed
    # It also automatically finds torch headers. We just need our own.
    include_dirs=[CPP_INCLUDE_DIR],
    extra_compile_args=extra_compile_args,
    # Note: We no longer need to manually specify libtorch paths or libraries!
    # CppExtension finds them automatically from your torch installation.
)

setup(
    name="LiarsBarCore",
    version="1.0.2", # Incremented for the build system change
    author="<Your Name>",
    author_email="<your_email@example.com>",
    description="Core C++ components for the Liar's Bar project",
    ext_modules=[ext],
    # Use the custom BuildExtension from torch
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)