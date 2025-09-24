# setup.py
# Build configuration for the Liars Bar C++ extension.
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob
import os

# --- Define project structure ---
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
CPP_INCLUDE_DIR = os.path.join(PROJ_ROOT, "src", "cpp", "include")
CPP_SRC_DIR = os.path.join(PROJ_ROOT, "src", "cpp", "src")
BINDINGS_DIR = os.path.join(PROJ_ROOT, "src", "cpp", "bindings")

# --- Collect all C++ source files ---
sources = sorted(
    glob(os.path.join(CPP_SRC_DIR, "*.cpp")) +
    glob(os.path.join(BINDINGS_DIR, "*.cpp"))
)

# ==============================================================================
# SECTION: Find and configure LibTorch (PyTorch C++ API)
# ==============================================================================
libtorch_home = os.environ.get("LIBTORCH_HOME")
if not libtorch_home or not os.path.isdir(libtorch_home):
    print(
        "Warning: LIBTORCH_HOME is not set or not a valid directory. "
        "The C++ extension will be built WITHOUT TorchScript support."
    )
    # Set empty paths if libtorch is not found
    libtorch_include_dirs = []
    libtorch_library_dirs = []
    extra_link_args = []
    torch_libs = []
else:
    print(f"Found LibTorch at: {libtorch_home}")
    libtorch_include_dirs = [
        os.path.join(libtorch_home, "include"),
        os.path.join(libtorch_home, "include", "torch", "csrc", "api", "include"),
    ]
    libtorch_library_dirs = [os.path.join(libtorch_home, "lib")]

    if sys.platform == "win32":
        # For Windows, linker arguments are typically not needed in extra_link_args
        # The library names with .lib are sufficient.
        extra_link_args = []
        # Names of the libraries to link against on Windows
        torch_libs = ["c10", "torch", "torch_cpu"]
        # Add the CUDA library if a GPU-enabled libtorch is used
        if os.path.exists(os.path.join(libtorch_home, "lib", "torch_cuda.lib")):
             torch_libs.append("torch_cuda")
    else:
        # For Linux/macOS, we often need to specify the rpath
        extra_link_args = [f"-Wl,-rpath,{libtorch_library_dirs[0]}"]
        # Names of the libraries to link against on Linux/macOS
        torch_libs = ["c10", "torch", "torch_cpu"]
        # Add the CUDA library if a GPU-enabled libtorch is used
        if os.path.exists(os.path.join(libtorch_home, "lib", "libtorch_cuda.so")):
             torch_libs.append("torch_cuda")
# ==============================================================================

# --- Platform-specific compiler arguments ---
if sys.platform == "win32":
    # MSVC compiler flags
    extra_compile_args = ["/O2", "/std:c++17", "/EHsc"] # Added /EHsc for exception handling
else:
    # GCC/Clang compiler flags
    extra_compile_args = ["-O3", "-fPIC", "-fvisibility=hidden", "-std=c++17"]

ext = Pybind11Extension(
    name="src.misc.lb",
    sources=sources,
    # Combine project headers with libtorch headers
    include_dirs=[CPP_INCLUDE_DIR] + libtorch_include_dirs,
    cxx_std=17,
    extra_compile_args=extra_compile_args,
    # Add libtorch linker arguments and library paths
    extra_link_args=extra_link_args,
    library_dirs=libtorch_library_dirs,
    libraries=torch_libs,
)

setup(
    name="LiarsBarCore",
    version="1.0.1", # Incremented version for the new feature
    author="<Your Name>",
    author_email="<your_email@example.com>",
    description="Core C++ components for the Liar's Bar project",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)