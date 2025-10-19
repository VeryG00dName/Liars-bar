# setup.py
import os, sys
from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/mnt/l/Coding_Projects/Liars_bar_2/Liars-bar/persistent_cache/inductor")
os.environ.setdefault("TRITON_CACHE_DIR",        "/mnt/l/Coding_Projects/Liars_bar_2/Liars-bar/persistent_cache/triton")

PROJ_ROOT     = os.path.dirname(os.path.abspath(__file__))
CPP_INCLUDE   = os.path.join(PROJ_ROOT, "src", "cpp", "include")
CPP_SRC_DIR   = os.path.join(PROJ_ROOT, "src", "cpp", "src")
BINDINGS_DIR  = os.path.join(PROJ_ROOT, "src", "cpp", "bindings")
SOURCES = sorted(
    glob(os.path.join(CPP_SRC_DIR, "*.cpp"))
    + glob(os.path.join(CPP_SRC_DIR, "*.cu"))
    + glob(os.path.join(BINDINGS_DIR, "*.cpp"))
)

# Toggle with: PROFILE=1 python setup.py build_ext -i
PROFILE = 0

def linux_macos_flags(profile: bool):
    if profile:
        cxx = ["-O2", "-g", "-fno-omit-frame-pointer", "-fno-lto", "-std=c++17", "-UNDEBUG"]
        nvcc = ["-O2", "-lineinfo"]
        link = ["-fno-lto"]
    else:
        cxx = ["-O3", "-DNDEBUG", "-std=c++17"]
        nvcc = ["-O3"]
        link = []
    return {"cxx": cxx, "nvcc": nvcc}, link

def windows_flags(profile: bool):
    # /Zi: debug symbols, /Zo: enhanced optimized debugging, /Oy-: keep frame ptrs
    if profile:
        cxx = ["/O2", "/Zi", "/Zo", "/Oy-", "/std:c++17"]
        link = ["/DEBUG"]     # generate PDB
        # Avoid LTCG in profiling: /GL disables, so don't add it
    else:
        cxx = ["/O2", "/DNDEBUG", "/std:c++17"]
        link = []
    # NVCC flags are unused on Windows where CUDA builds are not supported here.
    return {"cxx": cxx}, link

if sys.platform == "win32":
    extra_compile_args, extra_link_args = windows_flags(PROFILE)
else:
    extra_compile_args, extra_link_args = linux_macos_flags(PROFILE)

# Resolve CUDA paths explicitly (WSL-friendly)
cuda_home = CUDA_HOME or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
cuda_ver_suffix = os.environ.get("CUDA_VER_SUFFIX", "")  # e.g., "-12.9" if you prefer versioned path
cuda_root_candidates = [
    cuda_home,
    f"{cuda_home}{cuda_ver_suffix}",
    "/usr/local/cuda-12.9",
    "/usr/local/cuda-12.8",
]
cuda_include_dirs = []
cuda_lib_dirs = []
for root in cuda_root_candidates:
    inc1 = os.path.join(root, "include")
    inc2 = os.path.join(root, "targets", "x86_64-linux", "include")
    lib1 = os.path.join(root, "lib64")
    lib2 = os.path.join(root, "targets", "x86_64-linux", "lib")
    for p in (inc1, inc2):
        if os.path.isdir(p) and p not in cuda_include_dirs:
            cuda_include_dirs.append(p)
    for p in (lib1, lib2):
        if os.path.isdir(p) and p not in cuda_lib_dirs:
            cuda_lib_dirs.append(p)

# Prepend CUDA include dirs for NVCC specifically as well
if "nvcc" in extra_compile_args:
    for p in cuda_include_dirs:
        extra_compile_args["nvcc"].insert(0, f"-I{p}")

ext = CUDAExtension(
    name="src.misc.lb",
    sources=SOURCES,
    include_dirs=[CPP_INCLUDE] + cuda_include_dirs,
    library_dirs=cuda_lib_dirs,
    libraries=['cublasLt'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="LiarsBarCore",
    version="1.0.3",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
