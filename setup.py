# setup.py
import os, sys
from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

# Cache dirs (optional, helps perf)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/mnt/l/Coding_Projects/Liars_bar_2/Liars-bar/persistent_cache/inductor")
os.environ.setdefault("TRITON_CACHE_DIR",        "/mnt/l/Coding_Projects/Liars_bar_2/Liars-bar/persistent_cache/triton")

PROJ_ROOT    = os.path.dirname(os.path.abspath(__file__))
CPP_INCLUDE  = os.path.join(PROJ_ROOT, "src", "cpp", "include")
CPP_SRC_DIR  = os.path.join(PROJ_ROOT, "src", "cpp", "src")
BINDINGS_DIR = os.path.join(PROJ_ROOT, "src", "cpp", "bindings")

SOURCES = sorted(
    glob(os.path.join(CPP_SRC_DIR, "*.cpp"))
    + glob(os.path.join(CPP_SRC_DIR, "*.cu"))
    + glob(os.path.join(BINDINGS_DIR, "*.cpp"))
)

PROFILE = int(os.getenv("PROFILE", "0"))  # PROFILE=1 python setup.py build_ext -i


def linux_macos_flags(profile: bool):
    if profile:
        cxx  = ["-O2", "-g", "-fno-omit-frame-pointer", "-fno-lto", "-std=c++17", "-UNDEBUG"]
        nvcc = ["-O0", "-G", "-lineinfo", "-Xptxas", "-O0"]
        link = ["-fno-lto"]
    else:
        cxx  = ["-O3", "-DNDEBUG", "-std=c++17"]
        nvcc = ["-O3"]
        link = []
    return {"cxx": cxx, "nvcc": nvcc}, link


def windows_flags(profile: bool):
    if profile:
        cxx  = ["/O2", "/Zi", "/Zo", "/Oy-", "/std:c++17"]
        link = ["/DEBUG"]
    else:
        cxx  = ["/O2", "/DNDEBUG", "/std:c++17"]
        link = []
    return {"cxx": cxx}, link


if sys.platform == "win32":
    extra_compile_args, extra_link_args = windows_flags(PROFILE)
else:
    extra_compile_args, extra_link_args = linux_macos_flags(PROFILE)


def _ensure_flag(lst, flag):
    if flag not in lst:
        lst.append(flag)


# ---- CUTLASS arches/macros (host & nvcc) ----
for key in ("cxx", "nvcc"):
    if key in extra_compile_args:
        _ensure_flag(extra_compile_args[key], "-DCUTLASS_ENABLE_SM86=1")
        _ensure_flag(extra_compile_args[key], "-DCUTLASS_ENABLE_SM80=1")
        _ensure_flag(extra_compile_args[key], '-DCUTLASS_ARCHS="86"')  # restrict templates to 86


# ---- CUDA include/lib dirs (WSL-friendly) ----
cuda_home = CUDA_HOME or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
cuda_candidates = [
    cuda_home,
    f"{cuda_home}{os.environ.get('CUDA_VER_SUFFIX', '')}",  # e.g. "-13.0"
    "/usr/local/cuda-13.0",
]
cuda_include_dirs, cuda_lib_dirs = [], []
for root in cuda_candidates:
    incs = [os.path.join(root, "include"),
            os.path.join(root, "targets", "x86_64-linux", "include")]
    libs = [os.path.join(root, "lib64"),
            os.path.join(root, "targets", "x86_64-linux", "lib")]
    for p in incs:
        if os.path.isdir(p) and p not in cuda_include_dirs:
            cuda_include_dirs.append(p)
    for p in libs:
        if os.path.isdir(p) and p not in cuda_lib_dirs:
            cuda_lib_dirs.append(p)


# ---- CUTLASS: choose EXACTLY ONE include root (prefer <root>/include) ----
def pick_cutlass_include_dir(candidates):
    # Prefer <root>/include/cutlass
    for root in candidates:
        if not root:
            continue
        inc = os.path.join(root, "include")
        if os.path.isdir(os.path.join(inc, "cutlass")):
            return [inc]
    # Fallback: <root>/cutlass (older layouts)
    for root in candidates:
        if not root:
            continue
        if os.path.isdir(os.path.join(root, "cutlass")):
            return [root]
    return []

cutlass_candidates = [
    os.environ.get("CUTLASS_PATH"),
    os.path.join(PROJ_ROOT, "third_party", "cutlass"),
]
cutlass_include_dirs = pick_cutlass_include_dir(cutlass_candidates)
if not cutlass_include_dirs:
    raise RuntimeError(
        "CUTLASS headers not found. Set CUTLASS_PATH or place third_party/cutlass "
        "with either <root>/include/cutlass or <root>/cutlass."
    )

# Make sure NVCC also sees all include dirs (PyTorch passes them, but we’re explicit).
if "nvcc" in extra_compile_args:
    for p in cuda_include_dirs + cutlass_include_dirs + [CPP_INCLUDE]:
        flag = f"-I{p}"
        if flag not in extra_compile_args["nvcc"]:
            extra_compile_args["nvcc"].insert(0, flag)
    # Target Ampere (86)
    for g in ("-gencode=arch=compute_86,code=compute_86",
              "-gencode=arch=compute_86,code=sm_86"):
        _ensure_flag(extra_compile_args["nvcc"], g)

# Final include dirs: one CUTLASS dir only
include_dirs = [CPP_INCLUDE] + cuda_include_dirs + cutlass_include_dirs

ext = CUDAExtension(
    name="src.misc.lb",
    sources=SOURCES,
    include_dirs=include_dirs,
    library_dirs=cuda_lib_dirs,
    libraries=["cublasLt"],
    define_macros=[
        ("CUTLASS_ENABLE_SM86", "1"),
        ("CUTLASS_ENABLE_SM80", "1"),
        ("TORCH_USE_CUDA_DSA", "1"),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="LiarsBarCore",
    version="2.0.0",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
