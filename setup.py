# setup.py
import os, sys
from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/mnt/l/Coding_Projects/Liars_bar_2/Liars-bar/persistent_cache/inductor")
os.environ.setdefault("TRITON_CACHE_DIR",        "/mnt/l/Coding_Projects/Liars_bar_2/Liars-bar/persistent_cache/triton")

PROJ_ROOT     = os.path.dirname(os.path.abspath(__file__))
CPP_INCLUDE   = os.path.join(PROJ_ROOT, "src", "cpp", "include")
CPP_SRC_DIR   = os.path.join(PROJ_ROOT, "src", "cpp", "src")
BINDINGS_DIR  = os.path.join(PROJ_ROOT, "src", "cpp", "bindings")
SOURCES = sorted(glob(os.path.join(CPP_SRC_DIR, "*.cpp")) +
                 glob(os.path.join(BINDINGS_DIR, "*.cpp")))

# Toggle with: PROFILE=1 python setup.py build_ext -i
PROFILE = 0

def linux_macos_flags(profile: bool):
    if profile:
        cxx = ["-O2", "-g", "-fno-omit-frame-pointer", "-fno-lto", "-std=c++17", "-UNDEBUG"]
        link = ["-fno-lto"]
    else:
        cxx = ["-O3", "-DNDEBUG", "-std=c++17"]
        link = []
    return {"cxx": cxx}, link

def windows_flags(profile: bool):
    # /Zi: debug symbols, /Zo: enhanced optimized debugging, /Oy-: keep frame ptrs
    if profile:
        cxx = ["/O2", "/Zi", "/Zo", "/Oy-", "/std:c++17"]
        link = ["/DEBUG"]     # generate PDB
        # Avoid LTCG in profiling: /GL disables, so don't add it
    else:
        cxx = ["/O2", "/DNDEBUG", "/std:c++17"]
        link = []
    return {"cxx": cxx}, link

if sys.platform == "win32":
    extra_compile_args, extra_link_args = windows_flags(PROFILE)
else:
    extra_compile_args, extra_link_args = linux_macos_flags(PROFILE)

ext = CppExtension(
    name="src.misc.lb",
    sources=SOURCES,
    include_dirs=[CPP_INCLUDE],
    extra_compile_args=extra_compile_args,   # dict form is supported (cxx/nvcc)
    extra_link_args=extra_link_args,
)

setup(
    name="LiarsBarCore",
    version="1.0.3",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
