import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import sys


if sys.platform == 'darwin':
    ext_modules = [Extension(
        name="fasttsne",
        sources=["orig-lvdm/quadtree.cpp", "orig-lvdm/tsne.cpp", "fasttsne.pyx"],
        include_dirs = [numpy.get_include(), "orig-lvdm/"],
        extra_compile_args=['-I/System/Library/Frameworks/vecLib.framework/Headers', '-DMACOSX'],
        extra_link_args=["-Wl,-framework", "-Wl,Accelerate", "-lcblas"],
        language="c++"
        )]
else:
    ext_modules = [Extension(
        name="fasttsne",
        sources=["orig-lvdm/quadtree.cpp", "orig-lvdm/tsne.cpp", "fasttsne.pyx"],
        include_dirs = [numpy.get_include(), "/usr/local/include", "orig-lvdm/"],
        library_dirs = ["/usr/local/lib"],
        extra_compile_args=['-march=native', '-O3', '-ffast-math', '-fPIC', '-w', '-ggdb'],
        extra_link_args=["-lcblas"],
        language="c++"
        )]

setup(
    name = "fasttsne",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules,
    )
