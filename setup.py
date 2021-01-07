from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import urllib.request
import subprocess
from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
import zipfile

__version__ = '0.0.5'


def download_eigen():
    zippath = os.path.join('deps', 'eigen.zip')
    if not os.path.exists('deps'): os.mkdir('deps')
    if not os.path.exists(zippath):
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Chrome')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve("https://gitlab.com/libeigen/eigen/-/archive/3.3.4/eigen-3.3.4.zip", zippath)
    
    f = zipfile.ZipFile(zippath)
    f.extractall('deps')
    return os.path.join('deps', "eigen-3.3.4")
    
def get_eigen():
    if 'EIGEN3_INCLUDE_DIR' in os.environ:
        return os.environ['EIGEN3_INCLUDE_DIR']
    return download_eigen()
    
def download_pybind():
    zippath = os.path.join('deps', 'pybind.zip')
    if not os.path.exists('deps'): os.mkdir('deps')
    if not os.path.exists(zippath):
        urlretrieve("https://github.com/pybind/pybind11/archive/v2.2.1.zip", zippath)
    
    f = zipfile.ZipFile(zippath)
    f.extractall('deps')
    return os.path.join('deps', "pybind11-2.2.1", "include")

def get_pybind():
    if 'PYBIND11_INCLUDE_DIR' in os.environ:
        return os.environ['PYBIND11_INCLUDE_DIR']
    return download_pybind()
 
# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++14 support '
                           'is needed!')

if sys.platform == 'win32' and sys.version_info > (2, 6):
   # 2.6's distutils.msvc9compiler can raise an IOError when failing to
   # find the compiler
   # It can also raise ValueError http://bugs.python.org/issue7511
   ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError,
                 IOError, ValueError)
else:
   ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)

class BuildFailed(Exception): pass

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    defopts = []
    c_opts = {
        'msvc': ['/EHsc', '/D_USE_MATH_DEFINES'] + defopts,
        'unix': defopts[:],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def _do_build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

    def build_extensions(self):
        try:
            return self._do_build_extensions()
        except ext_errors:
            raise BuildFailed()

def try_setup(build_binary):
    params = dict(
        name='nslr',
        version=__version__,
        author='Jami Pekkanen',
        author_email='jami.pekkanen@gmail.com',
        url='https://gitlab.com/nslr/nslr',
        description='Naive Segmented Linear Regression',
        long_description='',
        install_requires=[],
        packages=['nslr'],
        zip_safe=False,
    platforms=['any'],
    )
    if not build_binary:
        return setup(**params)
    
    ext_modules = [
        Extension(
            'nslr.cppnslr',
            ['nslr/cppnslr.cpp'],
            include_dirs=[
                # Try to get eigen include directory. Set
                # EIGEN3_INCLUDE_DIR environment variable to
                # set manually.
                get_eigen(),
		# Try to get pybind11 directory. Set
                # PYBIND11_INCLUDE_DIR environment variable to
                # set manually.
		get_pybind(),
            ],
            language='c++'
        ),
    ]
    
    params.update(**dict(
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExt}
    ))

    setup(**params)

try:
    try_setup(True)
except BuildFailed:
    print("Failed to build C++ extension.")
    raise
