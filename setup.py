from distutils.core import setup, Extension
import os

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

module1 = Extension('synthesis',
                    include_dirs = ['/home/greg/cegar-net/eigen', '/home/greg/apron-install/include', '/home/greg/include',
                        '/home/greg/apron/taylor1plus/'],
                    library_dirs = ['/home/greg/apron-install/lib', '/home/greg/lib'],
                    libraries = ['gmp', 'mpfr', 'apron', 't1pD', 'boxD', 'polkaMPQ'],
                    sources = ['abstract.cpp', 'synthesis.cpp'],
                    extra_compile_args = ['-std=c++17', '-g', '-O0'],
                    runtime_library_dirs = ['/home/greg/apron-install/lib', '/home/greg/lib']
                    )

setup(name = 'Shield Synthesis',
      version = '0.1',
      description = 'Shield synthesis for RL systems',
      ext_modules = [module1])

