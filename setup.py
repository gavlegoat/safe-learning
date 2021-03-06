from distutils.core import setup, Extension
import os

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

module1 = Extension('synthesis',
                    include_dirs = ['/home/greg/Documents/eigen'],
                    libraries = ['gmp', 'mpfr', 'apron', 't1pD', 'boxD', 'polkaMPQ'],
                    sources = ['abstract.cpp', 'synthesis.cpp'],
                    extra_compile_args = ['-std=c++17', '-g', '-O0']
                    )

setup(name = 'Shield Synthesis',
      version = '0.1',
      description = 'Shield synthesis for RL systems',
      ext_modules = [module1])

