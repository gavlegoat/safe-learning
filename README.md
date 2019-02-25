# VRL

VRL is a toolbox for verification in reinforcement learning systems.  

## Install
Require Python2.7 and usual python packages such as numpy, scipy, matplotlib, ...  

### pip install:
1. pip install tensorflow  
2. pip install tflearn   
3. pip install numba . 

### Install Z3:
1. From https://github.com/Z3Prover/z3/releases, download Z3 binary.   
2. `export PYTHONPATH=PATH_UNPACK_Z3/z3/bin/:` . 

### Install Julia-0.6
1. Download Julia-0.6 from https://julialang.org/downloads/oldreleases.html.  
2. `export PATH="$PATH:PATH_UNPACK_JULIA/julia/bin"`

### Install SumOfSqure tool:

Download and licsence the Mosek solver 

   - Download the MAC OS 64bit x86 MOSEK Optimization Suite distribution from https://mosek.com/downloads/ and unpack it into a chosen directory.  

   - For MAC, run the command:  
   > `python <MSKHOME>/mosek/8/tools/platform/osx64x86/bin/install.py`  
   > where `<MSKHOME>` is the directory where MOSEK was installed. This will set up the appropriate shared objects required when using MOSEK.  

   - Add the path `<MSKHOME>/mosek/8/tools/platform/osx64x86/bin` to the OS variable PATH.  

   - License the tool.  

### Configure Julia  
1. Enter Julia command line.  

2. Enter following commands:  

``` julia  
julia> Pkg.status()  

julia> Pkg.add("SumOfSquares")
julia> Pkg.status("SumOfSquares")
SumOfSquares 0.2.0

julia> Pkg.status("JuMP")
JuMP 0.18.2

julia> Pkg.add("MathOptInterface")
julia> Pkg.add("DynamicPolynomials")
julia> Pkg.add("SCS")
julia> Pkg.add("MathOptInterfaceMosek")
julia> Pkg.add("Clp") 
```
