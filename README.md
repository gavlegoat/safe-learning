# VRL

VRL is a toolbox for verification in reinforcement learning systems.  

## Install
Require Python2.7 and usual python packages such as numpy, scipy, matplotlib, ...  

### Install Tensorflow and tflearn:
1. pip install tensorflow  
   pip install tflearn  

### Install Z3:
1. From https://github.com/Z3Prover/z3/releases, download Z3 binary for MacOS (source code not required) to z3-macos/  
2. export PYTHONPATH=`pwd`/z3-macos/bin/:  

### Install SumOfSqure tool:
0. brew cask uninstall julia  
0. rm -rf ~/.julia/  
0. rm -rf /Applications/Julia-*.*.app/  

1. Download and licsence the Mosek solver (Mac OS)  

Download the MAC OS 64bit x86 MOSEK Optimization Suite distribution from https://mosek.com/downloads/ and unpack it into a chosen directory.  

Run the command:  
python <MSKHOME>/mosek/8/tools/platform/osx64x86/bin/install.py  
where <MSKHOME> is the directory where MOSEK was installed. This will set up the appropriate shared objects required when using MOSEK.  

Optionally add the path  

<MSKHOME>/mosek/8/tools/platform/osx64x86/bin  
to the OS variable PATH.  

License the tool.  

2. Download Julia-0.6 to /Applications/Julia-0.6.app/  

3. /Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia  

/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia --version  
julia version 0.6.4

4. julia> Pkg.status()  

5. julia> Pkg.add("SumOfSquares")  

julia> Pkg.status("SumOfSquares")  
 - SumOfSquares                  0.2.0  

julia> Pkg.status("JuMP")  
 - JuMP                          0.18.2  

6. julia> Pkg.add("MathOptInterface")  
7. julia> Pkg.add("DynamicPolynomials")  
8. julia> Pkg.add("SCS")
9. julia> Pkg.add("MathOptInterfaceMosek")  
10.julia> Pkg.add("Clp")  
