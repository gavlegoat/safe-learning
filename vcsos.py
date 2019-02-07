def genSOS(nvar, sysdynamics, init, unsafe, 
	initSOSPoly="@variable m Zinit SOSPoly(Z)", 
	unsafeSOSPoly="@variable m Zunsafe SOSPoly(Z)", 
	init_cnstr="- Zinit*init", 
	unsafe_cnstr="- Zunsafe*unsafe",
	degree=4):

	sos = """#Use this Julia module to generate barrier certificates.
using MathOptInterface
const MOI = MathOptInterface
using JuMP
using SumOfSquares
using PolyJuMP
using Base.Test
using MultivariatePolynomials
using SemialgebraicSets
using Mosek

import DynamicPolynomials.@polyvar

@polyvar x[1:{}]
f = [{}]
{}
{}

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:{})
@variable m B Poly(Z)

#=
x1 = x[1]+0.01*f[1]
x2 = x[2]+0.01*f[2]
x3 = x[3]+0.01*f[3]
x4 = x[4]+0.01*f[4]

B1 = subs(B, x[1]=>x1, x[2]=>x2, x[3]=>x3, x[4]=>x4)

f1 = B
f2 = -B
f3 = B - B1
=#

f1 = B{}
f2 = -B{}
f3 = -dot(differentiate(B, x), f)

@constraint m f1 >= 1
@constraint m f2 >= 0
@constraint m f3 >= 0

status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))""".format(nvar, sysdynamics, init, unsafe, initSOSPoly, unsafeSOSPoly, degree, unsafe_cnstr, init_cnstr)

	return sos


def genSOSwithBound(nvar, sysdynamics, init, unsafe, bound,
	initSOSPoly="@variable m Zinit SOSPoly(Z)", 
	unsafeSOSPoly="@variable m Zunsafe SOSPoly(Z)", 
	boundSOSPoly="@variable m Zbound SOSPoly(Z)",
	init_cnstr="- Zinit*init", 
	unsafe_cnstr="- Zunsafe*unsafe",
	bound_cnstr="- Zbound*bound",
	degree=4):

	sos = """#Use this Julia module to generate barrier certificates.
using MathOptInterface
const MOI = MathOptInterface
using JuMP
using SumOfSquares
using PolyJuMP
using Base.Test
using MultivariatePolynomials
using SemialgebraicSets
using Mosek

import DynamicPolynomials.@polyvar

@polyvar x[1:{}]
f = [{}]
{}
{}
{}

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:{})
@variable m B Poly(Z)

f1 = B{}
f2 = -B{}
f3 = -dot(differentiate(B, x), f){}

@constraint m f1 >= 1
@constraint m f2 >= 0
@constraint m f3 >= 0

status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))""".format(nvar, sysdynamics, init, unsafe, bound, initSOSPoly, unsafeSOSPoly, boundSOSPoly, degree, unsafe_cnstr, init_cnstr, bound_cnstr)

	return sos

def genSOSwithDisturbance(nvar, sysdynamics, init, unsafe, bound,
	initSOSPoly="@variable m Zinit SOSPoly(Z)", 
	unsafeSOSPoly="@variable m Zunsafe SOSPoly(Z)", 
	boundSOSPoly="@variable m Zbound SOSPoly(Z)",
	init_cnstr="- Zinit*init", 
	unsafe_cnstr="- Zunsafe*unsafe",
	bound_cnstr="- Zbound*bound",
	degree=4):

	allvars = ""
	for i in range(nvar):
		allvars += "x[" + str(i+1) + "],"
	for i in range(nvar):
		if i == nvar -1:
			allvars += "d[" + str(i+1) + "]"
		else:
			allvars += "d[" + str(i+1) + "],"

	sos = """#Use this Julia module to generate barrier certificates.
using MathOptInterface
const MOI = MathOptInterface
using JuMP
using SumOfSquares
using PolyJuMP
using Base.Test
using MultivariatePolynomials
using SemialgebraicSets
using Mosek

import DynamicPolynomials.@polyvar

@polyvar d[1:{}]
@polyvar x[1:{}]
f = [{}]
{}
{}
{}

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:1)
{}
D = monomials([{}], 0:1)
{}
Z = monomials(x, 0:{})
@variable m B Poly(Z)

f1 = B{}
f2 = -B{}
f3 = -dot(differentiate(B, x), f){}

@constraint m f1 >= 1
@constraint m f2 >= 0
@constraint m f3 >= 0

status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))""".format(nvar, nvar, sysdynamics, init, unsafe, bound, initSOSPoly, unsafeSOSPoly, allvars, boundSOSPoly, degree, unsafe_cnstr, init_cnstr, bound_cnstr)

	return sos

'''
From this line, new code to satisfy the need to generate correct barrier certificates efficiently
'''

def genSOSContinuousOneUnsafe(nvar, sysdynamics, init, unsafe, 
	initSOSPoly="@variable m Zinit SOSPoly(Z)", 
	unsafeSOSPoly="@variable m Zunsafe SOSPoly(Z)", 
	init_cnstr="- Zinit*init", 
	unsafe_cnstr="- Zunsafe*unsafe",
	degree=4):

	sos = """#Use this Julia module to generate barrier certificates.
using MathOptInterface
const MOI = MathOptInterface
using JuMP
using SumOfSquares
using PolyJuMP
using Base.Test
using MultivariatePolynomials
using SemialgebraicSets
using Mosek

import DynamicPolynomials.@polyvar

@polyvar x[1:{}]
f = [{}]
{}
{}

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:{})
@variable m B Poly(Z)

f1 = B{}
f2 = -B{}
f3 = -dot(differentiate(B, x), f)

@constraint m f1 >= 1
@constraint m f2 >= 0
@constraint m f3 >= 0

status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))""".format(nvar, sysdynamics, init, unsafe, initSOSPoly, unsafeSOSPoly, degree, unsafe_cnstr, init_cnstr)

	return sos

def genSOSContinuousAsDiscreteOneUnsafe(timestep, nvar, sysdynamics, init, unsafe, 
	initSOSPoly="@variable m Zinit SOSPoly(Z)", 
	unsafeSOSPoly="@variable m Zunsafe SOSPoly(Z)", 
	init_cnstr="- Zinit*init", 
	unsafe_cnstr="- Zunsafe*unsafe",
	degree=4):

	transition = ""
	subs = ""
	for i in range(nvar):
		transition += ("x" + str(i+1) + " = x[" + str(i+1) + "] + " + str(timestep) + "*f[" + str(i+1) + "]\n")
		if i == 0:
			subs += "x[" + str(i+1) + "]=>x" + str(i+1)
		else:
			subs += ", x[" + str(i+1) + "]=>x" + str(i+1)

	transitionconstraint = transition + "\nB1 = subs(B, " + subs + ")"

	sos = """#Use this Julia module to generate barrier certificates.
using MathOptInterface
const MOI = MathOptInterface
using JuMP
using SumOfSquares
using PolyJuMP
using Base.Test
using MultivariatePolynomials
using SemialgebraicSets
using Mosek

import DynamicPolynomials.@polyvar

@polyvar x[1:{}]
f = [{}]
{}
{}

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:{})
@variable m B Poly(Z)

{}

f1 = B{}
f2 = -B{}
f3 = B - B1

@constraint m f1 >= 1
@constraint m f2 >= 0
@constraint m f3 >= 0

status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))""".format(nvar, sysdynamics, init, unsafe, initSOSPoly, unsafeSOSPoly, degree, transitionconstraint, unsafe_cnstr, init_cnstr)

	return sos

def genSOSDiscreteOneUnsafe(nvar, sysdynamics, init, unsafe, 
	initSOSPoly="@variable m Zinit SOSPoly(Z)", 
	unsafeSOSPoly="@variable m Zunsafe SOSPoly(Z)", 
	init_cnstr="- Zinit*init", 
	unsafe_cnstr="- Zunsafe*unsafe",
	degree=4):

	
	transition = ""
	subs = ""
	for i in range(nvar):
		transition += ("x" + str(i+1) + " = f[" + str(i+1) + "]\n")
		if i == 0:
			subs += "x[" + str(i+1) + "]=>x" + str(i+1)
		else:
			subs += ", x[" + str(i+1) + "]=>x" + str(i+1)

	transitionconstraint = transition + "\nB1 = subs(B, " + subs + ")"

	sos = """#Use this Julia module to generate barrier certificates.
using MathOptInterface
const MOI = MathOptInterface
using JuMP
using SumOfSquares
using PolyJuMP
using Base.Test
using MultivariatePolynomials
using SemialgebraicSets
using Mosek

import DynamicPolynomials.@polyvar

@polyvar x[1:{}]
f = [{}]
{}
{}

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:{})
@variable m B Poly(Z)

{}

f1 = B{}
f2 = -B{}
f3 = B - B1

@constraint m f1 >= 1
@constraint m f2 >= 0
@constraint m f3 >= 0

status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))""".format(nvar, sysdynamics, init, unsafe, initSOSPoly, unsafeSOSPoly, degree, transitionconstraint, unsafe_cnstr, init_cnstr)

	return sos

def genSOSContinuousMultipleUnsafes(nvar, sysdynamics, init, unsafe, 
	initSOSPoly="@variable m Zinit SOSPoly(Z)", 
	unsafeSOSPoly="@variable m Zunsafe SOSPoly(Z)", 
	init_cnstr="- Zinit*init", 
	unsafe_cnstrs=["- Zunsafe*unsafe"],
	degree=4):

	unsafe_vcs = "" 
	unsafe_constraints = ""
	i = 3
	for unsafe_it in unsafe_cnstrs:
		unsafe_vcs += ("f" + str(i) + " = B" + unsafe_it + "\n")
		unsafe_constraints += ("@constraint m f" + str(i) + " >= 1\n")
		i = i + 1

	sos = """#Use this Julia module to generate barrier certificates.
using MathOptInterface
const MOI = MathOptInterface
using JuMP
using SumOfSquares
using PolyJuMP
using Base.Test
using MultivariatePolynomials
using SemialgebraicSets
using Mosek

import DynamicPolynomials.@polyvar

@polyvar x[1:{}]
f = [{}]
{}
{}

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:{})
@variable m B Poly(Z)

f1 = -B{}
f2 = -dot(differentiate(B, x), f)
{}

@constraint m f1 >= 0
@constraint m f2 >= 0
{}

status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))""".format(nvar, sysdynamics, init, unsafe, initSOSPoly, unsafeSOSPoly, degree, init_cnstr, unsafe_vcs, unsafe_constraints)

	return sos

def genSOSContinuousAsDiscreteMultipleUnsafes(timestep, nvar, sysdynamics, init, unsafe, 
	initSOSPoly="@variable m Zinit SOSPoly(Z)", 
	unsafeSOSPoly="@variable m Zunsafe SOSPoly(Z)", 
	init_cnstr="- Zinit*init", 
	unsafe_cnstrs=["- Zunsafe*unsafe"],
	degree=4):

	transition = ""
	subs = ""
	for i in range(nvar):
		transition += ("x" + str(i+1) + " = x[" + str(i+1) +  "] + " + str(timestep) + "*f[" + str(i+1) + "]\n")
		if i == 0:
			subs += "x[" + str(i+1) + "]=>x" + str(i+1)
		else:
			subs += ", x[" + str(i+1) + "]=>x" + str(i+1)

	transitionconstraint = transition + "\nB1 = subs(B, " + subs + ")"

	unsafe_vcs = "" 
	unsafe_constraints = ""
	i = 3
	for unsafe_it in unsafe_cnstrs:
		unsafe_vcs += ("f" + str(i) + " = B" + unsafe_it + "\n")
		unsafe_constraints += ("@constraint m f" + str(i) + " >= 1\n")
		i = i + 1

	sos = """#Use this Julia module to generate barrier certificates.
using MathOptInterface
const MOI = MathOptInterface
using JuMP
using SumOfSquares
using PolyJuMP
using Base.Test
using MultivariatePolynomials
using SemialgebraicSets
using Mosek

import DynamicPolynomials.@polyvar

@polyvar x[1:{}]
f = [{}]
{}
{}

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:{})
@variable m B Poly(Z)

{}

f1 = -B{}
f2 = B - B1
{}

@constraint m f1 >= 0
@constraint m f2 >= 0
{}

status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))""".format(nvar, sysdynamics, init, unsafe, initSOSPoly, unsafeSOSPoly, degree, transitionconstraint, init_cnstr, unsafe_vcs, unsafe_constraints)

	return sos

def genSOSDiscreteMultipleUnsafes(nvar, sysdynamics, init, unsafe, 
	initSOSPoly="@variable m Zinit SOSPoly(Z)", 
	unsafeSOSPoly="@variable m Zunsafe SOSPoly(Z)", 
	init_cnstr="- Zinit*init", 
	unsafe_cnstrs=["- Zunsafe*unsafe"],
	degree=4):

	transition = ""
	subs = ""
	for i in range(nvar):
		transition += ("x" + str(i+1) + " = f[" + str(i+1) + "]\n")
		if i == 0:
			subs += "x[" + str(i+1) + "]=>x" + str(i+1)
		else:
			subs += ", x[" + str(i+1) + "]=>x" + str(i+1)

	transitionconstraint = transition + "\nB1 = subs(B, " + subs + ")"

	unsafe_vcs = "" 
	unsafe_constraints = ""
	i = 3
	for unsafe_it in unsafe_cnstrs:
		unsafe_vcs += ("f" + str(i) + " = B" + unsafe_it + "\n")
		unsafe_constraints += ("@constraint m f" + str(i) + " >= 1\n")
		i = i + 1

	sos = """#Use this Julia module to generate barrier certificates.
using MathOptInterface
const MOI = MathOptInterface
using JuMP
using SumOfSquares
using PolyJuMP
using Base.Test
using MultivariatePolynomials
using SemialgebraicSets
using Mosek

import DynamicPolynomials.@polyvar

@polyvar x[1:{}]
f = [{}]
{}
{}

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:1)
{}
Z = monomials(x, 0:{})
@variable m B Poly(Z)

{}

f1 = -B{}
f2 = B - B1
{}

@constraint m f1 >= 0
@constraint m f2 >= 0
{}

status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))""".format(nvar, sysdynamics, init, unsafe, initSOSPoly, unsafeSOSPoly, degree, transitionconstraint, init_cnstr, unsafe_vcs, unsafe_constraints)

	return sos	