#Use this Julia module to generate barrier certificates.
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

@polyvar x[1:4]
f = [0.0*x[1]+0.0*x[2]+1.0*x[3]+0.0*x[4],0.0*x[1]+0.0*x[2]+0.0*x[3]+1.0*x[4],12.531037868516844*x[1]+-1.41563815381789*x[2]+5.753378550776059*x[3]+-1.0942922496733771*x[4],-1.2392946479332074*x[1]+13.451870351144958*x[2]+-1.2783556372902831*x[3]+6.1639334497787095*x[4]]
init1 = (x[1] - -7)*(-6-x[1])
init2 = (x[2] - -7)*(-6-x[2])
init3 = (x[3] - 0)*(0-x[3])
init4 = (x[4] - 0)*(0-x[4])
init5 = (x[1] - (-7.932728599993485))*((-6.0)-x[1])
init6 = (x[2] - (-7.0))*((-5.059854225351314)-x[2])
init7 = (x[3] - (0.0))*((0.0)-x[3])
init8 = (x[4] - (0.0))*((0.0)-x[4])
unsafe1 = (x[1] - -1.0)*(-0.5-x[1])
unsafe2 = (x[2] - -1.5)*(-1.0-x[2])
unsafe1 = (x[1] - -3.1)*(-2.9-x[1])
unsafe2 = (x[2] - -4.1)*(-2.4-x[2])
unsafe1 = (x[1] - -1.7)*(-1.1-x[1])
unsafe2 = (x[2] - -4.1)*(-3.4-x[2])
unsafe1 = (x[1] - -1.0)*(-0.1-x[1])
unsafe2 = (x[2] - -0.4)*(1.0-x[2])

m = SOSModel(solver = MosekSolver())

Z = monomials(x, 0:1)
@variable m Zinit1 SOSPoly(Z)
@variable m Zinit2 SOSPoly(Z)
@variable m Zinit3 SOSPoly(Z)
@variable m Zinit4 SOSPoly(Z)
@variable m Zinit5 SOSPoly(Z)
@variable m Zinit6 SOSPoly(Z)
@variable m Zinit7 SOSPoly(Z)
@variable m Zinit8 SOSPoly(Z)
Z = monomials(x, 0:1)
@variable m Zunsafe1 SOSPoly(Z)
@variable m Zunsafe2 SOSPoly(Z)
@variable m Zunsafe1 SOSPoly(Z)
@variable m Zunsafe2 SOSPoly(Z)
@variable m Zunsafe1 SOSPoly(Z)
@variable m Zunsafe2 SOSPoly(Z)
@variable m Zunsafe1 SOSPoly(Z)
@variable m Zunsafe2 SOSPoly(Z)
Z = monomials(x, 0:4)
@variable m B Poly(Z)

x1 = x[1] + 0.01*f[1]
x2 = x[2] + 0.01*f[2]
x3 = x[3] + 0.01*f[3]
x4 = x[4] + 0.01*f[4]

B1 = subs(B, x[1]=>x1, x[2]=>x2, x[3]=>x3, x[4]=>x4)

f1 = -B - Zinit1*init1 - Zinit2*init2 - Zinit3*init3 - Zinit4*init4 - Zinit5*init5 - Zinit6*init6 - Zinit7*init7 - Zinit8*init8
f2 = B - B1
f3 = B - Zunsafe1*unsafe1 - Zunsafe2*unsafe2
f4 = B - Zunsafe1*unsafe1 - Zunsafe2*unsafe2
f5 = B - Zunsafe1*unsafe1 - Zunsafe2*unsafe2
f6 = B - Zunsafe1*unsafe1 - Zunsafe2*unsafe2


@constraint m f1 >= 0
@constraint m f2 >= 0
@constraint m f3 >= 1
@constraint m f4 >= 1
@constraint m f5 >= 1
@constraint m f6 >= 1


status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))