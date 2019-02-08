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
f = [0.0*x[1]+0.0*x[2]+1.0*x[3]+0.0*x[4],0.0*x[1]+0.0*x[2]+0.0*x[3]+1.0*x[4],-4.384074022136936*x[1]+3.983575126493455*x[2]+-12.48875692661895*x[3]+3.1821734098768433*x[4],-4.567427959341512*x[1]+2.3377978177404106*x[2]+3.410368845548593*x[3]+-16.581705570698325*x[4]]
init1 = (x[1] - -7)*(-6-x[1])
init2 = (x[2] - -7)*(-8-x[2])
init3 = (x[3] - 0)*(0-x[3])
init4 = (x[4] - 0)*(0-x[4])
init5 = (x[1] - (-7.0))*((-5.33632058021777)-x[1])
init6 = (x[2] - (-7.0))*((-7.4424958729017)-x[2])
init7 = (x[3] - (0.0))*((0.0)-x[3])
init8 = (x[4] - (0.0))*((0.0)-x[4])
unsafe1 = (x[1] - -3.1)*(-2.9-x[1])
unsafe2 = (x[2] - -3.1)*(-2.9-x[2])

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


@constraint m f1 >= 0
@constraint m f2 >= 0
@constraint m f3 >= 1


status = solve(m)
print(STDERR,status)
print(STDERR,'#')
print(STDERR,getvalue(B))