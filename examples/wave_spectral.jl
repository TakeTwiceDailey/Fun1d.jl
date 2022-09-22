
# using DifferentialEquations
# using BoundaryValueDiffEq
# using OrdinaryDiffEq
# using DataFrames
# using CSV
# using ApproxFun
# using RecursiveArrayTools
# using LinearAlgebra
# using Distributions
#
# using Plots

# VarContainer{T} = ArrayPartition{T, NTuple{numvar, Vector{T}}}
#
# struct Domain{S}
#     xmin::S
#     xmax::S
# end
#
# struct Grid{S}
#     domain::Domain{S}
#     ncells::Int
# end
#
# ################################################################################
#
# function setup(::Type{S}) where {S}
#     domain = Domain{S}(0, 1)
#     grid = Grid(domain, 20)
#     return grid
# end
#
# function init(::Type{T}, grid::Grid) where {T}
#     # println("WaveToy.init")
#     # ∂ₜₜu = ∂ₓₓu
#     # u(t,x) = f(t+x) + g(t-x)
#     # u(t,x) = sin(π t) cos(π x)
#     ϕ = project(T, grid, x -> 0)
#     ψ = project(T, grid, x -> π * cospi(x))
#     return Wave(ϕ, ψ)
# end
#
# function rhs(state::Wave, param, t)
#     # println("WaveToy.rhs t=$t")
#     ϕdot = state.ψ
#     ψdot = deriv2(state.ϕ)
#     staterhs = Wave(ϕdot, ψdot)
#     return staterhs::Wave
# end
#
# function main()
#     T = Float64
#     grid = setup(T)
#     state = init(T, grid)::Wave
#     tspan = T[0, 1]
#     atol = eps(T)^(T(3) / 4)
#     prob = ODEProblem(rhs, state, tspan)
#     sol = solve(prob, Tsit5(); abstol=atol)
#     return sol
# end

####
# The following solves heat equation with mixed boundary conditions.
# we first formulate the construction as
#
# B*u = r
# C*u_t = L*u
#
# where B represents the mixed boundary conditions, r is their values
# C is a conversion matrix and L is the Laplacian.
####

using ApproxFun, Plots, LinearAlgebra

S = Chebyshev()
D = Derivative() : S

# n = 10
#
# a = Fun(cos,S,n)
# b = Fun(cos,S,n)
#
# c=a*b; ncoefficients(c)





#
