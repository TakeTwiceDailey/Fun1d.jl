module GR_Spherical

using DifferentialEquations
using BoundaryValueDiffEq
using OrdinaryDiffEq
#using Fun1d
using DataFrames
using CSV
#using Plots
using Roots

#using BenchmarkTools
using InteractiveUtils
using RecursiveArrayTools
#using StaticArrays
using LinearAlgebra

using Distributions

#using Profile

#numvar = 13
numvar = 9

VarContainer{T} = ArrayPartition{T, NTuple{numvar, Vector{T}}}

struct Domain{S}
    xmin::S
    xmax::S
end

struct Grid{S}
    domain::Domain{S}
    ncells::Int
end

struct Param{T}
    rtmin::T
    rtmax::T
    drt::T
    Mtot::T
    grid::Grid{T}
    reg_list::Vector{Int64}
    r::Function
    drdrt::Function
    d2rdrt::Function
    rsamp::Vector{T}
    drdrtsamp::Vector{T}
    d2rdrtsamp::Vector{T}
    gauge::VarContainer{T}
    init_state::VarContainer{T}
    init_drstate::VarContainer{T}
    state::VarContainer{T}
    drstate::VarContainer{T}
    dtstate::VarContainer{T}
    dissipation::VarContainer{T}
    temp::VarContainer{T}
end

@inline function Base.similar(::Type{ArrayPartition},::Type{T},size::Int) where T
    return ArrayPartition([similar(Vector{T}(undef,size)) for i=1:numvar]...)::VarContainer{T}
end

function printlogo()
    # Just a fancy ASCII logo for the program
    println(
"\n",
"           /\\\\\\\\\\\\\\\\\\       /\\\\\\\\\\\\\\\\\\\\\\\\\n",
" ________/\\\\\\\\\\\\\\\\\\\\\\\\\\\\___/ /\\\\\\\\/ / /\\\\\\\\_____________________________\n",
"  ______//\\\\\\\\\\/ / / / /___\\/ /\\\\\\\\/_/ / /\\\\\\\\___________________________\n",
"   _____//\\\\\\\\/_/_/_/_/_____\\/ /\\\\\\\\ \\/ / /\\\\\\\\___________________________\n",
"    ____/ /\\\\\\\\____/\\\\\\\\\\\\___\\/ /\\\\\\\\\\\\\\\\\\\\\\\\\\\\_______in___________________\n",
"     ___\\/ /\\\\\\\\__/ /  /\\\\\\___\\/ /\\\\\\\\\\\\\\\\\\\\\\\\/________Spherical____________\n",
"      ___\\/ /\\\\\\\\_\\/__/ /\\\\\\___\\/ /\\\\\\\\ / /\\\\\\\\__________Symmetry____________\n",
"       ___\\/ /\\\\\\\\____\\/ /\\\\\\___\\/ /\\\\\\\\_/ //\\\\\\\\_____________________________\n",
"        ___\\/ /\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\___\\/ /\\\\\\\\\\// //\\\\\\\\____________________________\n",
"         ___\\/ / /\\\\\\\\\\\\\\\\\\  /____\\/ /\\\\\\\\_\\// //\\\\\\\\_____by Conner Dailey______\n",
"          ___\\/_/ / / / / /_/______\\/ /  /___\\// /  /____________________________\n",
"           _____\\/_/_/_/_/__________\\/__/______\\/__/______________________________\n"
)

end

spacing(grid::Grid) = (grid.domain.xmax - grid.domain.xmin) / (grid.ncells + 1)

function sample!(f::Vector{T}, grid::Grid{S}, fun) where {S,T}

    drt = spacing(grid)
    rtmin = grid.domain.xmin

    f .= T[fun(rtmin + drt*(n-1)) for n in 1:(grid.ncells + 2)]

end


#Kerr-Schild Coordinates
# Derivatives still need to be done properly

fαt(M,r,rt) = 1/(r(rt)^2+2*M(rt)*r(rt))
f∂rαt(M,r,rt) = -(2/r(rt)^2)*(M(rt)+r(rt))/(r(rt)+2*M(rt))^2
f∂r2αt(M,r,rt) = 1/(M(rt)*r(rt)^3) - 1/(M(rt)*(2*M(rt)+r(rt))^3)

fβr(M,r,rt) = 2*M(rt)/(2*M(rt)+r(rt))
f∂rβr(M,r,rt) = -2*M(rt)/((r(rt)+2*M(rt))^2)
f∂r2βr(M,r,rt) = 4*M(rt)/((r(rt)+2*M(rt))^3)

fγrr(M,r,rt) = 1 + 2*M(rt)/r(rt)
f∂rγrr(M,r,rt) = -2*M(rt)/(r(rt)^2)

fγθθ(M,r,rt) = r(rt)^2
f∂rγθθ(M,r,rt) = 2*r(rt)

fKrr(M,∂M,r,rt) = (2*(r(rt)*∂M(rt)-M(rt))/r(rt)^3)*(r(rt)+M(rt))/sqrt((1+2*M(rt)/r(rt)))
f∂rKrr(M,r,rt) = (2*M(rt)/r(rt)^3)*(5*M(rt)^2+6*M(rt)*r(rt)+2*r(rt)^2)*sqrt((1+2*M(rt)/r(rt)))/(r(rt)+2*M(rt))^2

fKθθ(M,r,rt) = 2*M(rt)/sqrt((1+2*M(rt)/r(rt)))
f∂rKθθ(M,r,rt) = (2*M(rt)^2/r(rt)^2)/sqrt((1+2*M(rt)/r(rt)))^3

ffrrr(M,∂M,r,rt) = (7*M(rt) + (4 + ∂M(rt))*r(rt))/(r(rt)^2)
f∂rfrrr(M,r,rt) = -2*(7*M(rt) + 2*r(rt))/(r(rt)^3)

ffrθθ(M,r,rt) = r(rt)
f∂rfrθθ(M,r,rt) = 1.

#Orthogonal Spherical Minkowski

# fα(M,r,rt) = 1
# f∂rα(M,r,rt) = 0
# f∂r2α(M,r,rt) = 0
#
# fβr(M,r,rt) = 0
# f∂rβr(M,r,rt) = 0
# f∂r2βr(M,r,rt) = 0
#
# fγrr(M,r,rt) = 1
# f∂rγrr(M,r,rt) = 0
# f∂r2γrr(M,r,rt) = 0
#
# fγθθ(M,r,rt) = r(rt)^2
# f∂rγθθ(M,r,rt) = 2*r(rt)
# f∂r2γθθ(M,r,rt) = 2.
#
# fKrr(M,r,rt) = 0
# f∂rKrr(M,r,rt) = 0
#
# fKθθ(M,r,rt) = 0
# f∂rKθθ(M,r,rt) = 0
#
# ffrrr(M,r,rt) = 4/r(rt)
# f∂rfrrr(M,r,rt) = -4/r(rt)^2
#
# ffrθθ(M,r,rt) = r(rt)
# f∂rfrθθ(M,r,rt) = 1.


function init!(state::VarContainer{T}, param) where T

    ############################################
    # Specifies the Initial Conditions
    ############################################




    init_state = param.init_state
    init_drstate = param.init_drstate
    gauge = param.gauge

    γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,ψ,Π = state.x
    αt,βr,∂rαt,∂rβr,∂r2αt,∂r2βr,∂r3βr,∂r4βr,∂r5βr = gauge.x
    γrri,γθθi,Krri,Kθθi,frrri,frθθi,𝜙i,ψi,Πi = init_state.x
    ∂rγrr,∂rγθθ,∂rKrr,∂rKθθ,∂rfrrr,∂rfrθθ,∂r𝜙,∂rψ,∂rΠ = init_drstate.x

    grid = param.grid
    drt = spacing(grid)
    r = param.r
    drdrt = param.drdrt
    d2rdrt = param.d2rdrt
    rtmin = param.rtmin
    rtmax = param.rtmax
    reg_list = param.reg_list

    n = grid.ncells + 2
    m = 1.
    rtspan = (rtmin,rtmax)

    # Mass (no real reason not to use 1 here)
    #M = 1

    r0 = 8.
    σr = 0.4
    Amp = 0.1
    #min = 5
    # f𝜙(rt) = 0.
    # fψ(rt) = 0.
    # fΠ(rt) = 0.
    # f𝜙(rt) = Amp*(1/r(rt))*exp(-(1/2)*((r(rt)-r0)/σr)^2)
    # fψ(rt) = Amp*exp(-(1/2)*((r(rt)-r0)/σr)^2)*(r(rt)*r0-r(rt)^2-σr^2)/(r(rt)^2*σr^2)
    # fΠ(rt) = fβr(M,r,rt)*fψ(rt)

    f𝜙(rt) = Amp*(1/r(rt))*exp(-(1/2)*((r(rt)-r0)/σr)^2)
    fψ(rt) = Amp*exp(-(1/2)*((r(rt)-r0)/σr)^2)*(r(rt)*r0-r(rt)^2-σr^2)/(r(rt)^2*σr^2)
    fΠ(rt) = fβr(M,r,rt)*fψ(rt)

    # fρ(M,rt) = 0*(2*fK𝜙(rt)^2 + (1/2)*(fχ(M,r,rt)/fγtrr(M,r,rt))*f∂𝜙(rt)^2
    #     + (1/2)*m^2*f𝜙(rt)^2)

    # fρ(M,rt) = 0
    #
    # function f∂rtM(M,rt)
    #      if r(rt) < 2*M
    #          return 0.
    #      else
    #          4*pi*(r(rt)^2)*fρ(M,rt)*drdrt(rt)
    #      end
    # end
    #
    # function f∂M(M,rt)
    #      if r(rt) < 2*M
    #          return 0.
    #      else
    #          4*pi*(r(rt)^2)*fρ(M,rt)
    #      end
    # end
    #
    # # Constraint Equations
    #
    # function constraintSystem(M, param, rt)
    #     f∂rtM(M,rt)*0
    # end
    #
    # atol = 1e-15
    #
    # BVP = ODEProblem(constraintSystem, 1., rtspan, param)
    # M = solve(BVP, Tsit5(), abstol=atol, dt=drt, adaptive=false)

    #∂M(rt) = f∂M(M(rt),rt)
    M(rt) = 1.0
    ∂M(rt) = 0.0

    sample!(αt, grid, rt -> fαt(M,r,rt) )
    sample!(βr, grid, rt -> fβr(M,r,rt) )
    sample!(γrri, grid, rt -> fγrr(M,r,rt) )
    sample!(γθθi, grid, rt -> fγθθ(M,r,rt) )
    sample!(Krri, grid, rt -> fKrr(M,∂M,r,rt) )
    sample!(Kθθi, grid, rt -> fKθθ(M,r,rt) )
    sample!(frrri, grid, rt -> ffrrr(M,∂M,r,rt) )
    sample!(frθθi, grid, rt -> ffrθθ(M,r,rt) )
    sample!(𝜙i, grid, f𝜙)
    sample!(ψi, grid, fψ)
    sample!(Πi, grid, fΠ)

    sample!(∂rαt, grid, rt -> f∂rαt(M,r,rt) )
    sample!(∂rβr, grid, rt -> f∂rβr(M,r,rt) )
    sample!(∂rγrr, grid, rt -> f∂rγrr(M,r,rt) )
    sample!(∂rγθθ, grid, rt -> f∂rγθθ(M,r,rt) )
    sample!(∂rKrr, grid, rt -> f∂rKrr(M,r,rt) )
    sample!(∂rKθθ, grid, rt -> f∂rKθθ(M,r,rt) )
    sample!(∂rfrrr, grid, rt -> f∂rfrrr(M,r,rt) )
    sample!(∂rfrθθ, grid, rt -> f∂rfrθθ(M,r,rt) )

    sample!(∂r2αt, grid, rt -> f∂r2αt(M,r,rt) )
    sample!(∂r2βr, grid, rt -> f∂r2βr(M,r,rt) )
    #sample!(∂r2𝜙, grid, f𝜙)

    s = 0*10^(-10)

    for i in 1:numvar
        if i in reg_list
            for j in 1:n
               state.x[i][j] = 1. + s*rand(Uniform(-1,1))
            end
            state.x[i][1] = 1.
            state.x[i][n] = 1.
        else
            for j in 1:n
               state.x[i][j] = init_state.x[i][j] + s*rand(Uniform(-1,1))
            end
            state.x[i][1] = init_state.x[i][1]
            state.x[i][n] = init_state.x[i][n]
        end
    end

end

@inline function deriv!(df::Vector{T}, f::Vector{T}, n::Int64, dx::T) where T

    #######################################################
    # Calculates derivatives using a 4th order SBP operator
    #######################################################

    # @inbounds @fastmath @simd

    df[1] = (-48*f[1] + 59*f[2] - 8*f[3] - 3*f[4])/(34*dx)

    df[2] = (-f[1] + f[3])/(2*dx)

    df[3] = (8*f[1] - 59*f[2] + 59*f[4] - 8*f[5])/(86*dx)

    df[4] = (3*f[1] - 59*f[3] + 64*f[5] - 8*f[6])/(98*dx)

    for i in 5:(n - 4)
        df[i] = (f[i-2] - 8*f[i-1] + 8*f[i+1] - f[i+2])/(12*dx)
    end

    df[n-3] = -(3*f[n] - 59*f[n-2] + 64*f[n-4] - 8*f[n-5])/(98*dx)

    df[n-2] = -(8*f[n] - 59*f[n-1] + 59*f[n-3] - 8*f[n-4])/(86*dx)

    df[n-1] = -(-f[n] + f[n-2])/(2*dx)

    df[n] = -(-48*f[n] + 59*f[n-1] - 8*f[n-2] - 3*f[n-3])/(34*dx)

end

@inline function dissipation!(df::Vector{T}, f::Vector{T}, n::Int64) where T

    ############################################
    # Calculates the numerical dissipation terms
    ############################################

    # @simd @inbounds @fastmath

    df[1] = (-48*f[1] + 96*f[2] - 48*f[3])/(17)

    df[2] = (96*f[1] - 240*f[2] + 192*f[3] - 48*f[4])/(59)

    df[3] = (-48*f[1] + 192*f[2] - 288*f[3] + 192*f[4] - 48*f[5])/(43)

    df[4] = (-48f[2] + 192*f[3] - 288*f[4] + 192*f[5] - 48*f[6])/(49)

    for i in 5:(n - 4)
        df[i] = (-f[i-2] + 4*f[i-1] - 6*f[i] + 4*f[i+1] - f[i+2])
    end

    df[n-3] = (-48f[n-1] + 192*f[n-2] - 288*f[n-3] + 192*f[n-4] - 48*f[n-5])/(49)

    df[n-2] = (-48*f[n] + 192*f[n-1] - 288*f[n-2] + 192*f[n-3] - 48*f[n-4])/(43)

    df[n-1] = (96*f[n] - 240*f[n-1] + 192*f[n-2] - 48*f[n-3])/(59)

    df[n] = (-48*f[n] + 96*f[n-1] - 48*f[n-2])/(17)

    # df[1] = 0.
    # df[n] = 0.

end

function rhs!(dtstate::VarContainer{T},regstate::VarContainer{T}, param::Param{T}, t) where T

    ############################################
    # Caculates the right hand ride of the
    # evolved variables
    #
    # This is the main meat of the program.
    # This function contains all of the boundary
    # conditions, coordinate conversions,
    # spatial derivative calculations,
    # evolution equations, and numerical
    # dissipation. Each time the Julia DiffEq
    # Solver moves one time step, it calls
    # this function to calculate the new
    # values of the evolved variables.
    ############################################

    # Unpack the parameters

    m = 0.
    Mtot = 1.
    #M = 1.

    grid = param.grid
    drt = param.drt
    r = param.rsamp
    drdrt = param.drdrtsamp
    d2rdrt = param.d2rdrtsamp
    rtmin = param.rtmin
    rtmax = param.rtmax
    reg_list = param.reg_list

    fr = param.r

    state = param.state
    drstate = param.drstate
    dissipation = param.dissipation
    dtstate2 = param.dtstate
    temp = param.temp

    init_state = param.init_state
    init_drstate = param.init_drstate
    gauge = param.gauge

    n = grid.ncells + 2

    # Copy the state into the parameters
    # so that it can be changed

    #######################
    # Attention!
    #
    # Do not do the following:
    # state .= regstate
    #
    # This results in an intense slowdown
    # Do instead:
    for i in 1:numvar
        state.x[i] .= regstate.x[i]
    end

    # Give names to individual variables

    γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,ψ,Π = state.x
    ∂rγrr,∂rγθθ,∂rKrr,∂rKθθ,∂rfrrr,∂rfrθθ,∂r𝜙,∂rψ,∂rΠ = drstate.x
    ∂tγrr,∂tγθθ,∂tKrr,∂tKθθ,∂tfrrr,∂tfrθθ,∂t𝜙,∂tψ,∂tΠ = dtstate.x
    ∂4γrr,∂4γθθ,∂4Krr,∂4Kθθ,∂4frrr,∂4frθθ,∂4𝜙,∂4ψ,∂4Π = dissipation.x
    ᾶ,βr,∂rᾶ,∂rβr,∂r2ᾶ,∂r2βr,α,∂r4βr,∂r5βr = gauge.x

    γrri,γθθi,Krri,Kθθi,frrri,frθθi,𝜙i,ψi,Πi = init_state.x
    ∂rγrri,∂rγθθi,∂rKrri,∂rKθθi,∂rfrrri,∂rfrθθi,∂r𝜙i,∂rψi,∂rΠi = init_drstate.x

    # Dirichlet boundary conditions on scalar field

    𝜙[1] = 0.
    Π[1] = 0.

    # Calculate first spatial derivatives

    for i in 1:numvar
        deriv!(drstate.x[i],state.x[i],n,drt)
    end

    # Calculate numerical dissipation

    for i in 1:numvar
        dissipation!(dissipation.x[i],state.x[i],n)
    end

    # Convert between the computational rt coordinate
    # and the traditional r coordinate

    for i in 1:numvar
        @. drstate.x[i] /= drdrt
    end

    # Convert between regularized variables and cannonical variables

    reg = temp.x[1]; ∂reg = temp.x[2];

    for i in reg_list
        @. reg = state.x[i]; @. ∂reg = drstate.x[i];
        @. state.x[i] *= init_state.x[i]
        @. drstate.x[i] = ∂reg*init_state.x[i] + reg*init_drstate.x[i]
    end

    # ∂rγrr[n] = 2*frrr[n] - 8*γrr[n]*frθθ[n]/γθθ[n]
    # ∂rγθθ[n] = 2*frθθ[n]
    #
    # ∂rKθθ[n] = frθθ[n]*Kθθ[n]/γθθ[n] + frθθ[n]*Krr[n]/γrr[n]
    #
    # ∂rfrθθ[n] = (frrr[n]*frθθ[n]/γrr[n] + γrr[n]*Kθθ[n]^2/(2*γθθ[n])
    #  + γrr[n]/2 + Krr[n]*Kθθ[n] - 7*frθθ[n]^2/(2*γθθ[n]))

    ∂rlnᾶ = temp.x[5]; ∂r2lnᾶ = temp.x[6];

    @. α = ᾶ*γθθ*sqrt(γrr)
    @. ∂rlnᾶ = ∂rᾶ/ᾶ
    @. ∂r2lnᾶ = (∂r2ᾶ*ᾶ - ∂rᾶ^2)/ᾶ^2

    #########################################################
    # Evolution Equations
    #
    # This is the full suite of evolution equations
    # for GR in spherical symmetry in the
    # 'Einstein-Christoffel' framework.
    #
    #########################################################

    @. ∂tγrr = ∂rγrr*βr + 2*∂rβr*γrr - 2*α*Krr

    @. ∂tγθθ = ∂rγθθ*βr - 2*α*Kθθ

    @. ∂tKrr = (∂rKrr*βr + 2*∂rβr*Krr + 2*α*frrr^2/γrr^2 - α*Krr^2/γrr
     - 6*α*frθθ^2/γθθ^2 + 2*α*Krr*Kθθ/γθθ - 8*α*frrr*frθθ/(γrr*γθθ)
     - α*∂rfrrr/γrr - α*frrr*∂rlnᾶ/γrr - α*∂rlnᾶ^2 - α*∂r2lnᾶ)

    @. ∂tKθθ = (∂rKθθ*βr + α + α*Krr*Kθθ/γrr - 2*α*frθθ^2/(γrr*γθθ)
     - α*∂rfrθθ/γrr - α*frθθ*∂rlnᾶ/γrr)

    @. ∂tfrrr = (∂rfrrr*βr + 3*∂rβr*frrr - α*∂rKrr - α*frrr*Krr/γrr
     + 12*α*frθθ*Kθθ*γrr/γθθ^2 - 10*α*frθθ*Krr/γθθ - 4*α*frrr*Kθθ/γθθ
     - α*Krr*∂rlnᾶ - 4*α*Kθθ*γrr*∂rlnᾶ/γθθ + γrr*∂r2βr)

    @. ∂tfrθθ = (∂rfrθθ*βr + ∂rβr*frθθ - α*∂rKθθ - α*frrr*Kθθ/γrr
     + 2*α*frθθ*Kθθ/γθθ - α*Kθθ*∂rlnᾶ)

    #########################################################
    # Source Terms and Source Evolution
    #
    # This currently includes the addition of source terms
    # to GR that come from a Klein-Gordon scalar field
    #
    #########################################################

    # Klein-Gordon System

    Γt = temp.x[7]; Γr = temp.x[8];

    @. Γt = (βr*∂rlnᾶ - ∂rβr)/α^2
    @. Γr = 2*βr*∂rβr/α^2 - (1/γrr + (βr/α)^2)*∂rlnᾶ - 4*frθθ/(γrr*γθθ)

    @. ∂t𝜙 = Π
    @. ∂tψ = ∂rΠ
    @. ∂tΠ = (α^2)*((1/γrr-(βr/α)^2)*∂rψ + 2*(βr/α^2)*∂rΠ - Γr*ψ - Γt*Π - m^2*𝜙)

    # @. ∂t𝜙 = βr*ψ - α*Π
    # @. ∂tψ = βr*∂rψ - α*∂rΠ - α*(frrr/γrr - 2*frθθ/γθθ + ∂rlnαt)*Π + ψ*∂rβr
    # @. ∂tΠ = (βr*∂rΠ - α*∂rψ/γrr + α*(Krr/γrr + 2*Kθθ/γθθ)*Π
    #  - α*(4*frθθ/γθθ + ∂rlnαt)*ψ/γrr)#- m^2*𝜙)

    Sr = temp.x[5]; Tt = temp.x[6]; Srr = temp.x[7]; Sθθ = temp.x[8];

    @. Sr = -ψ*(Π - βr*ψ)/α
    @. Tt = (Π - βr*ψ)^2/α^2 - ψ^2/γrr - 2*(m^2)*𝜙^2
    @. Srr = γrr*( (Π - βr*ψ)^2/α^2 + ψ^2/γrr - (m^2)*𝜙^2)/2
    @. Sθθ = γθθ*( (Π - βr*ψ)^2/α^2 - ψ^2/γrr - (m^2)*𝜙^2)/2

    # @. ∂tKrr += 4*pi*α*(γrr*Tt - 2*Srr)
    # @. ∂tKθθ += 4*pi*α*(γθθ*Tt - 2*Sθθ)
    # @. ∂tfrrr += 16*pi*α*γrr*Sr

    # Specify the inner and outer temporal boundary conditions
    # for metric variables

    # for i in 1:numvar
    #     dtstate.x[i][n] = 0.
    # end
    αi = ᾶ[1]*γθθi[1]*sqrt(γrri[1])
    c = -βr[1] - α[1]/sqrt(γrr[1])
    ci = -βr[1] - αi/sqrt(γrri[1])
    ∂tγrri = βr[1]*∂rγrri[1]
    ∂tγθθi = βr[1]*∂rγθθi[1]
    ∂tKrri = -ci*(∂rKrri[1] - ∂rfrrri[1]/sqrt(γrri[1]))/2
    ∂tKθθi = -ci*(∂rKθθi[1] - ∂rfrθθi[1]/sqrt(γrri[1]))/2
    ∂tγrr[1] = βr[1]*∂rγrr[1] - ∂tγrri
    ∂tγθθ[1] = βr[1]*∂rγθθ[1] - ∂tγθθi
    ∂tKrr[1] = -c*(∂rKrr[1] - ∂rfrrr[1]/sqrt(γrr[1]))/2 - ∂tKrri
    ∂tKθθ[1] = -c*(∂rKθθ[1] - ∂rfrθθ[1]/sqrt(γrr[1]))/2 - ∂tKθθi
    ∂tfrrr[1] = -sqrt(γrr[1])*∂tKrr[1]
    ∂tfrθθ[1] = -sqrt(γrr[1])*∂tKθθ[1]
    ∂tΠ[1] = 0.
    ∂t𝜙[1] = 0.
    #∂tψ[1] = c*sqrt(γrr[1])*(∂rΠ[1] - ∂rψ[1]/sqrt(γrr[1]))/2
    # ∂tΠ[1] = -c*(∂rΠ[1] - ∂rψ[1]/sqrt(γrr[1]))/2
    # ∂tψ[1] = -sqrt(γrr[1])*∂tΠ[1]

    αi = ᾶ[n]*γθθi[n]*sqrt(γrri[n])
    c = -βr[n] + α[n]/sqrt(γrr[n])
    ci = -βr[n] + αi/sqrt(γrri[n])
    ∂tKrri = -ci*(∂rKrri[n] + ∂rfrrri[n]/sqrt(γrri[n]))/2
    ∂tKθθi = -ci*(∂rKθθi[n] + ∂rfrθθi[n]/sqrt(γrri[n]))/2
    ∂tγrr[n] = 0.
    ∂tγθθ[n] = 0.
    ∂tKrr[n] = -c*(∂rKrr[n] + ∂rfrrr[n]/sqrt(γrr[n]))/2 - ∂tKrri
    ∂tKθθ[n] = -c*(∂rKθθ[n] + ∂rfrθθ[n]/sqrt(γrr[n]))/2 - ∂tKθθi
    ∂tfrrr[n] = sqrt(γrr[n])*∂tKrr[n]
    ∂tfrθθ[n] = sqrt(γrr[n])*∂tKθθ[n]
    ∂tΠ[n] = 0.
    ∂t𝜙[n] = 0.
    #∂tψ[n] = -c*sqrt(γrr[n])*(∂rΠ[n] + ∂rψ[n]/sqrt(γrr[n]))/2
    # ∂tΠ[n] = -c*(∂rΠ[n] + ∂rψ[n]/sqrt(γrr[n]))/2
    # ∂tψ[n] = sqrt(γrr[n])*∂tΠ[n]

    # Convert back to regularized variables

    for i in reg_list
        @. dtstate.x[i] /= init_state.x[i]
    end

    # Add the numerical dissipation to the regularized state

    σ = 0.
    for i in 1:numvar
        @. dtstate.x[i] += σ*dissipation.x[i]/16.
    end

    for i in 1:numvar-3
        @. dtstate.x[i] = 0.
    end

    # Store the calculated state into the param
    # so that we can print it to the screen

    for i in 1:numvar
        dtstate2.x[i] .= dtstate.x[i]
    end

end

function rhs_all(regstate::VarContainer{T}, param::Param{T}, t) where T

    # Runs the right-hand-side routine, but with allocation so that
    # the state can be saved at the end.

    n = param.grid.ncells + 2

    dtstate = similar(ArrayPartition,T,n)

    rhs!(dtstate,regstate,param,t)

    return dtstate

end

function constraints(regstate::VarContainer{T},param) where T

    ############################################
    # Caculates the constraints of the system
    #
    # These are outputed at every saved time step.
    # They should limit to zero as the spatial
    # resolution is increased. They are important to
    # monitor to make sure the physics of the system
    # is being properly modeled.
    ############################################

    # Unpack Variables

    state = param.state
    drstate = param.drstate
    gauge = param.gauge

    for i in 1:numvar
        state.x[i] .= regstate.x[i]
    end

    γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,ψ,Π = state.x
    ∂rγrr,∂rγθθ,∂rKrr,∂rKθθ,∂rfrrr,∂rfrθθ,∂r𝜙,∂rψ,∂rΠ = drstate.x
    αt,βr,∂rαt,∂rβr,∂r2αt,∂r2βr,∂r3βr,∂r4βr,∂r5βr = gauge.x

    init_state = param.init_state
    init_drstate = param.init_drstate

    m = 0.
    M = 1.
    n = param.grid.ncells + 2
    drt = param.drt
    r = param.rsamp
    drdrt = param.drdrtsamp
    d2rdrt = param.d2rdrtsamp
    temp = param.temp
    grid = param.grid
    reg_list = param.reg_list

    deriv!(∂rKθθ,Kθθ,n,drt)
    deriv!(∂rfrθθ,frθθ,n,drt)
    deriv!(∂r𝜙,𝜙,n,drt)

    ∂rKθθ ./= drdrt
    ∂rfrθθ ./= drdrt
    ∂r𝜙 ./= drdrt

    reg = temp.x[1]; ∂reg = temp.x[2]; ∂2reg = temp.x[3];

    for i in reg_list
        @. reg = state.x[i]; @. ∂reg = drstate.x[i];
        @. state.x[i] *= init_state.x[i]
        @. drstate.x[i] = ∂reg*init_state.x[i] + reg*init_drstate.x[i]
    end

    α = temp.x[4]; ρ = temp.x[5]; Sr = temp.x[6]

    @. α = αt*γθθ*sqrt(γrr)
    # @. ρ = (Π^2 + ψ^2/γrr + (m^2)*𝜙^2)/2
    # #Lower Index
    # @. Sr = ψ*Π
    @. ρ = ( (Π - βr*ψ)^2/α^2 + ψ^2/γrr + (m^2)*𝜙^2)/2
    #Lower Index
    @. Sr = -ψ*(Π - βr*ψ)/α

    γ = temp.x[6]; Er = temp.x[7];

    norm = ones(T,n)
    norm[1] = 17/48; norm[2] = 59/48; norm[3] = 43/48; norm[4] = 49/48;
    norm[n] = 17/48; norm[n-1] = 59/48; norm[n-2] = 43/48; norm[n-3] = 49/48;

    @. γ = γrr*γθθ^2
    @. Er = norm*sqrt(γ)*(α*ρ - βr*Sr)*drdrt

    E = 0
    for i in 1:n
        E += drt*Er[i]
    end

    # Constraint Equations

    𝓗 = zeros(T,n); 𝓜r = zeros(T,n);

    @. 𝓗 = (∂rfrθθ/(γθθ*γrr) + 7*frθθ^2/(2*γrr*γθθ^2) - frrr*frθθ/(γrr^2*γθθ)
     - Kθθ^2/(2*γθθ^2) - 1/(2*γθθ) - Krr*Kθθ/(γrr*γθθ) + 4*pi*ρ)

    @. 𝓜r = ∂rKθθ/γθθ - frθθ*Kθθ/γθθ^2 - frθθ*Krr/(γθθ*γrr) + 4*pi*Sr

    return [𝓗, 𝓜r, E]

end

function custom_progress_message(dt,state::VarContainer{T},param,t) where T

    ###############################################
    # Outputs status numbers while the program runs
    ###############################################

    dtstate = param.dtstate::VarContainer{T}

    ∂tγrr,∂tγθθ,∂tKrr,∂tKθθ,∂tfrrr,∂tfrθθ,∂t𝜙,∂tψ,∂tΠ = dtstate.x

    println("  ",
    rpad(string(round(t,digits=1)),10," "),
    rpad(string(round(maximum(abs.(∂tγrr)), digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂tγθθ)), digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂tKrr)), digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂tKθθ)), digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂tfrrr)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂tfrθθ)),digits=3)),12," ")
    )

    return

end


function solution_saver(T,grid,sol,param,folder)

    ###############################################
    # Saves all of the variables in nice CSV files
    # in the choosen data folder directory
    ###############################################

    old_files = readdir(string("data/",folder); join=true)
    for i in 1:length(old_files)
        rm(old_files[i])
    end

    vars = (["γrr","γθθ","Krr","Kθθ","frrr","frθθ","𝜙","ψ","Π",
    "∂tγrr","∂tγθθ","∂tKrr","∂tKθθ","∂tfrrr","∂tfrθθ","∂t𝜙","∂tψ",
    "∂tΠ","H","Mr","E","appHorizon"])
    varlen = length(vars)
    #mkdir(string("data\\",folder))
    tlen = size(sol)[2]
    rlen = grid.ncells + 2
    r = param.rsamp
    rtmin = param.rtmin
    reg_list = param.reg_list

    init_state = param.init_state
    init_drstate = param.init_drstate

    dtstate = [rhs_all(sol[i],param,0.) for i = 1:tlen]

    cons = [constraints(sol[i],param) for i = 1:tlen]

    array = Array{T,2}(undef,tlen+1,rlen+1)

    array[1,1] = 0
    array[1,2:end] .= r

    for j = 1:numvar

        if j in reg_list
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                @. array[i,2:end] = sol[i-1].x[j]*init_state.x[j]
            end
        else
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                @. array[i,2:end] = sol[i-1].x[j]
            end
        end


        CSV.write(
            string("data/",folder,"/",vars[j],".csv"),
            DataFrame(array, :auto),
            header=false
        )

    end

    for j = 1:numvar

        if j in reg_list
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                @. array[i,2:end] = dtstate[i-1].x[j]*init_state.x[j]
            end
        else
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                @. array[i,2:end] = dtstate[i-1].x[j]
            end
        end

        CSV.write(
            string("data/",folder,"/",vars[j+numvar],".csv"),
            DataFrame(array, :auto),
            header=false
        )

    end

    for j = 1:2

        for i = 2:tlen+1
            array[i,1] = sol.t[i-1]
            @. array[i,2:end] = cons[i-1][j]
        end

        CSV.write(
            string("data/",folder,"/",vars[j+2*numvar],".csv"),
            DataFrame(array, :auto),
            header=false
        )

    end

    for j = 3:3

        for i = 2:tlen+1
            array[i,1] = sol.t[i-1]
            array[i,2] = cons[i-1][j]
            @. array[i,3:end] = 0.
        end

        CSV.write(
            string("data/",folder,"/",vars[j+2*numvar],".csv"),
            DataFrame(array, :auto),
            header=false
        )

    end

    # for i = 2:tlen+1
    #     array[i,1] = sol.t[i-1]
    #     array[i,2] = cons[i-1][4]
    #     array[i,3:end] .= 0
    # end
    #
    # CSV.write(
    # string("data/",folder,"/","E-",rtmin,".csv"),
    # DataFrame(array, :auto),
    # header=false
    # )

end

function main(points,folder)

    ###############################################
    # Main Program
    #
    # Calls each of the above functions to run a
    # simulation. Sets up the numerical grid,
    # sets the gauge conditions, sets up a new
    # computational rt coordinate that makes the
    # physical r coordinate step larger near the
    # outer boundary, sets the initial conditions,
    # and finally runs the numerical DiffEq
    # package to run the time integration.
    #
    # All data is saved in the folder specified to
    # the solution_saver, each in their own CSV
    # file.
    ###############################################

    for i = 9:9


        T = Float64

        #rtspan = T[2.,22.] .+ (1.0 - 0.1*i)
        rtspan = T[3.0,13.0]
        rtmin, rtmax = rtspan
        rspan = T[rtmin,rtmax]

        #rspan = T[rtmin,rtmax*10.]
        # f(x) = x*tan((rtmax-rtmin)/x) + rtmin - rspan[2]
        #
        # rs = find_zero(f, 0.64*rtmax)
        #
        # r(rt) = rs*tan((rt-rtmin)/rs) + rtmin
        # drdrt(rt) = sec((rt-rtmin)/rs)^2
        # d2rdrt(rt) = (2/rs)*(sec((rt-rtmin)/rs)^2)*tan((rt-rtmin)/rs)

        r(rt) = rt
        drdrt(rt) = 1.
        d2rdrt(rt) = 0.

        println("Mirror: ",rtmin)

        domain = Domain{T}(rtmin, rtmax)
        grid = Grid(domain, points)

        n = grid.ncells + 2

        drt = spacing(grid)
        dt = drt/4.

        tspan = T[0., 31.]
        tmin, tmax = tspan

        printtimes = 0.5

        v = 1.

        m = 0.

        Mtot = 1.

        # γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,, = state.x
        #reg_list = [4,7,8]
        reg_list = [1,2,3,4,5,6]
        #reg_list = [7,8,9,10]
        #reg_list = [10]

        atol = eps(T)^(T(3) / 4)

        alg = RK4()

        #printlogo()

        custom_progress_step = round(Int, printtimes/dt)
        step_iterator = custom_progress_step

        regstate = similar(ArrayPartition,T,n)

        state = similar(ArrayPartition,T,n)
        drstate = similar(ArrayPartition,T,n)

        init_state = similar(ArrayPartition,T,n)
        init_drstate = similar(ArrayPartition,T,n)

        gauge = similar(ArrayPartition,T,n)
        dtstate = similar(ArrayPartition,T,n)
        dissipation = similar(ArrayPartition,T,n)
        temp = similar(ArrayPartition,T,n)

        #println("Defining Problem...")
        rsamp = similar(Vector{T}(undef,n))
        drdrtsamp = similar(Vector{T}(undef,n))
        d2rdrtsamp = similar(Vector{T}(undef,n))

        sample!(rsamp, grid, rt -> r(rt) )
        sample!(drdrtsamp, grid, rt -> drdrt(rt) )
        sample!(d2rdrtsamp, grid, rt -> d2rdrt(rt) )

        param = Param(
        rtmin,rtmax,drt,Mtot,grid,reg_list,
        r,drdrt,d2rdrt,
        rsamp,drdrtsamp,d2rdrtsamp,gauge,
        init_state,init_drstate,
        state,drstate,
        dtstate,dissipation,temp)

        init!(regstate, param)

        # M=1.
        # r=rsamp
        # γrr = init_state.x[1]; γθθ = init_state.x[2]; frθθ = init_state.x[6]
        # αt = gauge.x[1]; βr = gauge.x[2]; ∂rαt = gauge.x[3]; ∂rβr = gauge.x[4];
        #
        # α = temp.x[4]; ∂rlnαt = temp.x[5];
        #
        # @. α = αt*γθθ*sqrt(γrr)
        # @. ∂rlnαt = ∂rαt/αt
        #
        # Γt = temp.x[7]; Γr = temp.x[8];
        #
        # @. Γt = (βr*∂rlnαt - ∂rβr)/α^2 #- (-2*M/r^2)
        # @. Γr = 2*βr*∂rβr/α^2 - (1/γrr + (βr/α)^2)*∂rlnαt - 4*frθθ/(γrr*γθθ) #- (2*(M-r)/r^2)
        #
        # @. Γt = (-2*M/r^2)
        # @. Γr = (2*(M-r)/r^2)
        #
        # println(Γt[end-10:end])
        # println(Γr[end-10:end])
        #
        # return

        prob = ODEProblem(rhs!, regstate, tspan, param)

        #println("Starting Solution...")

        println("")
        println("| Time | max ∂tγrr | max ∂tγθθ | max ∂tKrr | max ∂tKθθ | max ∂tfrrr | max ∂tfrθθ |")
        println("|______|___________|___________|___________|___________|____________|____________|")
        println("")


        sol = solve(
            prob, alg,
            abstol = atol,
            dt = drt/4,
            adaptive = false,
            saveat = printtimes,
            alias_u0 = true,
            progress = true,
            progress_steps = custom_progress_step,
            progress_message = custom_progress_message
        )


        solution_saver(T,grid,sol,param,folder)


    end

    return

end


end
