module GR_Spherical

using DifferentialEquations
using BoundaryValueDiffEq
using OrdinaryDiffEq
#using Fun1d
using DataFrames
using CSV
using Plots
using Roots

using BenchmarkTools
using InteractiveUtils
using RecursiveArrayTools
#using StaticArrays
using LinearAlgebra

using Profile

numvar = 13

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
    init_state::VarContainer{T}
    init_drstate::VarContainer{T}
    init_dr2state::VarContainer{T}
    state::VarContainer{T}
    drstate::VarContainer{T}
    dr2state::VarContainer{T}
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

# fα(M,r,rt) = real((1-2*M/(r(rt))+0im)^(1/2))
# f∂α(M,r,rt) = (M/r(rt)^2)/fα(M,r,rt)
# f∂2α(M,r,rt) = (M/r(rt)^4)*(3*M-2*r(rt))/fα(M,r,rt)^3
#
# fA(rt) = 0.
# fβr(M,rt) = 0.
# fBr(rt) = 0.
# fχ(rt) = 1.
#
# fγtrr(M,r,rt) = (1-2*M/r(rt))^(-1)
# f∂γtrr(M,r,rt) = -(2*M/r(rt)^2)*fγtrr(M,r,rt)^2
# f∂2γtrr(M,r,rt) = (4*M/r(rt)^3)*fγtrr(M,r,rt)^3
#
# fγtθθ(M,r,rt) = r(rt)^2
# f∂γtθθ(M,r,rt) = 2*r(rt)
# f∂2γtθθ(M,r,rt) = 2
#
# fArr(M,∂M,rt) = 0.
# fK(M,∂M,rt) = 0.
#
# fΓtr(M,r,rt) = (3*M-2*r(rt))/(r(rt)^2)
# f∂Γtr(M,r,rt) = 2*(r(rt)-3*M)/(r(rt)^3)

# Sign for Kerr-Schild Coordinates
# +1 for in-going, -1 for out-going (-1 doesn't work, becomes unstable)

s = 1

fα(M,r,rt) = real((1+2*M/(r(rt))+0im)^(-1/2))
f∂α(M,r,rt) = (M/r(rt)^2)*fα(M,r,rt)^3
f∂2α(M,r,rt) = -(M/r(rt)^4)*(M+2*r(rt))*fα(M,r,rt)^5

fβr(M,r,rt) = s*(2*M/r(rt))*fα(M,r,rt)^2
f∂βr(M,r,rt) = -s*2*M/(r(rt)+2*M)^2
f∂2βr(M,r,rt) = s*4*M/(r(rt)+2*M)^3

fχ(M,r,rt) = 1.
f∂χ(M,r,rt) = 0.
f∂2χ(M,r,rt) = 0.

fγtrr(M,r,rt) = 1 + 2*M/r(rt)
f∂γtrr(M,r,rt) = -2*M/r(rt)^2
f∂2γtrr(M,r,rt) = 4*M/r(rt)^3

fγtθθ(M,r,rt) = r(rt)^2
f∂γtθθ(M,r,rt) = 2*r(rt)
f∂2γtθθ(M,r,rt) = 2

fK(M,r,rt) = s*(2*M/r(rt)^3)*(3*M+r(rt))*fα(M,r,rt)^3
f∂K(M,r,rt) = -s*(2*M/r(rt)^5)*(9*M^2+10*M*r(rt)+2*r(rt)^2)*fα(M,r,rt)^5

fArr(M,r,rt) = -s*(4/3)*(M/r(rt)^3)*(2*r(rt)+3*M)*fα(M,r,rt)
f∂Arr(M,r,rt) = s*(4/3)*(M/r(rt)^5)*(15*M^2+15*M*r(rt)+4*r(rt)^2)*fα(M,r,rt)^3

fΓtr(M,r,rt) = -(5*M+2*r(rt))/(r(rt)+2*M)^2
f∂Γtr(M,r,rt) = 2*(r(rt)+3*M)/(r(rt)+2*M)^3

# Minkowski with shift

# fα(M,r,rt) = sqrt(4/3)
# f∂α(M,r,rt) = 0
# f∂2α(M,r,rt) = 0
#
# fβr(M,r,rt) = 2/3
# f∂βr(M,r,rt) = 0
# f∂2βr(M,r,rt) = 0
#
# fχ(M,r,rt) = 1.
# f∂χ(M,r,rt) = 0.
# f∂2χ(M,r,rt) = 0.
#
# fγtrr(M,r,rt) = 3/4
# f∂γtrr(M,r,rt) = 0
# f∂2γtrr(M,r,rt) = 0
#
# fγtθθ(M,r,rt) = r(rt)^2
# f∂γtθθ(M,r,rt) = 2*r(rt)
# f∂2γtθθ(M,r,rt) = 2
#
# fK(M,r,rt) = sqrt(4/3)/r(rt)
# f∂K(M,r,rt) = -sqrt(4/3)/r(rt)^2
#
# fArr(M,r,rt) = -(1/sqrt(12))/r(rt)
# f∂Arr(M,r,rt) = (1/sqrt(12))/r(rt)^2
#
# fΓtr(M,r,rt) = -(8/3)/r(rt)
# f∂Γtr(M,r,rt) = (8/3)/r(rt)^2

# Minkowski without shift

# fα(M,r,rt) = 1
# f∂α(M,r,rt) = 0
# f∂2α(M,r,rt) = 0
#
# fβr(M,r,rt) = 0
# f∂βr(M,r,rt) = 0
# f∂2βr(M,r,rt) = 0
#
# fχ(M,r,rt) = 1.
# f∂χ(M,r,rt) = 0.
# f∂2χ(M,r,rt) = 0.
#
# fγtrr(M,r,rt) = 1
# f∂γtrr(M,r,rt) = 0
# f∂2γtrr(M,r,rt) = 0
#
# fγtθθ(M,r,rt) = r(rt)^2
# f∂γtθθ(M,r,rt) = 2*r(rt)
# f∂2γtθθ(M,r,rt) = 2
#
# fK(M,r,rt) = 0
# f∂K(M,r,rt) = 0
#
# fArr(M,r,rt) = 0
# f∂Arr(M,r,rt) = 0
#
# fΓtr(M,r,rt) = -2/r(rt)
# f∂Γtr(M,r,rt) = 2/r(rt)^2

function init!(state::VarContainer{T}, param) where T

    ############################################
    # Specifies the Initial Conditions
    ############################################

    init_state = param.init_state
    init_drstate = param.init_drstate
    init_dr2state = param.init_dr2state

    α,A,βr,Br,χ,γtrr,γtθθ,Arr,K,Γtr,𝜙,K𝜙,E = state.x
    αi,Ai,βri,Bri,χi,γtrri,γtθθi,Arri,Ki,Γtri,𝜙i,K𝜙i,Ei = init_state.x
    ∂α,∂A,∂βr,∂Br,∂χ,∂γtrr,∂γtθθ,∂Arr,∂K,∂Γtr,∂𝜙,∂K𝜙,∂E = init_drstate.x
    ∂2α,∂2A,∂2βr,∂2Br,∂2χ,∂2γtrr,∂2γtθθ,∂2Arr,∂2K,∂2Γtr,∂2𝜙,∂2K𝜙,∂2E = init_dr2state.x

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

    # Initial conditions for Schwarzschild metric (Kerr-Schild Coordinates)

    # Mass (no real reason not to use 1 here)
    #M = 1

    #Schwarzschild initial conditions

    r0 = 10.
    σr = 0.4
    #Amp = 1.
    Amp = 0.01
    min = 5


    #*(sign(rt-min) + 1)/2
    f𝜙(rt) = Amp*(1/r(rt))*exp(-(1/2)*((r(rt)-r0)/σr)^2)
    fψ(rt) = Amp*exp(-(1/2)*((r(rt)-r0)/σr)^2)*(r(rt)*r0-r(rt)^2-σr^2)/(r(rt)^2*σr^2)
    fΠ(rt) = fβr(M,r,rt)*fψ(rt)

    # fρ(M,rt) = 0*(2*fK𝜙(rt)^2 + (1/2)*(fχ(M,r,rt)/fγtrr(M,r,rt))*f∂𝜙(rt)^2
    #     + (1/2)*m^2*f𝜙(rt)^2)

    fρ(M,rt) = 0

    function f∂rtM(M,rt)
         if r(rt) < 2*M
             return 0.
         else
             4*pi*(r(rt)^2)*fρ(M,rt)*drdrt(rt)
         end
    end

    function f∂M(M,rt)
         if r(rt) < 2*M
             return 0.
         else
             4*pi*(r(rt)^2)*fρ(M,rt)
         end
    end

    # Constraint Equations

    function constraintSystem(M, param, rt)
        f∂rtM(M,rt)*0
    end

    atol = 1e-15

    BVP = ODEProblem(constraintSystem, 1., rtspan, param)
    M = solve(BVP, Tsit5(), abstol=atol, dt=drt, adaptive=false)

    ∂M(rt) = f∂M(M(rt),rt)

    sample!(αi, grid, rt -> fα(M(rt),r,rt) )
    sample!(Ai, grid, rt -> 0 )
    sample!(βri, grid, rt -> fβr(M(rt),r,rt) )
    sample!(Bri, grid, rt -> 0 )
    sample!(χi, grid, rt -> fχ(M(rt),r,rt) )
    sample!(γtrri, grid, rt -> fγtrr(M(rt),r,rt) )
    sample!(γtθθi, grid, rt -> fγtθθ(M(rt),r,rt) )
    sample!(Arri, grid, rt -> fArr(M(rt),r,rt) )
    sample!(Ki, grid, rt -> fK(M(rt),r,rt) )
    sample!(Γtri, grid, rt -> fΓtr(M(rt),r,rt) )
    sample!(𝜙i, grid, f𝜙)
    sample!(K𝜙i, grid, fψ)
    sample!(Ei, grid, rt -> fβr(M(rt),r,rt)*fψ(rt) )

    sample!(∂α, grid, rt -> f∂α(M(rt),r,rt) )
    sample!(∂βr, grid, rt -> f∂βr(M(rt),r,rt) )
    sample!(∂χ, grid, rt -> f∂χ(M(rt),r,rt) )
    sample!(∂γtrr, grid, rt -> f∂γtrr(M(rt),r,rt) )
    sample!(∂γtθθ, grid, rt -> f∂γtθθ(M(rt),r,rt) )
    sample!(∂Arr, grid, rt -> f∂Arr(M(rt),r,rt) )
    sample!(∂K, grid, rt -> f∂K(M(rt),r,rt) )
    sample!(∂Γtr, grid, rt -> f∂Γtr(M(rt),r,rt) )
    sample!(∂𝜙, grid, f𝜙)
    sample!(∂K𝜙, grid, fψ)

    sample!(∂2α, grid, rt -> f∂2α(M(rt),r,rt) )
    sample!(∂2βr, grid, rt -> f∂2βr(M(rt),r,rt) )
    sample!(∂2χ, grid, rt -> f∂2χ(M(rt),r,rt) )
    sample!(∂2γtrr, grid, rt -> f∂2γtrr(M(rt),r,rt) )
    sample!(∂2γtθθ, grid, rt -> f∂2γtθθ(M(rt),r,rt) )
    sample!(∂2𝜙, grid, f𝜙)

    for i in 1:numvar
        if i in reg_list
            sample!(state.x[i], grid, rt -> 1 )
        else
            @. state.x[i] = init_state.x[i]
        end
    end

end

@inline function deriv!(df::Vector{T}, f::Vector{T}, n::Int64, dx::T) where T

    # @inbounds @fastmath @simd

    # df[1] = T((-25. *f[1] + 48. *f[2] - 36. *f[3] + 16. *f[4] - 3. *f[5])/(12. *dx))
    #
    # df[2] = T((-3. *f[1] - 10. *f[2] + 18. *f[3] - 6. *f[4] + f[5])/(12. *dx))

    df[1] = (-48*f[1] + 59*f[2] - 8*f[3] - 3*f[4])/(34*dx)

    df[2] = (-f[1] + f[3])/(2*dx)

    df[3] = (8*f[1] - 59*f[2] + 59*f[4] - 8*f[5])/(86*dx)

    df[4] = (3*f[1] - 59*f[3] + 64*f[5] - 8*f[6])/(98*dx)

    for i in 5:(n - 2)
        df[i] = (f[i-2] - 8*f[i-1] + 8*f[i+1] - f[i+2])/(12*dx)
    end

    df[n-1] = T(-(-3. *f[n] - 10. *f[n-1] + 18. *f[n-2] - 6. *f[n-3] + f[n-4])/(12. *dx))

    df[n] = T(-(-25. *f[n] + 48. *f[n-1] - 36. *f[n-2] + 16. *f[n-3] - 3. *f[n-4])/(12. *dx))

end

@inline function deriv2!(df::Vector{T}, f::Vector{T}, n::Int64, dx::T) where T

    # @inbounds @fastmath @simd

    # df[1] = T((45. *f[1] - 154. *f[2] + 214. *f[3] - 156. *f[4] + 61. *f[5] - 10. *f[6])/(12. *dx^2))
    #
    # df[2] = T((10. *f[1] - 15. *f[2] - 4. *f[3] + 14. *f[4] - 6. *f[5] + f[6])/(12. *dx^2))

    df[1] = (2*f[1] - 5*f[2] + 4*f[3] - f[4])/(dx^2)

    df[2] = (f[1] - 2*f[2] + f[3])/(dx^2)

    df[3] = (-4*f[1] + 59*f[2] - 110*f[3] + 59*f[4] - 4*f[5])/(43*dx^2)

    df[4] = (-f[1] + 59*f[3] - 118*f[4] + 64*f[5] - 4*f[6])/(49*dx^2)

    for i in 5:(n - 2)
        df[i] = (-f[i-2] + 16*f[i-1] - 30*f[i] + 16*f[i+1] - f[i+2])/(12*dx^2)
    end

    df[n-1] = T((10. *f[n] - 15. *f[n-1] - 4. *f[n-2] + 14. *f[n-3] - 6. *f[n-4] + f[n-5])/(12. *dx^2))

    df[n] = T((45. *f[n] - 154. *f[n-1] + 214. *f[n-2] - 156. *f[n-3] + 61. *f[n-4] - 10. *f[n-5])/(12. *dx^2))

end

@inline function dissipation!(df::Vector{T}, f::Vector{T},drdrt::Vector{T}, n::Int64) where T

    ############################################
    # Calculates the numerical dissipation terms
    #
    # Finite differencing fails to model high
    # frequency modes in the system, and these
    # modes can lead to instabilities. Adding
    # numerical dissipation terms damps these
    # high frequency modes by subtracting off
    # a high order derivative from each dynamical
    # variable in the system.
    ############################################

    # @simd @inbounds @fastmath

    # df[1] = (f[1] - 4. *f[2] + 6. *f[3] - 4. *f[4] + f[5])/(drdrt[1])
    #
    # df[2] = (f[1] - 4. *f[2] + 6. *f[3] - 4. *f[4] + f[5])/(drdrt[2])
    #

    df[1] = (-48*f[1] + 96*f[2] - 48*f[3])/(17)

    df[2] = (96*f[1] - 240*f[2] + 192*f[3] - 48*f[4])/(59)

    df[3] = (-48*f[1] + 192*f[2] - 288*f[3] + 192*f[4] - 48*f[5])/(43)

    df[4] = (-48f[2] + 192*f[3] - 288*f[4] + 192*f[5] - 48*f[6])/(49)

    for i in 5:(n - 2)
        df[i] = (-f[i-2] + 4*f[i-1] - 6*f[i] + 4*f[i+1] - f[i+2])
    end

    df[(n-1):n] .= 0.


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
    M = 1.

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
    dr2state = param.dr2state
    dissipation = param.dissipation
    dtstate2 = param.dtstate
    temp = param.temp

    init_state = param.init_state
    init_drstate = param.init_drstate
    init_dr2state = param.init_dr2state

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

    α,A,βr,Br,χ,γtrr,γtθθ,Arr,K,Γtr,𝜙,ψ,Π = state.x
    ∂α,∂A,∂βr,∂Br,∂χ,∂γtrr,∂γtθθ,∂Arr,∂K,∂Γtr,∂𝜙,∂ψ,∂Π = drstate.x
    ∂2α,∂2A,∂2βr,∂2Br,∂2χ,∂2γtrr,∂2γtθθ,∂2Arr,∂2K,∂2Γtr,∂2𝜙,∂2ψ,∂2Π = dr2state.x
    ∂tα,∂tA,∂tβr,∂tBr,∂tχ,∂tγtrr,∂tγtθθ,∂tArr,∂tK,∂tΓtr,∂t𝜙,∂tψ,∂tΠ = dtstate.x
    ∂4α,∂4A,∂4βr,∂4Br,∂4χ,∂4γtrr,∂4γtθθ,∂4Arr,∂4K,∂4Γtr,∂4𝜙,∂4ψ,∂4Π = dissipation.x

    # Dirichlet boundary conditions on scalar field

    #Π[1] = 0.
    #Π[1] = -ψ[1]*(χ[1]/(βr[1]*γtrr[1]))*(α[1]^2 - (βr[1]^2)*γtrr[1]/χ[1])
    # ψ[1] = -Π[1]*βr[1]*γtrr[1]/(χ[1]*(α[1]^2 - (βr[1]^2)*γtrr[1]/χ[1]))
    # ψ[1] = 0.

    # Calculate first derivatives

    for i in 1:numvar
        deriv!(drstate.x[i],state.x[i],n,drt)
    end

    # Calculate second derivatives

    deriv2!(∂2α,α,n,drt)
    deriv2!(∂2βr,βr,n,drt)
    deriv2!(∂2χ,χ,n,drt)
    deriv2!(∂2γtrr,γtrr,n,drt)
    deriv2!(∂2γtθθ,γtθθ,n,drt)

    # Convert between computational rt coordinnate
    # and trasditional r coordinate

    for i in 1:numvar
        @. drstate.x[i] /= drdrt
    end

    @. ∂2α = (∂2α - d2rdrt*∂α)/(drdrt^2)
    @. ∂2βr = (∂2βr - d2rdrt*∂βr)/(drdrt^2)
    @. ∂2χ = (∂2χ - d2rdrt*∂χ)/(drdrt^2)
    @. ∂2γtrr = (∂2γtrr - d2rdrt*∂γtrr)/(drdrt^2)
    @. ∂2γtθθ = (∂2γtθθ - d2rdrt*∂γtθθ)/(drdrt^2)

    # Convert between regularized variables
    # and cannonical variables

    reg = temp.x[1]; ∂reg = temp.x[2]; ∂2reg = temp.x[3];

    for i in reg_list
        @. reg = state.x[i]; @. ∂reg = drstate.x[i]; @. ∂2reg = dr2state.x[i];
        @. state.x[i] *= init_state.x[i]
        @. drstate.x[i] = ∂reg*init_state.x[i] + reg*init_drstate.x[i]
        @. dr2state.x[i] = ∂2reg*init_state.x[i] + 2*∂reg*init_drstate.x[i] + reg*init_dr2state.x[i]
    end

    # Dirichlet boundary conditions on scalar field

    #K𝜙[1] = ∂𝜙[1]*βr[1]/(2*α[1])
    #K𝜙[1] = ∂𝜙[1]*χ[1]*α[1]/(2*βr[1]*γtrr[1])
    #∂𝜙[1] = 0

    #########################################################
    # Evolution Equations
    #
    # This is the full suite of evolution equations
    # for GR in spherical symmetry in the BSSN framework.
    # I have tried to keep them looking as close to their
    # mathematically written counterpart as possible.
    #
    # They are written in the order they appear in the
    # reference (arXiv:0705.3845v2) except for the ∂tBr
    # equation since it contains a ∂tΓtr term.
    #
    #########################################################

    # Lagrangian Gauge condition
    v = 1.

    @. ∂tχ = ((2/3)*K*α*χ - (1/3)*v*βr*χ*∂γtrr/γtrr - (2/3)*v*βr*χ*∂γtθθ/γtθθ
     - (2/3)*v*χ*∂βr + βr*∂χ)

    @. ∂tγtrr = (-2*Arr*α - (1/3)*v*βr*∂γtrr + βr*∂γtrr
     - (2/3)*v*γtrr*βr*∂γtθθ/γtθθ + 2*γtrr*∂βr - (2/3)*v*γtrr*∂βr)

    @. ∂tγtθθ = (Arr*γtθθ*α/γtrr - (1/3)*v*γtθθ*βr*∂γtrr/γtrr - (2/3)*v*βr*∂γtθθ
     + βr*∂γtθθ - (2/3)*v*γtθθ*∂βr)

    @. ∂tArr = (-2*α*(Arr^2)/γtrr + K*α*Arr - (1/3)*v*βr*Arr*∂γtrr/γtrr
     - (2/3)*v*βr*Arr*∂γtθθ/γtθθ - (2/3)*v*Arr*∂βr + 2*Arr*∂βr
     + (2/3)*α*χ*(∂γtrr/γtrr)^2 - (1/3)*α*χ*(∂γtθθ/γtθθ)^2
     - (1/6)*α*(∂χ^2)/χ - (2/3)*α*χ*γtrr/γtθθ + βr*∂Arr
     + (2/3)*α*χ*γtrr*∂Γtr - (1/2)*α*χ*(∂γtrr/γtrr)*(∂γtθθ/γtθθ)
     + (1/3)*χ*∂γtrr*∂α/γtrr + (1/3)*χ*∂α*∂γtθθ/γtθθ - (1/6)*α*∂γtrr*∂χ/γtrr
     - (1/6)*α*∂γtθθ*∂χ/γtθθ - (2/3)*∂α*∂χ - (1/3)*α*χ*∂2γtrr/γtrr
     + (1/3)*α*χ*∂2γtθθ/γtθθ - (2/3)*χ*∂2α + (1/3)*α*∂2χ)

    @. ∂tK = ((3/2)*α*(Arr/γtrr)^2 + (1/3)*α*K^2 + βr*∂K
     + (1/2)*χ*∂γtrr*∂α/(γtrr^2) - χ*∂α*(∂γtθθ/γtθθ)/γtrr
     + (1/2)*∂α*∂χ/γtrr - χ*∂2α/γtrr)

    @. ∂tΓtr = (-v*βr*((∂γtθθ/γtθθ)^2)/γtrr + α*Arr*(∂γtθθ/γtθθ)/(γtrr^2)
     - (1/3)*v*∂βr*(∂γtθθ/γtθθ)/γtrr + ∂βr*(∂γtθθ/γtθθ)/γtrr
     + βr*∂Γtr + α*Arr*∂γtrr/(γtrr^3) - (4/3)*α*∂K/γtrr
     - 2*Arr*∂α/(γtrr^2) + (1/2)*v*∂βr*∂γtrr/(γtrr^2)
     - (1/2)*∂βr*∂γtrr/(γtrr^2) - 3*α*Arr*(∂χ/χ)/(γtrr^2)
     + (1/6)*v*βr*∂2γtrr/(γtrr^2) + (1/3)*v*βr*(∂2γtθθ/γtθθ)/γtrr
     + (1/3)*v*∂2βr/γtrr + ∂2βr/γtrr)

    #######################################################
    # Gauge Evolution

    @. ∂tα = -2*α*A
    @. ∂tA = ∂tK

    @. ∂tβr = (3/4)*Br
    @. ∂tBr = ∂tΓtr

    for i in 1:numvar
        @. dtstate.x[i] = 0.
    end

    # Gauge choices for the evolution of the
    # determinant of the conformal metric
    # (must have v = 1 to use this)

    # ∂tlnγt = temp.x[5]
    # ∂rt∂tlnγt = temp.x[6]
    #
    # ∂tlnγt .= 0
    #
    # #∂tlnγt = -8*pi*Sr.*real((γtθθ./γtrr .+ 0im).^(1/2))
    #
    # deriv!(∂rt∂tlnγt,∂tlnγt,n,drt)
    #
    # @. ∂r∂tlnγt = ∂rt∂tlnγt/drdrt
    #
    # # ∂tα = -(1/2)*α.*∂tlnγt
    # # ∂tβr = (χ./γtrr).*∂tlnγt
    #
    # @. ∂tχ += (1/3)*χ*∂tlnγt
    # @. ∂tγtrr += (1/3)*γtrr*∂tlnγt
    # @. ∂tγtθθ += (1/3)*γtθθ*∂tlnγt
    # @. ∂tArr += (1/3)*Arr*∂tlnγt
    # @. ∂tΓtr += -(1/3)*Γtr*∂tlnγt - (1/6)*(χ/γtrr)*∂r∂tlnγt

    #########################################################
    # Source Terms and Source Evolution
    #
    # This currently includes the addition of source terms
    # to GR that come from a Klein-Gordon scalar field
    #
    #########################################################

    # Klein-Gordon System

    # @. ∂t𝜙 = βr*∂𝜙 - 2*α*K𝜙
    #
    # @. ∂tK𝜙 = (βr*∂K𝜙 + α*K*K𝜙 + (1/2)*(m^2)*α*𝜙 - (1/2)*χ*∂α*∂𝜙/γtrr
    #     - (1/2)*(α*χ/γtrr)*(∂2𝜙 + ∂𝜙*(∂γtθθ/γtθθ - (1/2)*∂γtrr/γtrr
    #     - (1/2)*(∂χ/χ))))

    # Γt = temp.x[5]
    # Γr = temp.x[6]
    #
    # M = 1.
    # @. Γt = -2*M/r^2
    # @. Γr = 2*(M-r)/r^2
    #
    # @. ∂t𝜙 = Π
    # @. ∂tψ = ∂Π
    #
    # # @. ∂tΠ = (2*βr*∂Π + ∂βr*Π - 2*βr*ψ*∂βr - βr*Π*∂α/α + (βr^2)*ψ*∂α/α
    # #  + α*χ*ψ*∂α/γtrr + (1/2)*βr*Π*∂γtrr/γtrr - (1/2)*(βr^2)*ψ*∂γtrr/γtrr
    # #  - (1/2)*(α^2)*χ*ψ*∂γtrr/γtrr^2 + βr*Π*∂γtθθ/γtθθ - (βr^2)*ψ*∂γtθθ/γtθθ
    # #  + (α^2)*χ*ψ*∂γtθθ/(γtθθ*γtrr) - (3/2)*βr*Π*∂χ/χ - (1/2)*(α^2)*ψ*∂χ/γtrr
    # #  + (3/2)*(βr^2)*ψ*∂χ/χ - (βr^2)*∂ψ + (α^2)*χ*∂ψ/γtrr)
    #
    # @. ∂tΠ = (α^2)*((χ/γtrr-(βr/α)^2)*∂ψ + 2*(βr/α^2)*∂Π - Γr*∂𝜙 - Γt*Π - m^2*𝜙)

    g = temp.x[5]

    @. g = -α^2*γtrr*γtθθ^2/χ^3

    ut = ψ
    ur = Π
    ∂ut = ∂ψ
    ∂ur = ∂Π
    ∂tut = ∂tψ
    ∂tur = ∂tΠ

    @. ∂t𝜙 = (-(α^2-γtrr*(βr)^2/χ)*ut + (γtrr*βr/χ)*ur)/sqrt(-g)
    @. ∂tur = (βr^2 - χ*α^2/γtrr)*∂ut + 2*βr*∂ur + (2/r)*ut - (6*M/r^2)*(ur + ut)
    @. ∂tut = -∂ur + m^2*𝜙


    ρ = temp.x[7]
    Sr = temp.x[8]
    S = temp.x[9]
    Srr = temp.x[10]

    @. ρ = (1/2)*(Π - βr*ψ)^2/α^2 + (1/2)*(χ/γtrr)*ψ^2 + (1/2)*(m^2)*𝜙^2
    #Lower Index
    @. Sr = -ψ*(Π - βr*ψ)/α
    @. S = (3/2)*(Π - βr*ψ)^2/α^2 - (1/2)*(χ/γtrr)*ψ^2 - (3/2)*(m^2)*𝜙^2
    @. Srr = (γtrr/χ)*( (Π - βr*ψ)^2/α^2 + (1/2)*(χ/γtrr)*ψ^2 - (1/2)*(m^2)*𝜙^2)

    # @. ∂tArr += -8*pi*α*(χ*Srr - (1/3)*S*γtrr)
    # @. ∂tK += 4*pi*α*(ρ + S)
    # @. ∂tΓtr += -16*pi*α*Sr/γtrr

    # fr = param.r
    #
    # fα(M,rt) = real((1+2*M/(fr(rt))+0im)^(-1/2))
    # fβr(M,rt) = (2*M/fr(rt))*fα(M,rt)^2
    # fγtrr(M,rt) = 1+2*M/fr(rt)
    # fγtθθ(rt) = fr(rt)^2
    # fArr(M,∂M,rt) = (4/3)*(fr(rt)*(M+fr(rt))*∂M-M*(3*M+2*fr(rt)))/real(((fr(rt)^5)*(fr(rt)+2*M)+0im)^(1/2))
    # fK(M,∂M,rt) = (2*M*(3*M+fr(rt))+2*fr(rt)*∂M*(M+fr(rt)))/real((fr(rt)*(fr(rt)+2*M)+0im)^(3/2))
    # fΓtr(M,∂M,rt) = (fr(rt)*∂M-2*fr(rt)-5*M)/(fr(rt)+2*M)^2
    #
    # f∂α(M,rt) = M*real((fr(rt)*(fr(rt)+2*M+0im)^3)^(-1/2))
    # f∂βr(M,rt) = -2*M/(fr(rt)+2*M)^2
    # f∂γtrr(M,rt) = -2*M/(fr(rt)^2)
    # f∂γtθθ(rt) = 2*fr(rt)
    # f∂Arr(M,rt) = (4*M/3)*(15*M^2+15*M*fr(rt)+4*fr(rt)^2)/real(((fr(rt)^7)*((fr(rt)+2*M)^3)+0im)^(1/2))
    # f∂K(M,rt) = -2*M*(9*M^2+10*M*fr(rt)+2*fr(rt)^2)/real((fr(rt)*(fr(rt)+2*M)+0im)^(5/2))
    # f∂Γtr(M,rt) = 2*(fr(rt)+3*M)/(fr(rt)+2*M)^3
    #
    # rt = sample(Float64, A.grid, rt->rt)
    #
    # # for i=1:2
    #
    # ∂tα[1:2] .= (α[1:2] .- fα.(1.,rt[1:2]))./r[1:2] + ∂α[1:2] - f∂α.(1.,rt[1:2])
    # ∂tA[1:2] .= (A[1:2] .- 0.)./r[1:2] + ∂A[1:2]
    # ∂tβr[1:2] .= (βr[1:2] .- fβr.(1.,rt[1:2]))./r[1:2] + ∂βr[1:2] - f∂βr.(1.,rt[1:2])
    # ∂tBr[1:2] .= (Br[1:2] .- 0.)./r[1:2] + ∂Br[1:2]
    # ∂tχ[1:2] .= (χ[1:2] .- 1.)./r[1:2] + ∂χ[1:2]
    # ∂tγtrr[1:2] .= (γtrr[1:2] .- fγtrr.(1.,rt[1:2]))./r[1:2] + ∂γtrr[1:2] - f∂γtrr.(1.,rt[1:2])
    # ∂tγtθθ[1:2] .= (γtθθ[1:2] .- fγtθθ.(rt[1:2]))./r[1:2] + ∂γtθθ[1:2] - f∂γtθθ.(rt[1:2])
    # ∂tArr[1:2] .= (Arr[1:2] .- fArr.(1.,0.,rt[1:2]))./r[1:2] + ∂Arr[1:2] - f∂Arr.(1.,rt[1:2])
    # ∂tK[1:2] .= (K[1:2] .- fK.(1.,0.,rt[1:2]))./r[1:2] + ∂K[1:2] - f∂K.(1.,rt[1:2])
    # ∂tΓtr[1:2] .= (Γtr[1:2] .- fΓtr.(1.,0.,rt[1:2]))./r[1:2] + ∂Γtr[1:2] - f∂Γtr.(1.,rt[1:2])
    # ∂t𝜙[1:2] .= (𝜙[1:2] .- 0.)./r[1:2] + ∂𝜙[1:2]
    # ∂tK𝜙[1:2] .= (K𝜙[1:2] .- 0.)./r[1:2] + ∂K𝜙[1:2]

    ######################################################



    # Specify the inner temporal boundary conditions

    for i in 1:(numvar-3)
        dtstate.x[i][1] = 0.
    end

    c_p = α[1]*sqrt(χ[1]/γtrr[1]) - βr[1]
    c_m = -α[1]*sqrt(χ[1]/γtrr[1]) - βr[1]
    c_in = -1

    gtt = -1/α[1]^2
    grt = βr[1]/α[1]^2
    grr = χ[1]/γtrr[1] - (βr[1]/α[1])^2

    ∂t𝜙[1] = 0
    ∂tut[1] = -∂ur[1] + c_in*(gtt*∂ut[1] - 2*grt*∂ur[1])/(grr*gtt)
    #∂tψ[1] = ∂Π[1]
    ∂tur[1] = 0


    # Calculate the numerical dissipation

    # Magnitude of dissipation
    σ = 0.3

    for i in 1:numvar
        dissipation!(dissipation.x[i],state.x[i],drdrt,n)
        @. dtstate.x[i] += σ*dissipation.x[i]/16
    end


    # σ1 = -2
    # σ2 = -2
    # σ3 = -1
    #
    # #∂tψ[1] += σ1*(ψ[1] + Π[1]*βr[1]*γtrr[1]/(χ[1]*(α[1]^2 - (βr[1]^2)*γtrr[1]/χ[1])))
    # ∂t𝜙[1] += (48/17)*σ1*𝜙[1]/drt
    # ∂tψ[1] += (48/17)*σ2*ψ[1]/drt
    # ∂tΠ[1] += (48/17)*σ3*Π[1]/drt

    # Convert back to regularized variables
    # for the time derivatives

    for i in reg_list
        @. dtstate.x[i] /= init_state.x[i]
    end

    # Specify the outer temporal boundary conditions

    for i in 1:numvar
        dtstate.x[i][n] = 0.
    end

    # Store the calculated state into the param
    # so that we can print it to the screen

    for i in 1:numvar
        dtstate2.x[i] .= dtstate.x[i]
    end

end

function rhs_all(regstate::VarContainer{T}, param::Param{T}, t) where T

    n = param.grid.ncells + 2

    dtstate = similar(ArrayPartition,T,n)

    rhs!(dtstate,regstate,param,t)

    return dtstate

end

function constraints(state::VarContainer{T},drstate::VarContainer{T},dr2state::VarContainer{T},param) where T

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

    α,A,βr,Br,χ,γtrr,γtθθ,Arr,K,Γtr,𝜙,ψ,Π = state.x
    ∂α,∂A,∂βr,∂Br,∂χ,∂γtrr,∂γtθθ,∂Arr,∂K,∂Γtr,∂𝜙,∂ψ,∂Π = drstate.x
    ∂2α,∂2A,∂2βr,∂2Br,∂2χ,∂2γtrr,∂2γtθθ,∂2Arr,∂2K,∂2Γtr,∂2𝜙,∂2ψ,∂2Π = dr2state.x

    init_state = param.init_state
    init_drstate = param.init_drstate
    init_dr2state = param.init_dr2state

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

    deriv!(∂χ,χ,n,drt)
    deriv!(∂γtrr,γtrr,n,drt)
    deriv!(∂γtθθ,γtθθ,n,drt)
    deriv!(∂Arr,Arr,n,drt)
    deriv!(∂K,K,n,drt)
    deriv!(∂Γtr,Γtr,n,drt)

    deriv2!(∂2χ,χ,n,drt)
    deriv2!(∂2γtrr,γtrr,n,drt)
    deriv2!(∂2γtθθ,γtθθ,n,drt)

    ∂χ ./= drdrt
    ∂γtrr ./= drdrt
    ∂γtθθ ./= drdrt
    ∂Arr ./= drdrt
    ∂K ./= drdrt
    ∂Γtr ./= drdrt

    @. ∂2χ = (∂2χ - d2rdrt*∂χ)/(drdrt^2)
    @. ∂2γtrr = (∂2γtrr - d2rdrt*∂γtrr)/(drdrt^2)
    @. ∂2γtθθ = (∂2γtθθ - d2rdrt*∂γtθθ)/(drdrt^2)

    reg = temp.x[1]
    ∂reg = temp.x[2]
    ∂2reg = temp.x[3]

    for i in reg_list
        @. reg = state.x[i]; @. ∂reg = drstate.x[i]; @. ∂2reg = dr2state.x[i];
        @. state.x[i] *= init_state.x[i]
        @. drstate.x[i] = ∂reg*init_state.x[i] + reg*init_drstate.x[i]
        @. dr2state.x[i] = ∂2reg*init_state.x[i] + 2*∂reg*init_drstate.x[i] + reg*init_dr2state.x[i]
    end

    # @. ∂2χ = (∂2χ - d2rdrt*∂χ)/(drdrt^2)
    # @. ∂2γtrr = (∂2γtrr - d2rdrt*∂γtrr)/(drdrt^2)
    # @. ∂2γtθθ = (∂2γtθθ - d2rdrt*∂γtθθ)/(drdrt^2)
    # @. ∂2𝜙 = (∂2𝜙 - d2rdrt*∂𝜙)/(drdrt^2)
    #
    # @. γtrr = γtrr/r + 1
    # @. ∂γtrr = (1 - γtrr + ∂γtrr)/r
    # @. ∂2γtrr = (∂2γtrr - 2*∂γtrr)/r
    #
    # @. γtθθ = (r^2)*(γtθθ + 1)
    # @. ∂γtθθ = (2*γtθθ + ∂γtθθ*r^3)/r
    # @. ∂2γtθθ = (4*∂γtθθ*r - 6*γtθθ + ∂2γtθθ*r^4)/(r^2)
    #
    # @. K = sqrt(r^(-3))*K
    # @. ∂K = sqrt(r^(-3))*∂K - (3/2)*K/r
    #
    # @. Arr = sqrt(r^(-5))*Arr
    # @. ∂Arr = sqrt(r^(-5))*∂Arr - (5/2)*Arr/r


    #ρ = 2*K𝜙.^2 + (1/2)*(χ./γtrr).*∂𝜙.^2 + (1/2)*m^2*𝜙.^2

    ρ = temp.x[4]
    Sr = temp.x[5]
    γ = temp.x[6]
    Er = temp.x[6]

    @. ρ = (1/2)*(Π - βr*ψ)^2/α^2 + (1/2)*(χ/γtrr)*ψ^2 + (1/2)*(m^2)*𝜙^2
    #Lower Index
    @. Sr = -ψ*(Π - βr*ψ)/α

    @. γ = γtrr*(γtθθ^2)/χ^3

    norm = ones(T,n)
    norm[1] = 17/48
    norm[2] = 59/48
    norm[3] = 43/48
    norm[4] = 49/48

    @. Er = norm*sqrt(γ)*(α*ρ - βr*Sr)*drdrt

    #@. Er = drt*norm*sqrt(γ)*((βr*∂𝜙 - 2*K𝜙*α)^2 + (χ/γtrr)*∂𝜙^2)*drdrt

    E = 0

    for i in 1:n
        E += drt*Er[i]
    end

    # Constraint Equations

    𝓗 = (-(3/2)*(Arr./γtrr).^2 + (2/3)*K.^2 - (5/2)*((∂χ.^2)./χ)./γtrr
     + 2*∂2χ./γtrr + 2*χ./γtθθ - 2*χ.*(∂2γtθθ./γtθθ)./γtrr + 2*∂χ.*(∂γtθθ./γtθθ)./γtrr
     + χ.*(∂γtrr./(γtrr.^2)).*(∂γtθθ./γtθθ) - ∂χ.*∂γtrr./(γtrr.^2)
     + (1/2)*χ.*((∂γtθθ./γtθθ).^2)./γtrr - 16*pi*ρ)

    𝓜r = (∂Arr./γtrr - (2/3)*∂K - (3/2)*Arr.*(∂χ./χ)./γtrr
     + (3/2)*Arr.*(∂γtθθ./γtθθ)./γtrr - Arr.*∂γtrr./(γtrr.^2)
     - 8*pi*Sr)

    𝓖r = -(1/2)*∂γtrr./(γtrr.^2) + Γtr + (∂γtθθ./γtθθ)./γtrr

    return (𝓗, 𝓜r, 𝓖r, E)

end

function horizon(state::VarContainer{T},param) where T

    ############################################
    # Caculates the apparent horizon
    #
    # Where the function crosses zero is the
    # apparent horizon of the black hole.
    ############################################

    #v = param[3]

    # Unpack Variables

    χ = state.χ
    γtrr = state.γtrr
    γtθθ = state.γtθθ
    Arr = state.Arr
    K = state.K

    # Gauge condition

    # if v == 1 # Lagrangian Condition
    #     γtθθreg[2] = ((-315*γtθθreg[3] + 210*γtθθreg[4] - 126*γtθθreg[5]
    #     + 45*γtθθreg[6] - 7*γtθθreg[7])/63)
    # end

    #r = sample(T, χ.grid, param[4])
    drdrt = param.drdrtsamp

    # Conversions from regularized variables to canonical variables

    # γtrr = γtrrreg./r .+ 1
    #
    # γtθθ = (r.^2).*(γtθθreg .+ 1)
    #
    # K = real((r .+ 0im).^(-3/2)).*Kreg
    #
    # Arr = real((r .+ 0im).^(-5/2)).*Arrreg

    # Intermediate calculations

    Kθθ = ((1/3)*γtθθ.*K - (1/2)*Arr.*γtθθ./γtrr)./χ

    grr =  γtrr./χ

    gθθ =  γtθθ./χ

    # Spatial Derivatives

    ∂rtgθθ = deriv(gθθ,4,1)

    # Coordinate transformations from computational rt coordinate
    # to physical r coordinate

    ∂gθθ = ∂rtgθθ./drdrt

    # Apparent horizon function

    Θ = (∂gθθ./gθθ)./real((grr .+ 0im).^(1/2)) - 2*Kθθ./gθθ

    # cross = GridFun(χ.grid, sign.(Θ))

    return Θ

end

# function crossings(Θ::GridFun)
#
#     GridFun(χ.grid, sign.(Θ))
#
#     for
#
# end


function custom_progress_message(dt,state::VarContainer{T},param,t) where T

    ###############################################
    # Outputs status numbers while the program runs
    ###############################################

    dtstate = param.dtstate::VarContainer{T}

    ∂tα,∂tA,∂tβr,∂tBr,∂tχ,∂tγtrr,∂tγtθθ,∂tArr,∂tK,∂tΓtr,∂t𝜙,∂tK𝜙 = dtstate.x

    println("  ",
    #rpad(string(param[1]),6," "),
    #rpad(string(round(dt,digits=3)),10," "),
    rpad(string(round(t,digits=1)),10," "),
    rpad(string(round(maximum(abs.(∂tα)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂tχ)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂tγtrr)),digits=3)),14," "),
    rpad(string(round(maximum(abs.(∂tγtθθ)),digits=3)),14," "),
    rpad(string(round(maximum(abs.(∂tArr)),digits=3)),12," "),
    # rpad(string(round(maximum(abs.(derivstate.𝜙)),digits=3)),12," "),
    # rpad(string(round(maximum(abs.(derivstate.K𝜙)),digits=3)),14," ")
    rpad(string(round(maximum(abs.(∂tK)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂tΓtr)),digits=3)),14," ")
    )

    #PrettyTables.jl

    # param[1] += param[2]

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

    vars = (["α","A","βr","Br","χ","γtrr","γtθθ","Arr","K","Γtr","𝜙","ψ","Π",
    "∂tα","∂tA","∂tβr","∂tBr","∂tχ","∂tγtrr","∂tγtθθ","∂tArr","∂tK","∂tΓtr","∂t𝜙","∂tψ",
    "∂tΠ","H","Mr","Gr","E","appHorizon"])
    varlen = length(vars)
    #mkdir(string("data\\",folder))
    tlen = size(sol)[2]
    rlen = grid.ncells + 2
    r = param.rsamp
    rtmin = param.rtmin
    #cons = Array{GridFun,2}(undef,tlen,4)
    #state = Array{ArrayPartition,1}(undef,tlen)
    #dstate = Array{ArrayPartition,1}(undef,tlen)
    #d2state = Array{ArrayPartition,1}(undef,tlen)
    #derivs = Array{ArrayPartition,1}(undef,tlen)
    #apphorizon = Array{GridFun,1}(undef,tlen)

    # for i in 1:tlen
    #
    #     αreg,A,βr,Br,χ,γtrrreg,γtθθreg,Arrreg,Kreg,Γtr,𝜙,K𝜙 = sol[i].x
    #
    #     # Conversions from regularized variables to canonical variables
    #     α = real((1 .+ αreg./r .+ 0im).^(-1/2))
    #     γtrr = γtrrreg./r .+ 1
    #     γtθθ = (r.^2).*(γtθθreg .+ 1)
    #     K = sqrt.(r.^(-3)).*Kreg
    #     Arr = sqrt.(r.^(-5)).*Arrreg
    #
    #     state[i] = ArrayPartition(α,A,βr,Br,χ,γtrr,γtθθ,Arr,K,Γtr,𝜙,K𝜙)
    #     derivs[i] = param.dtstate
    #     rhs!(derivs[i],sol[i],param,0)
    #     #cons[i,1:4] .= constraints!(T,state[i],dstate[i],d2state[i],param)
    #     #apphorizon[i] = horizon(state[i],param)
    # end

    drstate = param.drstate
    dr2state = param.dr2state

    dtstate = [rhs_all(sol[i],param,0.) for i = 1:tlen]

    cons = [constraints(sol[i],drstate,dr2state,param) for i = 1:tlen]

    array = Array{T,2}(undef,tlen+1,rlen+1)

    array[1,1] = 0
    array[1,2:end] .= r

    for j = 1:varlen
        if j < numvar+1
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                array[i,2:end] .= sol[i-1].x[j]
                # if j==1
                #     println(array[i,2:10])
                # end
            end
        elseif j < 2*numvar+1

            for i = 2:tlen+1

                array[i,1] = sol.t[i-1]
                array[i,2:end] .= dtstate[i-1].x[j-numvar]
            end
        elseif j < 2*numvar+5
            for i = 2:tlen+1
                #println(size(cons[i-1][j-24]))
                array[i,1] = sol.t[i-1]
                array[i,2:end] .= cons[i-1][j-2*numvar]
            end
        elseif j == 2*numvar+4
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                array[i,2] = cons[i-1][j-2*numvar]
                array[i,3:end] .= 0
            end
        end

        CSV.write(
        string("data/",folder,"/",vars[j],".csv"),
        DataFrame(array, :auto),
        header=false
        )

    end

    # for i = 2:tlen+1
    #     array[i,1] = sol.t[i-1]
    #     array[i,2:end] .= sol[i-1].x[13]
    # end
    #
    # CSV.write(
    # string("data/",folder,"/","p-",rtmin,".csv"),
    # DataFrame(array, :auto),
    # header=false
    # )

end

function main(points)

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

    for i = 1:1


        T = Float64

        rtspan = T[2.,22.] .+ (1.0 - 0.1*i)
        rtmin, rtmax = rtspan
        rspan = T[rtmin,rtmax*10.]

        f(x) = x*tan((rtmax-rtmin)/x) + rtmin - rspan[2]

        rs = find_zero(f, 0.64*rtmax)

        r(rt) = rs*tan((rt-rtmin)/rs) + rtmin
        drdrt(rt) = sec((rt-rtmin)/rs)^2
        d2rdrt(rt) = (2/rs)*(sec((rt-rtmin)/rs)^2)*tan((rt-rtmin)/rs)

        # r(rt) = rt
        # drdrt(rt) = 1
        # d2rdrt(rt) = 0

        println("Mirror: ",rtmin)

        domain = Domain{T}(rtmin, rtmax)
        grid = Grid(domain, points)

        n = grid.ncells + 2

        drt = spacing(grid)
        dt = drt/4.

        tspan = T[0., 15.]
        tmin, tmax = tspan

        printtimes = 0.5

        v = 1.

        m = 0.

        Mtot = 1.

        # α,A,βr,Br,χ,γtrr,γtθθ,Arr,K,Γtr,𝜙,K𝜙,p = state.x
        reg_list = [1,3,6,7,8,9,10]
        #reg_list = [7,8,9,10]
        #reg_list = [10]

        atol = eps(T)^(T(3) / 4)

        alg = RK4()
        #alg = Cash4()
        #alg = Rodas4()

        #printlogo()

        custom_progress_step = round(Int, printtimes/dt)
        step_iterator = custom_progress_step

        regstate = similar(ArrayPartition,T,n)

        state = similar(ArrayPartition,T,n)
        drstate = similar(ArrayPartition,T,n)
        dr2state = similar(ArrayPartition,T,n)

        init_state = similar(ArrayPartition,T,n)
        init_drstate = similar(ArrayPartition,T,n)
        init_dr2state = similar(ArrayPartition,T,n)

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
        rsamp,drdrtsamp,d2rdrtsamp,
        init_state,init_drstate,init_dr2state,
        state,drstate,dr2state,
        dtstate,dissipation,temp)

        init!(regstate, param)

        prob = ODEProblem(rhs!, regstate, tspan, param)

        #println("Starting Solution...")

        println("")
        println("| Time | max α'(t) | max χ'(t) | max γtrr'(t) | max γtθθ'(t) | max Arr'(t) | max K'(t) | max Γtr'(t) |")
        println("|______|___________|___________|______________|______________|_____________|___________|____________|")
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


        solution_saver(T,grid,sol,param,"test")


    end

    return

end


end

# function M_init(::Type{T}, grid::Grid, param) where {T}
#
#     ############################################
#     # Specifies the Initial Conditions
#     ############################################
#
#     n = grid.ncells + 4
#     domain = grid.domain
#     initgrid = grid
#     drt = spacing(grid)
#     r = param[4]
#     drdrt = param[5]
#     d2rdrt = param[6]
#     m = param[7]
#     rtspan = param[8]
#
#     num = 0
#
#     fρ(M,rt) = (2*fK𝜙(rt)^2 + (1/2)*(fχ(rt)/fγtrr(M,rt))*f∂𝜙(rt)^2
#         + (1/2)*m^2*f𝜙(rt)^2)
#
#     fSr(rt) = 2*fK𝜙(rt)*f∂𝜙(rt)
#
#     #f∂M(M,rt) = 4*pi*(r(rt)^2)*fρ(M,rt)
#
#     function f∂M(M,rt)
#          if rt < 2
#              return 0.
#          else
#              4*pi*(r(rt)^2)*fρ(M,rt)
#          end
#     end
#
#     function f𝓗(M,∂M,rt)
#          (-(3/2)*(fArr(M,∂M,rt)/fγtrr(M,rt))^2 + (2/3)*fK(M,∂M,rt)^2
#          - (5/2)*((f∂χ(rt)^2)/fχ(rt))/fγtrr(M,rt) + 2*f∂2χ(rt)/fγtrr(M,rt)
#          + 2*fχ(rt)/fγtθθ(rt) - 2*fχ(rt)*(f∂2γtθθ(rt)/fγtθθ(rt))/fγtrr(M,rt)
#          + 2*f∂χ(rt)*(f∂γtθθ(rt)/fγtθθ(rt))/fγtrr(M,rt)
#          + fχ(rt)*(f∂γtrr(M,∂M,rt)/(fγtrr(M,rt)^2))*(f∂γtθθ(rt)/fγtθθ(rt))
#          - f∂χ(rt)*f∂γtrr(M,∂M,rt)/(fγtrr(M,rt)^2)
#          + (1/2)*fχ(rt)*((f∂γtθθ(rt)/fγtθθ(rt))^2)/fγtrr(M,rt) - 16*pi*fρ(M,rt))
#     end
#
#     fαreg(M,rt) = 2*M
#     fγtrrreg(M,rt) = 2*M
#     fArrreg(M,∂M,rt) = real((r(rt)+ 0im)^(5/2))*fArr(M,∂M,rt)
#     fKreg(M,∂M,rt) = real((r(rt)+ 0im)^(3/2))*fK(M,∂M,rt)
#
#     # Constraint Equations
#
#     rtspan = (rtspan[1], rtspan[2])
#     #rtspan = (rtspan[2], 0.5)
#
#     function constraintSystem(M, param, rt)
#         f∂M(M,rt)
#     end
#
#     # function boundaryCondition!(residual, M, param, rt)
#     #     residual = M[end] - 1. #inner boundary condition
#     # end
#
#     atol = 1e-15
#
#     BVP = ODEProblem(constraintSystem, 1., rtspan, param)
#     M = solve(BVP, Tsit5(), abstol=atol, dt=drt, adaptive=false)
#
#     ∂M(rt) = f∂M(M(rt),rt)
#
#     # M(rt) = 1.
#     # ∂M(rt) = 0
#
#
#     state = GridFun(grid, M)
#
#     return
#
# end
#
# function M_rhs(M::GridFun, param, t)
#
#     #########################################################
#     # Source Terms and Source Evolution
#     #
#     # This currently includes the addition of source terms
#     # to GR that come from a Klein-Gordon scalar field
#     #
#     #########################################################
#
#     # Klein-Gordon System
#
#     ∂t𝜙 = βr.*∂𝜙 - 2*α.*K𝜙
#     ∂tK𝜙 = (βr.*∂K𝜙 + α.*K.*K𝜙 - (1/2)*α.*χ.*∂2𝜙./γtrr
#         + (1/4)*α.*χ.*∂γtrr.*∂𝜙./γtrr.^2 - (1/4)*α.*∂χ.*∂𝜙./γtrr
#         - (1/2)*χ.*∂α.*∂𝜙./γtrr - (1/2)*χ.*∂γtθθ.*∂𝜙./(γtrr.*γtθθ)
#         + (1/2)*∂χ.*∂𝜙./(γtrr) + (1/2)*m^2*𝜙)
#
#     ρ = 2*K𝜙.^2 + (1/2)*(χ./γtrr).*∂𝜙.^2 + (1/2)*m^2*𝜙.^2
#     #Lower Index
#     Sr = 2*γtrr.*K𝜙.*∂𝜙./χ
#     # S = 6*K𝜙.^2 - (1/2)*(χ./γtrr).*∂𝜙.^2 - (3/2)*m^2*𝜙.^2
#     # Srr = (γtrr./χ).*(2*K𝜙.^2 + (1/2)*(χ./γtrr).*∂𝜙.^2 - (1/2)*m^2*𝜙.^2)
#
#     # ∂tArr .+= -8*pi*α.*(χ.*Srr - (1/3)*S.*γtrr)
#     # ∂tK .+= 4*pi*α.*(ρ + S)
#     # ∂tΓtr .+= -16*pi*α.*Sr./γtrr
#
#     # Inner temporal boundary Conditions
#
#     # ∂tα[1:2] .= 0.
#     # ∂tA[1:2] .= 0.
#     # ∂tβr[1:2] .= 0.
#     # ∂tBr[1:2] .= 0.
#     # ∂tχ[1:2] .= 0.
#     # ∂tγtrr[1:2] .= 0.
#     # ∂tγtθθ[1:2] .= 0.
#     # ∂tArr[1:2] .= 0.
#     # ∂tK[1:2] .= 0.
#     # ∂tΓtr[1:2] .= 0.
#     # ∂t𝜙[1:2] .= 0.
#     # ∂tK𝜙[1:2] .= 0.
#
#     return GBSSN_Variables(∂tαreg,∂tA,∂tβr,∂tBr,∂tχ,∂tγtrrreg,∂tγtθθreg,∂tArrreg,∂tKreg,∂tΓtr,∂t𝜙,∂tK𝜙)
#
# end
