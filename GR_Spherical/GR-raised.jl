module GR_Spherical

using DifferentialEquations
using BoundaryValueDiffEq
using OrdinaryDiffEq
#using Roots

using InteractiveUtils
using RecursiveArrayTools
using SparseArrays
using LinearAlgebra

using Distributions
using ForwardDiff

using HDF5
using FileIO

using LoopVectorization
using BenchmarkTools

# Include the input parameter file

include("inputfile.jl")

# Macro for applying get_index to an expression
# helps to clean up the boundary conditions
macro part(index, expr)
    esc(parse_index(index, expr))
end

function parse_index(index, expr)
    # Do nothing to numbers
    expr isa Number && return expr
    # Add the `expr_index` call to symbols
    expr isa Symbol && return :(expr_index($expr, $index))
    if expr isa Expr
        # if the expr is an assignment, assign that index
        expr.head ≡ :(=) && return :(setindex!($(expr.args[1]),
                               $(parse_index(index, expr.args[2])),
                               $index))
       # Handle function calls recursively for each argument.
       # `args[1]` is the function itself, which isn't modified.
       # `args[2:end]` are the function arguments.
       expr.head ≡ :call && return Expr(expr.head,
                                         expr.args[1],
                                         parse_index.(index, @view expr.args[2:end])...)
    end
    # Abort if we find something unexpected
    @assert false
end

expr_index(x::Number, i...) = x
expr_index(a::AbstractArray, i...) = a[i...]

# Total number of variables in the state vector
const numvar = 9

# Type to store all of the grid functions for the ODE Solver
VarContainer{T} = ArrayPartition{T, NTuple{numvar+1,Vector{T}}}

struct Domain{S}
    rmin::S
    rmax::S
end

struct Grid{S}
    domain::Domain{S}
    ncells::Int
end

# Main parameter struct passed to ODE Solver
struct Param{T}
    grid::Grid{T}
    gauge::VarContainer{T}
    drstate::VarContainer{T}
    temp::VarContainer{T}
end

# Defines how to allocate the grid functions
@inline function Base.similar(::Type{ArrayPartition},::Type{T},size::Int) where T
    return ArrayPartition([similar(Vector{T}(undef,size)) for i=1:numvar]...,similar(Vector{T}(undef,2)))::VarContainer{T}
end

# Just a fancy ASCII logo for the program
function printlogo()
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

# Sample analytic functions to the grid
function sample!(f::Vector{T}, grid::Grid{S}, fun) where {S,T}

    rmin = grid.domain.rmin

    f .= T[fun(rmin + dr*(j-1)) for j in 1:(grid.ncells)]

end

#############################
#   Define coordinate systems
#############################

# All metric variables that explictly depend on M
# need a function that overloads the M argument to a ::Number
# so the constraint solver can work properly.
#
# To define a new coordinate system, all you need is to specify
# the set (ᾶ, βʳ, γrr, γθθ) as functions of (M,r),
# All of the other variables are fixed automatically based on definitions

# Kerr-Schild Coordinates
# sign=1 for ingoing (black hole), sign=-1 for outgoing (white hole)

sign = 1.

fᾶ(M,r) = 1/(r^2 + 2*M(r)*r)
fβʳ(M,r) = sign*2*M(r)/(2*M(r)+r)
fγrr(M,r) = 1 + 2*M(r)/r
fγθθ(M,r) = r^2

fᾶ(M::Number,r) = 1/(r^2+2*M*r)
fβʳ(M::Number,r) = sign*2*M/(2*M+r)
fγrr(M::Number,r) = 1 + 2*M/r

#Painleve-Gullstrand Coordinates

# fᾶ(M,r) = 1.
# fβʳ(M,r) = sqrt(2*M(r)/r)
# fγrr(M,r) = 1.
# fγθθ(M,r) = r^2
#
# fβʳ(M::Number,r) = sqrt(2*M/r)

# Schwarzschild
#
# fᾶ(M,r) = sqrt(1. - 2*M(r)/r)
# fβʳ(M,r) = 0.
# fγrr(M,r) = 1/(1 - 2*M(r)/r)
# fγθθ(M,r) = r^2
#
# fᾶ(M::Number,r) = sqrt(1. - 2*M/r)
# fγrr(M::Number,r) = 1/(1 - 2*M/r)

# Define derivatives, extrinsic curavture, and the f_{ijk} variables
# using ForwardDiff so that you can switch coordinate systems
# without having to explicitly calculate the derivatives of the
# functions.

fcp(M,r) = -fβʳ(M,r) + fα(M,r)/sqrt(fγrr(M,r))
fcm(M,r) = -fβʳ(M,r) - fα(M,r)/sqrt(fγrr(M,r))

f∂ᵣᾶ(M,r)         = ForwardDiff.derivative(r -> fᾶ(M,r), r)
f∂ᵣ2ᾶ(M,r)        = ForwardDiff.derivative(r -> f∂ᵣᾶ(M,r), r)
f∂ᵣβʳ(M,r)        = ForwardDiff.derivative(r -> fβʳ(M,r), r)
f∂ᵣ2βʳ(M,r)       = ForwardDiff.derivative(r -> f∂ᵣβʳ(M,r), r)

fα(M,r)           = fᾶ(M,r)*fγθθ(M,r)*sqrt(fγrr(M,r))
f∂ᵣlnᾶ(M,r)       = f∂ᵣᾶ(M,r)/fᾶ(M,r) 
f∂ᵣ2lnᾶ(M,r)      = (f∂ᵣ2ᾶ(M,r)*fᾶ(M,r) - f∂ᵣᾶ(M,r)^2)/fᾶ(M,r)^2

f∂ᵣγrr(M,r) = ForwardDiff.derivative(r -> fγrr(M,r), r)
f∂ᵣγθθ(M,r) = ForwardDiff.derivative(r -> fγθθ(M,r), r)

fKrr(M,∂ₜγrr,r) = -(∂ₜγrr(M,r) - fβʳ(M,r)*f∂ᵣγrr(M,r) - 2*fγrr(M,r)*f∂ᵣβʳ(M,r))/(2*fα(M,r))
fKθθ(M,∂ₜγθθ,r) = -(∂ₜγθθ(M,r) - fβʳ(M,r)*f∂ᵣγθθ(M,r))/(2*fα(M,r))
#upper first index
ffrθθ(M,r) = f∂ᵣγθθ(M,r)/fγrr(M,r)/2
ffrrr(M,r) = (f∂ᵣγrr(M,r)/fγrr(M,r) + 8*fγrr(M,r)*ffrθθ(M,r)/fγθθ(M,r))/2

# ffrθθ(M,r) = f∂ᵣγθθ(M,r)/2
# ffrrr(M,r) = (f∂ᵣγrr(M,r) + 8*fγrr(M,r)*ffrθθ(M,r)/fγθθ(M,r))/2

f∂ᵣKrr(M,∂ₜγrr,r)   = ForwardDiff.derivative(r -> fKrr(M,∂ₜγrr,r), r)
f∂ᵣfrrr(M,r)       = ForwardDiff.derivative(r -> ffrrr(M,r), r)
f∂ᵣKθθ(M,∂ₜγθθ,r)   = ForwardDiff.derivative(r -> fKθθ(M,∂ₜγθθ,r), r)
f∂ᵣfrθθ(M,r)       = ForwardDiff.derivative(r -> ffrθθ(M,r), r)

#f∂ᵣ𝜙(M,r)          = ForwardDiff.derivative(r -> f𝜙(M,r), r)
f∂ᵣ𝜙(M,r)        = ForwardDiff.derivative(r -> f𝜙(M,r), r)

# Define the initial conditions of the scalar field at t=0
fψr(M,r) = f∂ᵣ𝜙(M,r)/fγrr(M,r)
fΠ(M,r) = -(f∂ₜ𝜙(M,r) - fβʳ(M,r)*f∂ᵣ𝜙(M,r) )/fα(M,r)


#Boundary Injection Model (Kerr-Schild only)

function initial_conditions!(state::VarContainer{T}, param) where T

    ############################################
    # Specifies the Initial Conditions
    # and solves the constraints at the initial
    # time slice.
    ############################################

    # Unpack the parameters
    gauge = param.gauge
    grid = param.grid
    rmin = grid.domain.rmin
    rmax = grid.domain.rmax
    rspan = (rmin,rmax)

    # Give names to individual variables
    γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,ψr,Π,Ms = state.x
    ᾶ,βʳ,∂ᵣᾶ,∂ᵣβʳ,∂ᵣ2ᾶ,∂ᵣ2βʳ,α,∂ᵣlnᾶ,∂ᵣ2lnᾶ = gauge.x

    Ma,Mb = Ms

    fρ(M,r) = ( fΠ(M,r)^2 + fγrr(M,r)*fψr(M,r)^2 + m^2*f𝜙(M,r)^2 )/2.
    #upper index S^r now
    fSr(M,r) = fψr(M,r)*fΠ(M,r)
    # fρ(M,r) = 0.
    # fSr(M,r) = 0.

    f∂rM(M,r) = 4*pi*r^2*(fρ(M,r) - fγrr(M,r)*fβʳ(M,r)*fSr(M,r)/fα(M,r))
    f∂ₜγrr(M,r) = -8*pi*r*fγrr(M,r)*fSr(M,r)/fα(M,r)
    f∂ₜγθθ(M,r) = 0.

    # Constraint Solver

    constraintSystem(M, param, r) = f∂rM(M,r)
  
    BVP = ODEProblem(constraintSystem, M0, rspan, param)
    Mass = solve(BVP, Tsit5(), abstol=1e-15, dt=dr, adaptive=false)

    Ms[1] = Mass(rmin)
    Ms[2] = Mass(rmax)
    global Mtot = Mass(rmax)
    M(r) = Mass(r)

    println("")
    println(string("Total Mass: ",round(Mtot, digits=3)))

    # M0(r̃) = M0

    # Sample the initial state vector
    sample!(γrr,    grid, r -> fγrr(M,r)        )
    sample!(γθθ,    grid, r -> fγθθ(M,r)        )
    sample!(Krr,    grid, r -> fKrr(M,f∂ₜγrr,r)  )
    sample!(Kθθ,    grid, r -> fKθθ(M,f∂ₜγθθ,r)  )
    sample!(frrr,   grid, r -> ffrrr(M,r)       )
    sample!(frθθ,   grid, r -> ffrθθ(M,r)       )
    sample!(𝜙,      grid, r -> f𝜙(M,r)          )
    sample!(ψr,     grid, r -> fψr(M,r)         )
    sample!(Π,      grid, r -> fΠ(M,r)          )

    # Sample the gauge variables
    sample!(ᾶ,      grid, r -> fᾶ(M,r)          )
    sample!(βʳ,     grid, r -> fβʳ(M,r)         )
    sample!(∂ᵣᾶ,    grid, r -> f∂ᵣᾶ(M,r)        )
    sample!(∂ᵣβʳ,   grid, r -> f∂ᵣβʳ(M,r)       )
    sample!(∂ᵣ2ᾶ,   grid, r -> f∂ᵣ2ᾶ(M,r)       )
    sample!(∂ᵣ2βʳ,  grid, r -> f∂ᵣ2βʳ(M,r)      )
    sample!(α,      grid, r -> fα(M,r)          )
    sample!(∂ᵣlnᾶ,  grid, r -> f∂ᵣlnᾶ(M,r)      )
    sample!(∂ᵣ2lnᾶ, grid, r -> f∂ᵣ2lnᾶ(M,r)     )

    # Sample initial values of the r characteristics

    global Upri1 = @part 1  Krr + frrr*sqrt(γrr)
    global Umri1 = @part 1 -Krr + frrr*sqrt(γrr)

    global Uprin = @part n  Krr + frrr*sqrt(γrr)
    global Umrin = @part n -Krr + frrr*sqrt(γrr)
    global Upθin = @part n  Kθθ + frθθ*sqrt(γrr)
    global Umθin = @part n -Kθθ + frθθ*sqrt(γrr)

    # global Upri1 = @part 1 Krr + frrr/sqrt(γrr)
    # global Umri1 = @part 1 Krr - frrr/sqrt(γrr)

    # global Uprin = @part n Krr + frrr/sqrt(γrr)
    # global Umrin = @part n Krr - frrr/sqrt(γrr)
    # global Upθin = @part n Kθθ + frθθ/sqrt(γrr)
    # global Umθin = @part n Kθθ - frθθ/sqrt(γrr)

    global is_AH = any(@. (Kθθ - frθθ/sqrt(γrr)) > 0)

    # add noise to initial values to assess stability with magnitude s
    s = 0*10^(-10)

    for i in 1:numvar
        for j in 1:n
            state.x[i][j] += s*rand(Uniform(-1,1))
        end
    end

end

function rhs!(dtstate::VarContainer{T},state::VarContainer{T}, param::Param{T}, t) where T

    #########################################################
    # Caculates the right hand ride of the evolved variables
    #
    # This is the main meat of the program. This function 
    # contains all of the boundary conditions, spatial 
    # derivative calculations, evolution equations, 
    # and numerical dissipation. Each time the Julia DiffEq
    # Solver moves one time step, it calls this function to 
    # calculate the new values of the evolved variables
    # several times depending on the solving algorithm. 
    #########################################################

    # In order to catch errors and still have the integrator finish
    # everything is wrapped in a try-catch block.
    try

    # Unpack the parameters
    drstate = param.drstate
    temp = param.temp
    gauge = param.gauge
    a,b = rspan

    # Give names to individual variables
    γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,ψr,Π,Ms = state.x
    ∂ᵣγrr,∂ᵣγθθ,∂ᵣKrr,∂ᵣKθθ,∂ᵣfrrr,∂ᵣfrθθ,∂ᵣ𝜙,∂ᵣψr,∂ᵣΠ = drstate.x
    ∂ₜγrr,∂ₜγθθ,∂ₜKrr,∂ₜKθθ,∂ₜfrrr,∂ₜfrθθ,∂ₜ𝜙,∂ₜψr,∂ₜΠ,∂ₜMs = dtstate.x
    ᾶ,βʳ,rootγ,∂ᵣβʳ,∂ᵣ2ᾶ,∂ᵣ2βʳ,α,∂ᵣlnᾶ,∂ᵣ2lnᾶ = gauge.x

    Ma,Mb = Ms

    ∇ᵣψr = temp.x[1]; ∇ᵣfrrr = temp.x[2]; ∇ᵣfrθθ = temp.x[3];

    # Calculate first spatial derivatives by multiplying D operator
    # about ~40% of runtime is derivatives

    Threads.@threads for i in 1:numvar mul!(drstate.x[i],D,state.x[i]) end

    @. rootγ = sqrt(γrr)*γθθ

    # Calculate 3-divergence operators
    ∇ᵣψr   .= (D*(rootγ.*ψr  ))./rootγ
    ∇ᵣfrrr .= (D*(rootγ.*frrr))./rootγ
    ∇ᵣfrθθ .= (D*(rootγ.*frθθ))./rootγ

    # ∇ʳψr   .= (D*(γθθ.*ψr./sqrt.(γrr)  ))./(sqrt.(γrr).*γθθ)
    # ∇ʳfrrr .= (D*(γθθ.*frrr./sqrt.(γrr)))./(sqrt.(γrr).*γθθ)
    # ∇ʳfrθθ .= (D*(γθθ.*frθθ./sqrt.(γrr)))./(sqrt.(γrr).*γθθ)

    # Define source terms to GR

    ρ = temp.x[4]; Sr = temp.x[5]; Tt = temp.x[6]; Srr = temp.x[7]; Sθθ = temp.x[8];

    @. ρ = ( Π^2 + γrr*ψr^2 + (m^2)*𝜙^2)/2 # Energy Density
    @. Sr = ψr*Π  # Momentum Density (upper index)
    @. Tt = Π^2 - γrr*ψr^2 - 2*(m^2)*𝜙^2  # Trace of the Stress-Energy tensor (T unavailable)
    @. Srr = γrr*( Π^2 + γrr*ψr^2 - (m^2)*𝜙^2)/2  # Radial pressure component
    @. Sθθ = γθθ*( Π^2 - γrr*ψr^2 - (m^2)*𝜙^2)/2  # Angular pressure component

    # @. ρ = ( Π^2 + ψr^2/γrr + (m^2)*𝜙^2)/2 # Energy Density
    # @. Sr = ψr*Π  # Momentum Density (upper index)
    # @. Tt = Π^2 - ψr^2/γrr - 2*(m^2)*𝜙^2  # Trace of the Stress-Energy tensor (T unavailable)
    # @. Srr = γrr*( Π^2 + ψr^2/γrr - (m^2)*𝜙^2)/2  # Radial pressure component
    # @. Sθθ = γθθ*( Π^2 - ψr^2/γrr - (m^2)*𝜙^2)/2  # Angular pressure component

    # Calculate lapse, may be different at every step.
    @. α = ᾶ*γθθ*sqrt(γrr)

    # about ~25% of runtime is the actual RHS calculations

    #####################################################################
    # Evolution Equations
    #
    # This is the full suite of evolution equations for GR in spherical 
    # symmetry in the 'Einstein-Christoffel' framework.
    #
    # Note: I have used subscript r and t where possible. I could do this 
    # in principle for tensor variable indices like γᵣᵣ for example, but 
    # annoyingly a subscript theta does not exist in unicode, so tensor 
    # indices are instead normal sized.
    #####################################################################

    @. ∂ₜγrr = βʳ*∂ᵣγrr + 2*∂ᵣβʳ*γrr - 2*α*Krr

    @. ∂ₜγθθ = βʳ*∂ᵣγθθ - 2*α*Kθθ

    @. ∂ₜKrr  = ( βʳ*∂ᵣKrr - α*∇ᵣfrrr + α*frrr^2 - 6*α*(γrr*frθθ/γθθ)^2
     - α*Krr^2/γrr + 2*α*Krr*Kθθ/γθθ - 2*α*γrr*frrr*frθθ/γθθ
     - α*frrr*∂ᵣlnᾶ - α*∂ᵣlnᾶ^2 - α*∂ᵣ2lnᾶ + 2*∂ᵣβʳ*Krr )

    @. ∂ₜKθθ  = ( βʳ*∂ᵣKθθ - α*∇ᵣfrθθ + α + α*Krr*Kθθ/γrr
     - α*frrr*frθθ + 4*α*γrr*frθθ^2/γθθ - α*frθθ*∂ᵣlnᾶ )

    @. ∂ₜfrrr = ( βʳ*∂ᵣfrrr - α*∂ᵣKrr/γrr + α*frrr*Krr/γrr
     + 12*α*frθθ*Kθθ*γrr/γθθ^2 - 10*α*frθθ*Krr/γθθ - 4*α*frrr*Kθθ/γθθ
     - α*Krr*∂ᵣlnᾶ/γrr - 4*α*Kθθ*∂ᵣlnᾶ/γθθ + ∂ᵣβʳ*frrr + ∂ᵣ2βʳ )

    @. ∂ₜfrθθ = ( βʳ*∂ᵣfrθθ - α*∂ᵣKθθ/γrr + 2*α*frθθ*Krr/γrr + 2*α*frθθ*Kθθ/γθθ
     - α*frrr*Kθθ/γrr - α*Kθθ*∂ᵣlnᾶ/γrr - ∂ᵣβʳ*frθθ )

    #  @. ∂ₜγrr = βʳ*∂ᵣγrr + 2*∂ᵣβʳ*γrr - 2*α*Krr

    #  @. ∂ₜγθθ = βʳ*∂ᵣγθθ - 2*α*Kθθ
 
    #  @. ∂ₜKrr  = ( βʳ*∂ᵣKrr - α*∇ᵣfrrr/γrr + 3*α*frrr^2/γrr^2 - 6*α*frθθ^2/γθθ^2
    #   - α*Krr^2/γrr + 2*α*Krr*Kθθ/γθθ - 10*α*frrr*frθθ/(γrr*γθθ)
    #   - α*frrr*∂ᵣlnᾶ/γrr - α*∂ᵣlnᾶ^2 - α*∂ᵣ2lnᾶ + 2*∂ᵣβʳ*Krr )
 
    #  @. ∂ₜKθθ  = ( βʳ*∂ᵣKθθ - α*∇ᵣfrθθ/γrr + α + α*Krr*Kθθ/γrr
    #   + α*frrr*frθθ/γrr^2 - 4*α*frθθ^2/(γrr*γθθ) - α*frθθ*∂ᵣlnᾶ/γrr )
 
    #  @. ∂ₜfrrr = ( βʳ*∂ᵣfrrr - α*∂ᵣKrr - α*frrr*Krr/γrr
    #   + 12*α*frθθ*Kθθ*γrr/γθθ^2 - 10*α*frθθ*Krr/γθθ - 4*α*frrr*Kθθ/γθθ
    #   - α*Krr*∂ᵣlnᾶ - 4*α*Kθθ*γrr*∂ᵣlnᾶ/γθθ + 3*∂ᵣβʳ*frrr + γrr*∂ᵣ2βʳ )
 
    #  @. ∂ₜfrθθ = ( βʳ*∂ᵣfrθθ - α*∂ᵣKθθ - α*frrr*Kθθ/γrr + 2*α*frθθ*Kθθ/γθθ
    #   - α*Kθθ*∂ᵣlnᾶ + ∂ᵣβʳ*frθθ )

    # Klein-Gordon System

    @. ∂ₜ𝜙 = βʳ*∂ᵣ𝜙 - α*Π

    @. ∂ₜψr = ( βʳ*∂ᵣψr - α*∂ᵣΠ - α*(frrr - 2*γrr*frθθ/γθθ + ∂ᵣlnᾶ)*Π/γrr 
        + ∂ᵣβʳ*ψr + 2*α*Krr*ψr/γrr )

    @. ∂ₜΠ = ( βʳ*∂ᵣΠ - α*∇ᵣψr + α*(Krr/γrr + 2*Kθθ/γθθ)*Π
    - α*(frrr - 2*γrr*frθθ/γθθ + ∂ᵣlnᾶ)*ψr + m^2*α*𝜙 )

    # @. ∂ₜ𝜙 = βʳ*∂ᵣ𝜙 - α*Π

    # @. ∂ₜψr =  βʳ*∂ᵣψr - α*∂ᵣΠ - α*(frrr/γrr - 2*frθθ/γθθ + ∂ᵣlnᾶ)*Π + ∂ᵣβʳ*ψr

    # @. ∂ₜΠ = ( βʳ*∂ᵣΠ - α*∇ᵣψr/γrr + α*(Krr/γrr + 2*Kθθ/γθθ)*Π
    # + α*(frrr/γrr - 6*frθθ/γθθ - ∂ᵣlnᾶ)*ψr/γrr + m^2*α*𝜙 )

    # Source terms to GR

    @. ∂ₜKrr  += 4*pi*α*(γrr*Tt - 2*Srr)
    @. ∂ₜKθθ  += 4*pi*α*(γθθ*Tt - 2*Sθθ)
    @. ∂ₜfrrr += 16*pi*α*γrr*Sr

    # @. ∂ₜKrr  += 4*pi*α*(γrr*Tt - 2*Srr)
    # @. ∂ₜKθθ  += 4*pi*α*(γθθ*Tt - 2*Sθθ)
    # @. ∂ₜfrrr += 16*pi*α*γrr*Sr

    # Calculates the Apparent Horizon, if there is one
    # in the domain, no inner boundary conditions are applied

    # AH = temp.x[9]
    # @. AH = Kθθ - frθθ/sqrt(γrr)
    # is_AH = false
    # for i in 1:n-1 
    #     if AH[i]*AH[i+1] <= 0. 
    #         is_AH = true
    #         break
    #     end 
    # end

    ###################################
    global is_AH = any(@. (Kθθ - frθθ*sqrt(γrr)) > 0)

    # #EM = 4*pi*dr*sum(A*(σv.*((frθθ.*ρ .- Kθθ.*Sr).*sqrt.(γθθ))))

    # # Define characteristic variables at r=a
    Umθ = @part 1 ( -Kθθ + sqrt(γrr)*frθθ )
    Upθ = @part 1 (  Kθθ + sqrt(γrr)*frθθ )

    Umr = @part 1 ( -Krr + sqrt(γrr)*frrr )
    Upr = @part 1 (  Krr + sqrt(γrr)*frrr )

    Um𝜙 = @part 1 ( -Π + sqrt(γrr)*ψr )
    Up𝜙 = @part 1 (  Π + sqrt(γrr)*ψr )

    cm = @part 1 -βʳ - ᾶ*γθθ
    cp = @part 1 -βʳ + ᾶ*γθθ
    # Umθ = @part 1 ( Kθθ - frθθ/sqrt(γrr) )
    # Upθ = @part 1 ( Kθθ + frθθ/sqrt(γrr) )

    # Umr = @part 1 ( Krr - frrr/sqrt(γrr) )
    # Upr = @part 1 ( Krr + frrr/sqrt(γrr) )

    # Up𝜙 = @part 1 ( Π + ψr/sqrt(γrr) )
    # Um𝜙 = @part 1 ( Π - ψr/sqrt(γrr) )

    # cp = @part 1 -βʳ + ᾶ*γθθ
    # cm = @part 1 -βʳ - ᾶ*γθθ

    # BC on angular characteristics
    Upθb = @part 1 ((γθθ - 2*M0*sqrt(γθθ))/Umθ)
    #Upθb = @part 1 ((2*Ma*sqrt(γθθ) - γθθ)/Umθ)

    # if !(is_AH)
    #     # BC on scalar characteristics
    #     Up𝜙b = @part 1 ka*sqrt((cm*Upθb)/(cp*Umθ))*Um𝜙
    #     # Mass evolution at r=a
    #     ∂ₜMs[1] = @part 1 -pi*cm*sqrt(γrr*γθθ)*(1-ka^2)*Upθb*Um𝜙^2
    # else 
    #     # BC on scalar characteristics
    #     Up𝜙b = 0
    #     # Mass evolution at r=a
    #     #∂ₜMs[1] = @part 1 -pi*cm*sqrt(γrr*γθθ)*Upθb*Um𝜙^2
    #     ∂ₜMs[1] = @part 1 -pi*cm*sqrt(γrr*γθθ)*Upθb*Um𝜙^2
    # end

    # BC on scalar characteristics
    Up𝜙b = @part 1 ka*sqrt(-(cm*Upθb)/(cp*Umθ))*Um𝜙
    # Mass evolution at r=a
    ∂ₜMs[1] = @part 1 -pi*cm*sqrt(γrr*γθθ)*(1-ka^2)*Upθb*Um𝜙^2

    # BC on radial characteristics
    # Uprb = Upri
    # Krr = Krri
    Uprb = Umr - Umri1 + Upri1
    # # BC on angular characteristics
    # Upθb = @part 1 ((2*Ma*sqrt(γθθ) - γθθ)/Umθ)

    # # if !(is_AH)
    # #     # BC on scalar characteristics
    # #     Up𝜙b = @part 1 ka*sqrt((cm*Upθb)/(cp*Umθ))*Um𝜙
    # #     # Mass evolution at r=a
    # #     ∂ₜMs[1] = @part 1 -pi*cm*sqrt(γrr*γθθ)*(1-ka^2)*Upθb*Um𝜙^2
    # # else 
    # #     # BC on scalar characteristics
    # #     Up𝜙b = 0
    # #     # Mass evolution at r=a
    # #     #∂ₜMs[1] = @part 1 -pi*cm*sqrt(γrr*γθθ)*Upθb*Um𝜙^2
    # #     ∂ₜMs[1] = @part 1 -pi*cm*sqrt(γrr*γθθ)*Upθb*Um𝜙^2
    # # end

    # # BC on scalar characteristics
    # Up𝜙b = @part 1 ka*sqrt((cm*Upθb)/(cp*Umθ))*Um𝜙
    # # Mass evolution at r=a
    # ∂ₜMs[1] = @part 1 -pi*cm*sqrt(γrr*γθθ)*(1-ka^2)*Upθb*Um𝜙^2

    # # BC on radial characteristics
    # # Uprb = Upri
    # # Krr = Krri
    # Uprb = -(Umr - Umri1) + Upri1

    if cp > 0

        ## Applies Inner Boundary Conditions

        #SAT strength
        s1 = abs(cp)/Σ[1,1]

        # SAT boundary application
        ∂ₜΠ[1]    += s1*(Up𝜙b - Up𝜙)/2
        ∂ₜψr[1]   += s1*(Up𝜙b - Up𝜙)/sqrt(γrr[1])/2

        ∂ₜKrr[1]  += s1*(Uprb - Upr)/2
        ∂ₜfrrr[1] += s1*(Uprb - Upr)/sqrt(γrr[1])/2

        ∂ₜKθθ[1]  += s1*(Upθb - Upθ)/2
        ∂ₜfrθθ[1] += s1*(Upθb - Upθ)/sqrt(γrr[1])/2

    end
    # if cp > 0

    #     ## Applies Inner Boundary Conditions

    #     #SAT strength
    #     s1 = abs(cp)/Σ[1,1]

    #     # SAT boundary application
    #     ∂ₜΠ[1]    += s1*(Up𝜙b - Up𝜙)/2
    #     ∂ₜψr[1]   += s1*sqrt(γrr[1])*(Up𝜙b - Up𝜙)/2

    #     ∂ₜKrr[1]  += s1*(Uprb - Upr)/2
    #     ∂ₜfrrr[1] += s1*sqrt(γrr[1])*(Uprb - Upr)/2

    #     ∂ₜKθθ[1]  += s1*(Upθb - Upθ)/2
    #     ∂ₜfrθθ[1] += s1*sqrt(γrr[1])*(Upθb - Upθ)/2

    # end
    # ## Outer Boundary Conditions

    # # Define characteristic variables at r=b
    Umθ = @part n ( -Kθθ + sqrt(γrr)*frθθ )
    Upθ = @part n (  Kθθ + sqrt(γrr)*frθθ )

    Umr = @part n ( -Krr + sqrt(γrr)*frrr )
    Upr = @part n (  Krr + sqrt(γrr)*frrr )

    Um𝜙 = @part n ( -Π + sqrt(γrr)*ψr )
    Up𝜙 = @part n (  Π + sqrt(γrr)*ψr )

    cm = @part n -βʳ - ᾶ*γθθ
    cp = @part n -βʳ + ᾶ*γθθ

    # # Define characteristic variables now at r=b
    # Umθ = @part n ( Kθθ - frθθ/sqrt(γrr) )
    # Upθ = @part n ( Kθθ + frθθ/sqrt(γrr) )

    # Umr = @part n ( Krr - frrr/sqrt(γrr) )
    # Upr = @part n ( Krr + frrr/sqrt(γrr) )

    # Up𝜙 = @part n ( Π + ψr/sqrt(γrr) )
    # Um𝜙 = @part n ( Π - ψr/sqrt(γrr) )

    # cp = @part n -βʳ + ᾶ*γθθ
    # cm = @part n -βʳ - ᾶ*γθθ

    # # BC on angular characteristics
    Umθb = @part n ((γθθ - 2*Mtot*sqrt(γθθ))/Upθ)
    # Umθb = @part n ((2*Mb*sqrt(γθθ) - γθθ)/Upθ)

    # #Boundary injection model
    # # cm = @part n -βʳ - ᾶ*γθθ
    # # if (r0-σr)<(b-cm*t)<(r0+σr) 𝜙[n] = (Amp/b)*((b-cm*t)-(r0-σr))^4*((b-cm*t)-(r0+σr))^4/σr^8 end

    # # if (r0-σr)<(b-cm*t)<(r0+σr) 
    # #     ∂ₜcm    = @part n -ᾶ*∂ₜγθθ
    # #     ∂ₜ𝜙b    = -(8*Amp*(cm+t*∂ₜcm)/b)*(((b-cm*t)-r0)^2-σr^2)^3*((b-cm*t)-r0)/σr^8  
    # #     #∂ᵣ𝜙b   = (Amp/b^2)*(((b-cm*t)-r0)^2-σr^2)^3*((b-cm*t-r0)*(7*b+cm*t+r0)+σr^2)/σr^8 
    # #     ∂ᵣ𝜙b   = @part n ψr
    # #     Um𝜙b   = @part n -(∂ₜ𝜙b-βʳ*∂ᵣ𝜙b)/α - ∂ᵣ𝜙b/sqrt(γrr)
    # #     ∂ₜMs[2] = @part n -pi*cm*sqrt(γrr*γθθ)*Upθ*Um𝜙b^2
    # # else
    # #     Um𝜙b   = 0
    # #     ∂ₜMs[2] = @part n pi*cp*sqrt(γrr*γθθ)*(1-kb^2)*Umθb*Up𝜙^2
    # # end

    # #Um𝜙 = @part n ( -(∂ₜ𝜙-βʳ*ψr)/α - ψr/sqrt(γrr) )

    # # BC on scalar characteristics
    Um𝜙b = @part n kb*sqrt(-(cp*Umθb)/(cm*Upθ))*Up𝜙

    # # BC on radial characteristics
    Umrb = Upr + Umrin - Uprin
    #Umrb = -(Upr - Uprin) + Umrin

    ∂ₜMs[2] = @part n -pi*cp*sqrt(γrr*γθθ)*(1-kb^2)*Umθb*Up𝜙^2

    # # Mass evolution at r=b
    # #∂ₜMs[2] = @part n pi*cp*sqrt(γrr*γθθ)*(1-kb^2)*Umθb*Up𝜙^2
    # #∂ₜMs[2] = @part 1 -pi*cm*sqrt(γrr*γθθ)*(1-ka^2)*Upθb*Um𝜙^2

    # Application of constraint BCs
    @part n ∂ₜγrr = ( γrr*(2*frrr - 8*frθθ*γrr/γθθ)*βʳ + 2*∂ᵣβʳ*γrr - 2*α*Krr )
    @part n ∂ₜγθθ = ( 2*γrr*frθθ*βʳ - 2*α*Kθθ )
    @part n ∂ₜ𝜙   = (γrr*βʳ*ψr - α*Π)

    #SAT strength
    sn = abs(cm)/Σ[n,n]

    # SAT boundary application
    ∂ₜΠ[n]    += -sn*(Um𝜙b - Um𝜙)/2
    ∂ₜψr[n]   +=  sn*(Um𝜙b - Um𝜙)/sqrt(γrr[n])/2

    ∂ₜKrr[n]  += -sn*(Umrb - Umr)/2
    ∂ₜfrrr[n] +=  sn*(Umrb - Umr)/sqrt(γrr[n])/2

    ∂ₜKθθ[n]  += -sn*(Umθb - Umθ)/2
    ∂ₜfrθθ[n] +=  sn*(Umθb - Umθ)/sqrt(γrr[n])/2


    # # Application of constraint BCs
    # @part n ∂ₜγrr = ( (2*frrr - 8*frθθ*γrr/γθθ)*βʳ + 2*∂ᵣβʳ*γrr - 2*α*Krr )
    # @part n ∂ₜγθθ = ( 2*frθθ*βʳ - 2*α*Kθθ )
    # @part n ∂ₜ𝜙   = (βʳ*ψr - α*Π)

    # #SAT strength
    # sn = abs(cm)/Σ[n,n]

    # # SAT boundary application
    # ∂ₜΠ[n]    += sn*(Um𝜙b - Um𝜙)/2
    # ∂ₜψr[n]   += -sn*sqrt(γrr[n])*(Um𝜙b - Um𝜙)/2

    # ∂ₜKrr[n]  += sn*(Umrb - Umr)/2
    # ∂ₜfrrr[n] += -sn*sqrt(γrr[n])*(Umrb - Umr)/2

    # ∂ₜKθθ[n]  += sn*(Umθb - Umθ)/2
    # ∂ₜfrθθ[n] += -sn*sqrt(γrr[n])*(Umθb - Umθ)/2

    # Add the numerical dissipation to dtstate
    # about ~30% of runtime is dissipation

    Threads.@threads for i in 1:numvar mul!(dtstate.x[i],A6,state.x[i],1,1) end
    # this syntax is equivalent to dtstate.x[i] .+= D4*state.x[i]

    # catch any errors, save them to print later
    catch e
        global_error.error = e
        global_error.stacktrace = stacktrace(catch_backtrace())
    end

end

function constraints(state::VarContainer{T},param) where T

    ##########################################################
    # Caculates the constraints of the system and the energies
    #
    # These are outputed at every saved time step. They should
    # limit to zero as the spatial resolution is increased. 
    # They are important to monitor to make sure the physics 
    # of the system is being properly modeled.
    ##########################################################

    # Unpack Variables
    drstate = param.drstate
    gauge = param.gauge
    temp = param.temp

    # Give usual names
    γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,ψr,Π = state.x
    ∂ᵣγrr,∂ᵣγθθ,∂ᵣKrr,∂ᵣKθθ,∂ᵣfrrr,∂ᵣfrθθ,∂ᵣ𝜙,∂ᵣψr,∂ᵣΠ = drstate.x
    ᾶ,βʳ,∂ᵣᾶ,∂ᵣβʳ,∂ᵣ2ᾶ,∂ᵣ2βʳ,α,∂ᵣlnᾶ,∂ᵣ2lnᾶ = gauge.x

    # Calculate derivatives
    for i in 1:numvar
        mul!(drstate.x[i],D,state.x[i])
    end

    # Calculate source terms
    ρ = temp.x[1]; Sr = temp.x[2]

    @. ρ = (Π^2 + γrr*ψr^2 + (m^2)*𝜙^2)/2.
    @. Sr = ψr*Π

    # Constraint Equations

    C = zeros(T,n); Cr = zeros(T,n); Crrr = zeros(T,n); Crθθ = zeros(T,n);
    C𝜙 = zeros(T,n);

    @. C = (∂ᵣfrθθ/γθθ - 9*γrr*frθθ^2/(2*γθθ^2) + frrr*frθθ/γθθ
     - Kθθ^2/(2*γθθ^2) - 1/(2*γθθ) - Krr*Kθθ/(γrr*γθθ) + 4*pi*ρ)

    @. Cr = (∂ᵣKθθ/γθθ/γrr - frθθ*Kθθ/γθθ^2 - frθθ*Krr/(γθθ*γrr) + 4*pi*Sr)

    @. Crrr = ∂ᵣγrr + 8*frθθ*γrr^2/γθθ - 2*γrr*frrr

    @. Crθθ = ∂ᵣγθθ - 2*γrr*frθθ

    @. C𝜙 = ∂ᵣ𝜙 - γrr*ψr

#     @. C = (∂ᵣfrθθ/(γθθ*γrr) + 7*frθθ^2/(2*γrr*γθθ^2) - frrr*frθθ/(γrr^2*γθθ)
#     - Kθθ^2/(2*γθθ^2) - 1/(2*γθθ) - Krr*Kθθ/(γrr*γθθ) + 4*pi*ρ)

#    @. Cr = (∂ᵣKθθ/γθθ - frθθ*Kθθ/γθθ^2 - frθθ*Krr/(γθθ*γrr) + 4*pi*Sr)

#    @. Crrr = ∂ᵣγrr + 8*frθθ*γrr/γθθ - 2*frrr

#    @. Crθθ = ∂ᵣγθθ - 2*frθθ

#    @. C𝜙 = ∂ᵣ𝜙 - ψr

    Γ = spdiagm(sqrt.(γrr).*γθθ)
    W = Σ*Γ;

    # Calculate Scalar energy
    E = (frθθ.*sqrt.(γθθ))'*Σ*ρ - (Kθθ.*sqrt.(γθθ))'*Σ*Sr

    # Calculate Constraint 'energy'
    Ec = sqrt(C'*W*C +  Cr'*W*(Cr./γrr))

    return [C, Cr, Crrr, Crθθ, C𝜙, E, Ec]

end

function solution_saver(T,sol,param)

    ###############################################
    # Saves all of the variables in nice HDF5 files 
    # in the choosen data folder directory
    ###############################################

    folder = string("n=",      n,
                    "_rspan=", round.(rspan, digits=2),
                    "_tspan=", round.(tspan, digits=2),
                    "_CFL=",   round(CFL, digits=2),
                    "_Mtot=",  round(Mtot, digits=2)
                    )

    path = string("data/",folder)

    mkpath(path);

    old_files = readdir(path; join=true)
    for i in 1:length(old_files)
        rm(old_files[i])
    end

    # Windows apparently doesn't support non-ASCII characters
    # in HDF5 file names, or at least not when imported with Mathematica...
    
    vars = (["grr","gtt","Krr","Ktt","frrr","frtt","phi","psi","Pi",
    #"∂ₜγrr","∂ₜγθθ","∂ₜKrr","∂ₜKθθ","∂ₜfrrr","∂ₜfrθθ","∂ₜ𝜙","∂ₜψr","∂ₜΠ",
    "C","Cr","Crrr","Crtt","Cphi","E","Ec"])

    #varlen = length(vars)
    tlen = size(sol)[2]
    grid = param.grid

    #dtstate = [rhs_all(sol[i],param,0.) for i = 1:tlen]

    # Calculate state, constraints, and coordinates
    cons = [constraints(sol[i],param) for i = 1:tlen]

    array = Array{T,2}(undef,tlen,n)

    r = zeros(T,n); sample!(r, grid, r -> r );

    # save coordinates
    save(string(path,"/coords.h5"), Dict("r"=>r,"t"=>sol.t[:]) )

    # save state and constraints
    for j = 1:numvar
        for i = 1:tlen @. array[i,:] = sol[i].x[j] end
        save(string(path,"/",vars[j],".h5"), Dict(vars[j]=>array ) )
    end

    for j = 1:5
        for i = 1:tlen @. array[i,:] = cons[i][j] end
        save(string(path,"/",vars[j+numvar],".h5"), Dict(vars[j+numvar]=>array ) )
    end

    for j = 6:7
        for i = 1:tlen array[i,1] = cons[i][j] end
        save(string(path,"/",vars[j+numvar],".h5"), Dict(vars[j+numvar]=>array[:,1] ) )
    end

    println("")
    println("Saved at ", path)

end

# Method of error handling such that if an error is encountered
# during evaluation, the state is still saved up until that point
# and the error is printed with a simplified stack trace

struct NothingException <: Exception end

mutable struct ErrorContainer
    error::Exception
    stacktrace::AbstractArray
end

# Create variable to store error, initially a NothingException
global_error = ErrorContainer(NothingException(),[1.])

# Returns true when global_error is changed to some thown error
function error_handler(state,t,integrator)
    return !(typeof(global_error.error) == NothingException)
end

AH(state,t,integrator) = is_AH

# Terminates the integrator when error_handler returns true
cbe = DiscreteCallback(error_handler,terminate!,save_positions=(false,false))

cbah = DiscreteCallback(AH,terminate!,save_positions=(false,false))

cb = CallbackSet(cbe,cbah)

function main()

    #########################################################
    # Main Program
    #
    # Calls each of the above functions to run a simulation. 
    # Sets up the numerical grid, sets the gauge conditions, 
    # sets the initial conditions, and finally runs the 
    # numerical DiffEq package to run the time integration.
    #
    # All data is saved in the folder specified to the 
    # solution_saver, each in their own HDF5 file.
    ########################################################

    rmin, rmax = rspan

    domain = Domain{T}(rmin, rmax)
    grid = Grid(domain, n)

    #atol = eps(T)^(T(3) / 4)

    # Set time-step algorithm
    alg = RK4()
    #alg = Vern6()

    #printlogo()

    # Allocate state, derivatives, and other params
    state = similar(ArrayPartition,T,n); drstate = similar(ArrayPartition,T,n);
    gauge = similar(ArrayPartition,T,n); temp = similar(ArrayPartition,T,n);

    # Allocate the parameter struct
    param = Param(grid,gauge,drstate,temp)

    # Initialize the problem, fill state with initial conditions
    initial_conditions!(state, param)

    #return 

    # Define the ODE problem
    prob = ODEProblem(rhs!, state, tspan, param)

    # dtstate = similar(ArrayPartition,T,n);
    # rhs!(dtstate,state, param, 0.)

    # for i in 1:numvar
    #     println(maximum(dtstate.x[i]))
    # end

    #return 

    #return @benchmark rhs!($dtstate,$state, $param, 0.)

    # Print helpful information to the screen
    println("")
    println("                            Max of time derivatives                            ")
    println(".-------.-----------.-----------.-----------.-----------.-----------.-----------.")
    println("| Time  |   ∂ₜγrr   |   ∂ₜγθθ   |   ∂ₜKrr   |   ∂ₜKθθ   |   ∂ₜfrrr  |   ∂ₜfrθθ  |")
    println(":-------+-----------+-----------+-----------+-----------+-----------+-----------:")

    # Define the ODE integrator
    integrator = init(prob, alg; dt = dt, adaptive = false, 
        saveat = save_interval, alias_u0 = true, callback = cb)

    # sets the times which we will print info to the screen
    tstops = [t for t in tspan[1]:print_interval:tspan[2]][2:end]

    # Runs the problem, saving the time it takes and printing 
    # the max time derivative values to the terminal
    el=@elapsed for (u,t) in TimeChoiceIterator(integrator,tstops) 
        print("| ", rpad(string(round(t,digits=1)),6," "),"|   ")
        for i in 1:6
            # Calculates max of 1st order time derivatives to print
            dudt = maximum((u.x[i] .- integrator.uprev.x[i])/(t - integrator.tprev))
            print(rpad(string(round(abs(dudt), digits=3)),8," "),"|   ")
        end
        println("")
        if !(typeof(global_error.error) == NothingException) || is_AH break end
    end

    println("'-------'-----------'-----------'-----------'-----------'-----------'-----------'")
    println("")

    if is_AH println("Apparent Horizon formed") end

    x, s = divrem(el, 60); h, m = divrem(x, 60);

    # Prints elapsed time
    println("Elapsed Time: ", round(h, digits=2)," h, ", round(m, digits=2)," m, ", round(s, digits=2)," s. ")

    # Saves the solution
    solution_saver(T,integrator.sol,param)

    # Print error if one is encountered, with line of occurance
    if !(typeof(global_error.error) == NothingException)

        iter = 1
        for i in 1:length(global_error.stacktrace)
            if global_error.stacktrace[i].file == Symbol(@__FILE__)
                iter = i
                break
            end
        end

        println(" ")
        println("Exited with error on line ", global_error.stacktrace[iter].line)
        println(" ")
        printstyled(stderr,"ERROR: ", bold=true, color=:red)
        printstyled(stderr,sprint(showerror,global_error.error))
        println(stderr)

    end

    return

end



end