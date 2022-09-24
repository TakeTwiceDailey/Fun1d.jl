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
VarContainer{T} = ArrayPartition{T, NTuple{numvar, Vector{T}}}

struct Domain{S}
    xmin::S
    xmax::S
end

struct Grid{S}
    domain::Domain{S}
    ncells::Int
end

# Main parameter struct passed to ODE Solver
struct Param{T}
    r̃min::T
    r̃max::T
    grid::Grid{T}
    r::Function
    drdr̃::Function
    d2rdr̃::Function
    rsamp::Vector{T}
    drdr̃samp::Vector{T}
    d2rdr̃samp::Vector{T}
    gauge::VarContainer{T}
    speeds::VarContainer{T}
    init_state::VarContainer{T}
    init_drstate::VarContainer{T}
    state::VarContainer{T}
    drstate::VarContainer{T}
    dtstate::VarContainer{T}
    dissipation::VarContainer{T}
    temp::VarContainer{T}
    Dr::SparseMatrixCSC{T,Int64}
end

# Defines how to allocate the grid functions
@inline function Base.similar(::Type{ArrayPartition},::Type{T},size::Int) where T
    return ArrayPartition([similar(Vector{T}(undef,size)) for i=1:numvar]...)::VarContainer{T}
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

    r̃min = grid.domain.xmin

    f .= T[fun(r̃min + dr̃*(j-1)) for j in 1:(grid.ncells)]

end

#############################
#   Define coordinate systems
#############################

# All metric variables that explictly depend on M
# need a function that overloads the M argument to a ::Number
# so the constraint solver can work properly.
#
# To define a new coordinate system, all you need is to specify
# the set (ᾶ, βʳ, γrr, γθθ) as functions of (M,r,r̃),
# All of the other variables are fixed automatically based on definitions

#Kerr-Schild Coordinates
# sign=1 for ingoing (black hole), sign=-1 for outgoing (white hole)

sign = 1.

fᾶ(M,r,r̃) = 1/(r(r̃)^2 + 2*M(r̃)*r(r̃))
fβʳ(M,r,r̃) = sign*2*M(r̃)/(2*M(r̃)+r(r̃))
fγrr(M,r,r̃) = 1 + 2*M(r̃)/r(r̃)
fγθθ(M,r,r̃) = r(r̃)^2

fᾶ(M::Number,r,r̃) = 1/(r(r̃)^2+2*M*r(r̃))
fβʳ(M::Number,r,r̃) = sign*2*M/(2*M+r(r̃))
fγrr(M::Number,r,r̃) = 1 + 2*M/r(r̃)

### NOTE: other coordinate systems require
#   careful consideration of reg_list in inputfile.jl

#Painleve-Gullstrand Coordinates

# fᾶ(M,r,r̃) = 1.
# fβʳ(M,r,r̃) = sqrt(2*M(r̃)/r(r̃))
# fγrr(M,r,r̃) = 1.
# fγθθ(M,r,r̃) = r(r̃)^2
#
# fβʳ(M::Number,r,r̃) = sqrt(2*M/r(r̃))

# Schwarzschild
#
# fᾶ(M,r,r̃) = sqrt(1. - 2*M(r̃)/r(r̃))
# fβʳ(M,r,r̃) = 0.
# fγrr(M,r,r̃) = 1/(1 - 2*M(r̃)/r(r̃))
# fγθθ(M,r,r̃) = r(r̃)^2
#
# fᾶ(M::Number,r,r̃) = sqrt(1. - 2*M/r(r̃))
# fγrr(M::Number,r,r̃) = 1/(1 - 2*M/r(r̃))

# Spherical Minkowski

# fᾶ(M,r,r̃) = 1.
# fβʳ(M,r,r̃) = 0.
# fγrr(M,r,r̃) = 1.
# fγθθ(M,r,r̃) = r(r̃)^2

# Define derivatives, extrinsic curavture, and the f_{ijk} variables

fα(M,r,r̃) = fᾶ(M,r,r̃)*fγθθ(M,r,r̃)*sqrt(fγrr(M,r,r̃))

fcp(M,r,r̃) = -fβʳ(M,r,r̃) + fα(M,r,r̃)/sqrt(fγrr(M,r,r̃))
fcm(M,r,r̃) = -fβʳ(M,r,r̃) - fα(M,r,r̃)/sqrt(fγrr(M,r,r̃))

f∂r̃ᾶ(M,r,r̃)         = ForwardDiff.derivative(r̃ -> fᾶ(M,r,r̃), r̃)
f∂r̃2ᾶ(M,r,r̃)        = ForwardDiff.derivative(r̃ -> f∂r̃ᾶ(M,r,r̃), r̃)
f∂r̃βʳ(M,r,r̃)        = ForwardDiff.derivative(r̃ -> fβʳ(M,r,r̃), r̃)
f∂r̃2βʳ(M,r,r̃)       = ForwardDiff.derivative(r̃ -> f∂r̃βʳ(M,r,r̃), r̃)
f∂r̃cp(M,r,r̃)        = ForwardDiff.derivative(r̃ -> fcp(M,r,r̃), r̃)
f∂r̃2cp(M,r,r̃)       = ForwardDiff.derivative(r̃ -> f∂r̃cp(M,r,r̃), r̃)
f∂r̃cm(M,r,r̃)        = ForwardDiff.derivative(r̃ -> fcm(M,r,r̃), r̃)
f∂r̃2cm(M,r,r̃)       = ForwardDiff.derivative(r̃ -> f∂r̃cm(M,r,r̃), r̃)

f∂ᵣβʳ(M,r,r̃)  = ForwardDiff.derivative(r̃ -> fβʳ(M,r,r̃),  r̃)/drdr̃(r̃)
f∂ᵣγrr(M,r,r̃) = ForwardDiff.derivative(r̃ -> fγrr(M,r,r̃), r̃)/drdr̃(r̃)
f∂ᵣγθθ(M,r,r̃) = ForwardDiff.derivative(r̃ -> fγθθ(M,r,r̃), r̃)/drdr̃(r̃)

fKrr(M,∂ₜγrr,r,r̃) = -(∂ₜγrr(M,r,r̃) - fβʳ(M,r,r̃)*f∂ᵣγrr(M,r,r̃) - 2*fγrr(M,r,r̃)*f∂ᵣβʳ(M,r,r̃))/(2*fα(M,r,r̃))
fKθθ(M,∂ₜγθθ,r,r̃) = -(∂ₜγθθ(M,r,r̃) - fβʳ(M,r,r̃)*f∂ᵣγθθ(M,r,r̃))/(2*fα(M,r,r̃))
ffrθθ(M,r,r̃) = f∂ᵣγθθ(M,r,r̃)/2
ffrrr(M,r,r̃) = (f∂ᵣγrr(M,r,r̃) + 8*fγrr(M,r,r̃)*ffrθθ(M,r,r̃)/fγθθ(M,r,r̃))/2

f∂ᵣKrr(M,∂ₜγrr,r,r̃)   = ForwardDiff.derivative(r̃ -> fKrr(M,∂ₜγrr,r,r̃), r̃)/drdr̃(r̃)
f∂ᵣfrrr(M,r,r̃)       = ForwardDiff.derivative(r̃ -> ffrrr(M,r,r̃), r̃)/drdr̃(r̃)
f∂ᵣKθθ(M,∂ₜγθθ,r,r̃)   = ForwardDiff.derivative(r̃ -> fKθθ(M,∂ₜγθθ,r,r̃), r̃)/drdr̃(r̃)
f∂ᵣfrθθ(M,r,r̃)       = ForwardDiff.derivative(r̃ -> ffrθθ(M,r,r̃), r̃)/drdr̃(r̃)

f∂ᵣ𝜙(M,r,r̃)         = ForwardDiff.derivative(r̃ -> f𝜙(M,r,r̃), r̃)/drdr̃(r̃)

fψ(M,r,r̃) = f∂ᵣ𝜙(M,r,r̃)
fΠ(M,r,r̃) = -(f∂ₜ𝜙(M,r,r̃) - fβʳ(M,r,r̃)*fψ(M,r,r̃) )/fα(M,r,r̃)

function init!(state::VarContainer{T}, param) where T

    ############################################
    # Specifies the Initial Conditions
    ############################################

    init_state = param.init_state
    init_drstate = param.init_drstate
    gauge = param.gauge
    speeds = param.speeds

    γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,ψ,Π = state.x
    ᾶ,βʳ,∂ᵣᾶ,∂ᵣβʳ,∂ᵣ2ᾶ,∂ᵣ2βʳ,∂ᵣ3βʳ,∂ᵣ4βʳ,∂ᵣ5βʳ = gauge.x
    cp,cm,∂ᵣcp,∂ᵣcm,∂ᵣ2cp,∂ᵣ2cm,∂ᵣ3cp,∂ᵣ4cp,∂ᵣ5cp = speeds.x
    γrri,γθθi,Krri,Kθθi,frrri,frθθi,𝜙i,ψi,Πi = init_state.x
    ∂ᵣγrr,∂ᵣγθθ,∂ᵣKrr,∂ᵣKθθ,∂ᵣfrrr,∂ᵣfrθθ,∂ᵣ𝜙,∂ᵣψ,∂ᵣΠ = init_drstate.x

    grid = param.grid
    # dr̃ = spacing(grid)
    r = param.r
    drdr̃ = param.drdr̃
    d2rdr̃ = param.d2rdr̃
    r̃min = param.r̃min
    r̃max = param.r̃max

    n = grid.ncells
    r̃span = (r̃min,r̃max)

    # fρ(M,r,r̃) = ( fΠ(M,r,r̃)^2 + fψ(M,r,r̃)^2/fγrr(M,r,r̃) + m^2*f𝜙(M,r,r̃)^2 )/2.
    # fSr(M,r,r̃) = fψ(M,r,r̃)*fΠ(M,r,r̃)
    fρ(M,r,r̃) = 0.
    fSr(M,r,r̃) = 0.

    f∂r̃M(M,r,r̃) = 4*pi*r(r̃)^2*(fρ(M,r,r̃) - fβʳ(M,r,r̃)*fSr(M,r,r̃)/fα(M,r,r̃))*drdr̃(r̃)
    f∂ₜγrr(M,r,r̃) = -8*pi*r(r̃)*fSr(M,r,r̃)/fα(M,r,r̃)
    f∂ₜγθθ(M,r,r̃) = 0.

    f∂ₜγrri(M,r,r̃) = 0.
    f∂ₜγθθi(M,r,r̃) = 0.

    # Constraint Equations

    function constraintSystem(M, param, r̃)
        r = param.r
        f∂r̃M(M,r,r̃)
    end

    BVP = ODEProblem(constraintSystem, M0, r̃span, param)
    Mass = solve(BVP, Tsit5(), abstol=1e-15, dt=dr̃, adaptive=false)

    global Mtot = Mass(r̃max)
    M(r̃) = Mass(r̃)

    println("")
    println(string("Total Mass: ",round(Mtot, digits=3)))

    # M0(r̃) = M0

    # Sample the 'regular' values and derivatives,
    # which are used in the regularization process
    sample!(γrri,   grid, r̃ -> fγrr(M0,r,r̃)                 )
    sample!(γθθi,   grid, r̃ -> fγθθ(M0,r,r̃)                 )
    sample!(Krri,   grid, r̃ -> fKrr(M0,f∂ₜγrri,r,r̃)          )
    sample!(Kθθi,   grid, r̃ -> fKθθ(M0,f∂ₜγθθi,r,r̃)          )
    sample!(frrri,  grid, r̃ -> ffrrr(M0,r,r̃)                )
    sample!(frθθi,  grid, r̃ -> ffrθθ(M0,r,r̃)                )
    sample!(𝜙i,     grid, r̃ -> f𝜙(M0,r,r̃)                   )
    sample!(ψi,     grid, r̃ -> fψ(M0,r,r̃)                   )
    sample!(Πi,     grid, r̃ -> fΠ(M0,r,r̃)                   )

    sample!(∂ᵣγrr,  grid, r̃ -> f∂ᵣγrr(M0,r,r̃)               )
    sample!(∂ᵣγθθ,  grid, r̃ -> f∂ᵣγθθ(M0,r,r̃)               )
    sample!(∂ᵣKrr,  grid, r̃ -> f∂ᵣKrr(M0,f∂ₜγrri,r,r̃)        )
    sample!(∂ᵣKθθ,  grid, r̃ -> f∂ᵣKθθ(M0,f∂ₜγθθi,r,r̃)        )
    sample!(∂ᵣfrrr, grid, r̃ -> f∂ᵣfrrr(M0,r,r̃)              )
    sample!(∂ᵣfrθθ, grid, r̃ -> f∂ᵣfrθθ(M0,r,r̃)              )

    # Sample the state initial vector
    sample!(γrr,    grid, r̃ -> fγrr(M,r,r̃)                  )
    sample!(γθθ,    grid, r̃ -> fγθθ(M,r,r̃)                  )
    sample!(Krr,    grid, r̃ -> fKrr(M,f∂ₜγrr,r,r̃)            )
    sample!(Kθθ,    grid, r̃ -> fKθθ(M,f∂ₜγθθ,r,r̃)            )
    sample!(frrr,   grid, r̃ -> ffrrr(M,r,r̃)                 )
    sample!(frθθ,   grid, r̃ -> ffrθθ(M,r,r̃)                 )
    sample!(𝜙,      grid, r̃ -> f𝜙(M,r,r̃)                    )
    sample!(ψ,      grid, r̃ -> fψ(M,r,r̃)                    )
    sample!(Π,      grid, r̃ -> fΠ(M,r,r̃)                    )

    Mg(r̃) = M(r̃)
    # Sample the gauge variables
    sample!(ᾶ,      grid, r̃ -> fᾶ(Mg,r,r̃)                  )
    sample!(βʳ,     grid, r̃ -> fβʳ(Mg,r,r̃)                 )
    sample!(∂ᵣᾶ,    grid, r̃ -> f∂r̃ᾶ(Mg,r,r̃)/drdr̃(r̃)        )
    sample!(∂ᵣβʳ,   grid, r̃ -> f∂r̃βʳ(Mg,r,r̃)/drdr̃(r̃)       )
    sample!(∂ᵣ2ᾶ,   grid, r̃ -> (f∂r̃2ᾶ(Mg,r,r̃) - d2rdr̃(r̃)*f∂r̃ᾶ(Mg,r,r̃)/drdr̃(r̃))/drdr̃(r̃)^2   )
    sample!(∂ᵣ2βʳ,  grid, r̃ -> (f∂r̃2βʳ(Mg,r,r̃) - d2rdr̃(r̃)*f∂r̃βʳ(Mg,r,r̃)/drdr̃(r̃))/drdr̃(r̃)^2 )

    sample!(cp,    grid, r̃ -> fcp(Mg,r,r̃)                  )
    sample!(cm,    grid, r̃ -> fcm(Mg,r,r̃)                  )
    sample!(∂ᵣcp,  grid, r̃ -> f∂r̃cp(Mg,r,r̃)/drdr̃(r̃)        )
    sample!(∂ᵣcm,  grid, r̃ -> f∂r̃cm(Mg,r,r̃)/drdr̃(r̃)        )
    sample!(∂ᵣ2cp, grid, r̃ -> (f∂r̃2cp(Mg,r,r̃) - d2rdr̃(r̃)*f∂r̃cp(Mg,r,r̃)/drdr̃(r̃))/drdr̃(r̃)^2 )
    sample!(∂ᵣ2cm, grid, r̃ -> (f∂r̃2cm(Mg,r,r̃) - d2rdr̃(r̃)*f∂r̃cm(Mg,r,r̃)/drdr̃(r̃))/drdr̃(r̃)^2 )

    # Sample initial values of the characteristics

    global Upri = @part 1 Krr + frrr/sqrt(γrr)
    global Umri = @part n Krr - frrr/sqrt(γrr)

    # add noise to initial values to assess stability with magnitude s
    s = 0*10^(-10)

    for i in 1:numvar
        if i in reg_list
            for j in 1:n
               state.x[i][j] /= init_state.x[i][j]
               state.x[i][j] += s*rand(Uniform(-1,1))
            end
        else
            for j in 1:n
               state.x[i][j] += s*rand(Uniform(-1,1))
            end
        end
    end

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

    # In order to catch errors and still have the integrator finish
    try

    # Unpack the parameters

    grid = param.grid
    r = param.rsamp
    drdr̃ = param.drdr̃samp
    d2rdr̃ = param.d2rdr̃samp
    r̃min = param.r̃min
    r̃max = param.r̃max

    fr = param.r

    state = param.state
    drstate = param.drstate
    dtstate2 = param.dtstate
    temp = param.temp

    init_state = param.init_state
    init_drstate = param.init_drstate
    gauge = param.gauge
    speeds = param.speeds
    Dr = param.Dr

    # Copy the state into the parameters so that it can be changed

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
    ∂ᵣγrr,∂ᵣγθθ,∂ᵣKrr,∂ᵣKθθ,∂ᵣfrrr,∂ᵣfrθθ,∂ᵣ𝜙,∇ᵣψ,∂ᵣΠ = drstate.x
    ∂ₜγrr,∂ₜγθθ,∂ₜKrr,∂ₜKθθ,∂ₜfrrr,∂ₜfrθθ,∂ₜ𝜙,∂ₜψ,∂ₜΠ = dtstate.x
    ᾶ,βʳ,∂ᵣᾶ,∂ᵣβʳ,∂ᵣ2ᾶ,∂ᵣ2βʳ,α,∂ᵣlnᾶ,∂ᵣ2lnᾶ = gauge.x
    cp,cm,∂ᵣcp,∂ᵣcm,∂ᵣ2cp,∂ᵣ2cm,∂ᵣ3cp,∂ᵣ4cp,∂ᵣ5cp = speeds.x

    γrri,γθθi,Krri,Kθθi,frrri,frθθi,𝜙i,ψi,Πi = init_state.x
    ∂ᵣγrri,∂ᵣγθθi,∂ᵣKrri,∂ᵣKθθi,∂ᵣfrrri,∂ᵣfrθθi,∂ᵣ𝜙i,∂ᵣψi,∂ᵣΠi = init_drstate.x

    # Calculate first spatial derivatives by multipling D operator
    # and convert between the computational r̃ coordinate
    # and the traditional r coordinate

    Dr .= spdiagm(1. ./(sqrt.(γrr).*γθθ))*D*spdiagm(sqrt.(γrr).*γθθ)

    for i in 1:numvar
        mul!(drstate.x[i],D,state.x[i])
        @. drstate.x[i] /= drdr̃
    end

    # mul!(∂ᵣfrrr,Dr,frrr)
    # mul!(∂ᵣfrθθ,Dr,frθθ)
    mul!(∇ᵣψ,Dr,ψ)

    # Convert between regularized variables and cannonical variables

    reg = temp.x[1]; ∂reg = temp.x[2];

    for i in reg_list
        @. reg = state.x[i]; @. ∂reg = drstate.x[i];
        @. state.x[i] *= init_state.x[i]
        @. drstate.x[i] = (init_state.x[i]*∂reg
              + init_drstate.x[i]*reg )
    end

    # Source terms to GR

    ρ = temp.x[5]; Sr = temp.x[6]; Tt = temp.x[7]; Srr = temp.x[8]; Sθθ = temp.x[9];

    @. ρ = ( Π^2 + ψ^2/γrr + (m^2)*𝜙^2)/2 # Energy Density
    @. Sr = ψ*Π  # Momentum Density
    @. Tt = Π^2 - ψ^2/γrr - 2*(m^2)*𝜙^2  # Trace of the Stress-Energy tensor (T unavailable)
    @. Srr = γrr*( Π^2 + ψ^2/γrr - (m^2)*𝜙^2)/2  # Radial pressure component
    @. Sθθ = γθθ*( Π^2 - ψ^2/γrr - (m^2)*𝜙^2)/2  # Angular pressure component

    # Calculated lapse and derivatives of densitized lapse

    @. α = ᾶ*γθθ*sqrt(γrr)
    @. ∂ᵣlnᾶ = ∂ᵣᾶ/ᾶ
    @. ∂ᵣ2lnᾶ = (∂ᵣ2ᾶ*ᾶ - ∂ᵣᾶ^2)/ᾶ^2

    # Calculate the advection speeds

    @. cp = -βʳ + ᾶ*γθθ
    @. cm = -βʳ - ᾶ*γθθ

    #########################################################
    # Evolution Equations
    #
    # This is the full suite of evolution equations
    # for GR in spherical symmetry in the
    # 'Einstein-Christoffel' framework.
    #
    # Note: I have used subscript r and t where possible.
    # I could do this in principle for tensor variable indices
    # like γᵣᵣ for example, but annoyingly a subscript theta
    # does not exist in unicode, so tensor indices
    # are instead normal sized.
    #
    #########################################################

    @. ∂ₜγrr = βʳ*∂ᵣγrr + 2*∂ᵣβʳ*γrr - 2*α*Krr

    @. ∂ₜγθθ = βʳ*∂ᵣγθθ - 2*α*Kθθ

    @. ∂ₜKrr  = ( βʳ*∂ᵣKrr - α*∂ᵣfrrr/γrr + 2*α*frrr^2/γrr^2 - 6*α*frθθ^2/γθθ^2
     - α*Krr^2/γrr + 2*α*Krr*Kθθ/γθθ - 8*α*frrr*frθθ/(γrr*γθθ)
     - α*frrr*∂ᵣlnᾶ/γrr - α*∂ᵣlnᾶ^2 - α*∂ᵣ2lnᾶ + 2*∂ᵣβʳ*Krr)

    @. ∂ₜKθθ  = ( βʳ*∂ᵣKθθ - α*∂ᵣfrθθ/γrr + α + α*Krr*Kθθ/γrr
     - 2*α*frθθ^2/(γrr*γθθ) - α*frθθ*∂ᵣlnᾶ/γrr)

    @. ∂ₜfrrr = ( βʳ*∂ᵣfrrr - α*∂ᵣKrr - α*frrr*Krr/γrr
     + 12*α*frθθ*Kθθ*γrr/γθθ^2 - 10*α*frθθ*Krr/γθθ - 4*α*frrr*Kθθ/γθθ
     - α*Krr*∂ᵣlnᾶ - 4*α*Kθθ*γrr*∂ᵣlnᾶ/γθθ + 3*∂ᵣβʳ*frrr + γrr*∂ᵣ2βʳ )

    @. ∂ₜfrθθ = ( βʳ*∂ᵣfrθθ - α*∂ᵣKθθ - α*frrr*Kθθ/γrr + 2*α*frθθ*Kθθ/γθθ
     - α*Kθθ*∂ᵣlnᾶ + ∂ᵣβʳ*frθθ )

    # Klein-Gordon System

    @. ∂ₜ𝜙 =   βʳ*∂ᵣ𝜙 - α*Π

    @. ∂ₜψ = ( βʳ*∇ᵣψ - α*∂ᵣΠ - α*(frrr/γrr - 2*frθθ/γθθ + ∂ᵣlnᾶ)*Π
    - (βʳ*frrr/γrr - 2*βʳ*frθθ/γθθ - ∂ᵣβʳ)*ψ )

    @. ∂ₜΠ = ( βʳ*∂ᵣΠ - α*∇ᵣψ/γrr + α*(Krr/γrr + 2*Kθθ/γθθ)*Π
    + α*(frrr/γrr - 6*frθθ/γθθ - ∂ᵣlnᾶ)*ψ/γrr + m^2*α*𝜙 )

    # @. ∂ₜψ =   βʳ*∂ᵣψ - α*∂ᵣΠ - α*(frrr/γrr - 2*frθθ/γθθ + ∂ᵣlnᾶ)*Π + ψ*∂ᵣβʳ
    # @. ∂ₜΠ = ( βʳ*∂ᵣΠ - α*∂ᵣψ/γrr + α*(Krr/γrr + 2*Kθθ/γθθ)*Π
    #  - α*(4*frθθ/γθθ + ∂ᵣlnᾶ)*ψ/γrr + m^2*α*𝜙 )

    # Source terms to GR

    @. ∂ₜKrr  += 4*pi*α*(γrr*Tt - 2*Srr)
    @. ∂ₜKθθ  += 4*pi*α*(γθθ*Tt - 2*Sθθ)
    @. ∂ₜfrrr += 16*pi*α*γrr*Sr

    # Calculates the Apparent Horizon, if there is one
    # in the domain, no inner boundary conditions are applied

    AH = temp.x[1]
    @. AH = Kθθ - frθθ/sqrt(γrr)
    is_AH = false
    for i in 1:n-1 if AH[i]*AH[i+1] <= 0. is_AH = true; break; end end

    # s1 = 0.5 /(dr̃*Σ[1,1]*sqrt(γrr[1])*γθθ[1])
    # sn = 0.5 /(dr̃*Σ[n,n]*sqrt(γrr[n])*γθθ[n])

    if !(is_AH)

        ## Apply Inner Boundary Conditions

        Umθ = @part 1 ( Kθθ - frθθ/sqrt(γrr) )
        Upθ = @part 1 ( Kθθ + frθθ/sqrt(γrr) )

        Umr = @part 1 ( Krr - frrr/sqrt(γrr) )
        Upr = @part 1 ( Krr + frrr/sqrt(γrr) )

        Up𝜙   = @part 1 ( Π + ψ/sqrt(γrr) )
        Um𝜙   = @part 1 ( Π - ψ/sqrt(γrr) )


        Upθb = @part 1 ((2*M0*sqrt(γθθ) - γθθ)/Umθ)

        #Dirichlet on scalar
        #Up𝜙b = @part 1 -sqrt((cm*Upθb)/(cp*Umθ))*Um𝜙
        # #Neumann on scalar
        #Up𝜙b = @part 1 sqrt((cm*Upθb)/(cp*Umθ))*Um𝜙

        # Static Dirichlet
        Up𝜙b = @part 1 (cm/cp)*Um𝜙

        # ∂ᵣUmθ = @part 1 ∂ᵣKθθ - ∂ᵣfrθθ/sqrt(γrr) + frθθ*(2*frrr - 8*frθθ*γrr/γθθ)/(2*sqrt(γrr)^3)
        #Uprb = @part 1 (-Umr - γrr*Umθ/γθθ - (2*∂ᵣUmθ*sqrt(γrr) + γrr)/Umθ )

        Uprb = Upri

        #∂ᵣUpθ = @part 1 ( ∂ᵣKθθ + ∂ᵣfrθθ/sqrt(γrr) - ∂ᵣγrr*frθθ/sqrt(γrr)^3/2 )
        #∂ᵣUmθ = @part n ( ∂ᵣKθθ - ∂ᵣfrθθ/sqrt(γrr) + ∂ᵣγrr*frθθ/sqrt(γrr)^3/2 )

        # Uprb = @part 1 (-Umr - Upθ*γrr/γθθ + 2*∂ᵣUpθ*sqrt(γrr)/Upθ - γrr/Upθ
        #      + 8*pi*γrr*γθθ*(ρ + Sr/sqrt(γrr))/Upθ )

        # Uprb = @part 1 (-Umr - Umθ*γrr/γθθ - 2*∂ᵣUmθ*sqrt(γrr)/Umθ - γrr/Umθ
        #   + 8*pi*γrr*γθθ*(ρ - Sr/sqrt(γrr))/Umθ )

        #Dirichlet on r-mode
        #Uprb = @part 1 (cm/cp)*(Umr-(Krri - frrri/sqrt(γrri))) + Krri + frrri/sqrt(γrri)

        # ∂ₜΠ[1] = 0
        # ∂ₜψ[1] += s*(-Π[1])/(dr̃*Σ[1,1])/2.

        s1 = abs(cp[1])/Σ[1,1]

        ∂ₜΠ[1] += s1*(Up𝜙b - Up𝜙)/2.
        ∂ₜψ[1] += s1*sqrt(γrr[1])*(Up𝜙b - Up𝜙)/2.

        ∂ₜKrr[1]  += s1*(Uprb - Upr)/2.
        ∂ₜfrrr[1] += s1*sqrt(γrr[1])*(Uprb - Upr)/2.

        ∂ₜKθθ[1]  += s1*(Upθb - Upθ)/2.
        ∂ₜfrθθ[1] += s1*sqrt(γrr[1])*(Upθb - Upθ)/2.

    end

    ## Outer Boundary Conditions

    Umθ = @part n ( Kθθ - frθθ/sqrt(γrr) )
    Upθ = @part n ( Kθθ + frθθ/sqrt(γrr) )

    Umr = @part n ( Krr - frrr/sqrt(γrr) )
    Upr = @part n ( Krr + frrr/sqrt(γrr) )

    Up𝜙 = @part n ( Π + ψ/sqrt(γrr) )
    Um𝜙 = @part n ( Π - ψ/sqrt(γrr) )

    # # Transmitting conditions
    #
    # Umθb =
    #
    # #Transmission on scalar
    # Up𝜙b = 0.

    # Reflecting conditions

    Umθb = @part n ((2*Mtot*sqrt(γθθ) - γθθ)/Upθ)

    #Dirichlet on scalar
    #Um𝜙b = @part n -sqrt((cp*Umθb)/(cm*Upθ))*Up𝜙
    # #Neumann on scalar
    # Up𝜙b = @part 1 sqrt((cm*Upθb)/(cp*Umθ))*Um𝜙

    # Static Neumann
    Um𝜙b = @part n -(cp/cm)*Up𝜙

    Umrb = Umri

    #∂ᵣUmθ = @part n ( ∂ᵣKθθ - ∂ᵣfrθθ/sqrt(γrr) + ∂ᵣγrr*frθθ/sqrt(γrr)^3/2 )

    # Umrb = @part n (-Upr - Umθ*γrr/γθθ - 2*∂ᵣUmθ*sqrt(γrr)/Umθ - γrr/Umθ
    #      + 8*pi*γrr*γθθ*(ρ - Sr/sqrt(γrr))/Umθ )

    @part n ∂ₜγrr = ( (2*frrr - 8*frθθ*γrr/γθθ)*βʳ + 2*∂ᵣβʳ*γrr - 2*α*Krr )
    @part n ∂ₜγθθ = ( 2*frθθ*βʳ - 2*α*Kθθ )
    #@part n ∂ₜγθθ = sqrt(γrr)*(cm*Umθb-cp*Upθ)
    @part n ∂ₜ𝜙   = (βʳ*ψ - α*Π)

    sn = abs(cm[n])/Σ[n,n]

    ∂ₜΠ[n] += sn*(Um𝜙b - Um𝜙)/2.
    ∂ₜψ[n] += -sn*sqrt(γrr[n])*(Um𝜙b - Um𝜙)/2.

    ∂ₜKrr[n]  += sn*(Umrb - Umr)/2.
    ∂ₜfrrr[n] += -sn*sqrt(γrr[n])*(Umrb - Umr)/2.

    ∂ₜKθθ[n]  += sn*(Umθb - Umθ)/2.
    ∂ₜfrθθ[n] += -sn*sqrt(γrr[n])*(Umθb - Umθ)/2.

    # γrrrhs = ∂ₜγrr[n]; γθθrhs = ∂ₜγθθ[n];
    # Krrrhs = ∂ₜKrr[n]; frrrrhs = ∂ₜfrrr[n];
    # Kθθrhs = ∂ₜKθθ[n]; frθθrhs = ∂ₜfrθθ[n];
    #
    # ∂ₜU0r = ∂ₜγrr[n]
    # ∂ₜUmr = 0.
    #
    # @part n ∂ₜKrr  = ∂ₜUmr/2 + Krrrhs/2 + frrrrhs/sqrt(γrr)/2 - frrr*γrrrhs/4/sqrt(γrr)^3
    # @part n ∂ₜfrrr = (frrrrhs/2 - ∂ₜUmr*sqrt(γrr)/2 + Krrrhs*sqrt(γrr)/2
    #  - frrr*γrrrhs/4/γrr + frrr*∂ₜU0r/2/γrr)
    #
    # ∂ᵣUmθ = @part n ( -(Umr + Upr)*Umθ/2/sqrt(γrr) - (1. + Umθ^2/γθθ)*sqrt(γrr)/2
    #     + 4*pi*sqrt(γrr)*γθθ*(ρ - Sr/sqrt(γrr)) )
    #
    # ∂ₜUmθ = @part n ( α - (-βʳ - α/sqrt(γrr))*∂ᵣUmθ + Upr*Umθ*α/γrr
    #     - (Upθ - Umθ)*Umθ*α/γθθ + α*∂ᵣlnᾶ*Umθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )
    #
    # @part n ∂ₜKθθ  = ∂ₜUmθ/2 + Kθθrhs/2 + frθθrhs/sqrt(γrr)/2 - frθθ*γrrrhs/4/sqrt(γrr)^3
    # @part n ∂ₜfrθθ = (frθθrhs/2 - ∂ₜUmθ*sqrt(γrr)/2 + Kθθrhs*sqrt(γrr)/2
    #  - frθθ*γrrrhs/γrr/4 + frθθ*∂ₜU0r/γrr/2)

    for i in 1:6
        @. dtstate.x[i] = 0.
    end

    # Store the calculated state into the param
    # so that we can print it to the screen

    for i in 1:numvar
        dtstate2.x[i] .= dtstate.x[i]
    end

    # Convert back to regularized variables

    for i in reg_list
        @. dtstate.x[i] /= init_state.x[i]
    end

    # Add the numerical dissipation to regularized dtstate

    # for i in 7:numvar
    #     mul!(dtstate.x[i],A,regstate.x[i],1.,1.)
    #     # this syntax is equivalent to dtstate.x[i] .+= A4*regstate.x[i]
    # end

    # catch any errors, save them to print later
    catch e
        global_error.error = e
        global_error.stacktrace = stacktrace(catch_backtrace())
    end

end

function rhs_all(regstate::VarContainer{T}, param::Param{T}, t) where T

    # Runs the right-hand-side routine, but with allocation so that
    # the state can be saved at the end.

    n = param.grid.ncells

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
    ∂ᵣγrr,∂ᵣγθθ,∂ᵣKrr,∂ᵣKθθ,∂ᵣfrrr,∂ᵣfrθθ,∂ᵣ𝜙,∂ᵣψ,∂ᵣΠ = drstate.x
    ᾶ,βʳ,∂ᵣᾶ,∂ᵣβʳ,∂ᵣ2ᾶ,∂ᵣ2βʳ,∂ᵣ3βʳ,∂ᵣ4βʳ,∂ᵣ5βʳ = gauge.x

    init_state = param.init_state
    init_drstate = param.init_drstate

    r = param.rsamp
    drdr̃ = param.drdr̃samp
    d2rdr̃ = param.d2rdr̃samp
    temp = param.temp
    grid = param.grid
    Dr = param.Dr

    # for i in reg_list
    #     @. state.x[i] /= init_state.x[i]
    # end

    for i in 1:numvar
        mul!(drstate.x[i],D,state.x[i])
        @. drstate.x[i] /= drdr̃
    end

    reg = temp.x[1]; ∂reg = temp.x[2];

    for i in reg_list
        @. reg = state.x[i]; @. ∂reg = drstate.x[i];
        @. state.x[i] *= init_state.x[i]
        @. drstate.x[i] = (init_state.x[i]*∂reg
              + init_drstate.x[i]*reg )
    end

    α = temp.x[3]; ρ = temp.x[4]; Sr = temp.x[5]

    @. α = ᾶ*γθθ*sqrt(γrr)
    @. ρ = (Π^2 + ψ^2/γrr + (m^2)*𝜙^2)/2.
    #Lower Index
    @. Sr = ψ*Π

    # Constraint Equations

    C = zeros(T,n); Cr = zeros(T,n); Crrr = zeros(T,n); Crθθ = zeros(T,n);
    C𝜙 = zeros(T,n);

    @. C = (∂ᵣfrθθ/(γθθ*γrr) + 7*frθθ^2/(2*γrr*γθθ^2) - frrr*frθθ/(γrr^2*γθθ)
     - Kθθ^2/(2*γθθ^2) - 1/(2*γθθ) - Krr*Kθθ/(γrr*γθθ) + 4*pi*ρ)

    @. Cr = (∂ᵣKθθ/γθθ - frθθ*Kθθ/γθθ^2 - frθθ*Krr/(γθθ*γrr) + 4*pi*Sr)

    @. Crrr = ∂ᵣγrr + 8*frθθ*γrr/γθθ - 2*frrr

    @. Crθθ = ∂ᵣγθθ - 2*frθθ

    @. C𝜙 = ∂ᵣ𝜙 - ψ

    #E = dr̃*(Krr')*Σ*(D*𝜙) + dr̃*(𝜙')*Σ*(D*Krr) - (Krr[n]*𝜙[n]-Krr[1]*𝜙[1])

    #E  = dr̃*sum(Σ*( @. (α*ρ - βʳ*Sr)*4*pi*sqrt(γrr)*γθθ*drdr̃))

    # E  = dr̃*(Π')*Σ*spdiagm(@. sqrt(γrr)*γθθ)*(@. α*Π/2.)
    #    + dr̃*(ψ')*Σ*spdiagm(@. sqrt(γrr)*γθθ)*(@. α*ψ/γrr/2.)
    #    - dr̃*(Π')*Σ*spdiagm(@. sqrt(γrr)*γθθ)*(@. βʳ*ψ/α )
    rootγ = spdiagm(sqrt.(γrr).*γθθ)
    invrootγ = spdiagm(1. ./(sqrt.(γrr).*γθθ))
    Dr .= invrootγ*D*rootγ
    Wv = Σ*rootγ; Ws = rootγ*Σ;
    Bvec1 = zeros(T,n); Bvec1[1] = -1.; Bvec1[n] = 1.;
    B = spdiagm(Bvec1);
    #println((dr̃*Wv*Dr + dr̃*(Ws*D)' + B*rootγ)[1:6,1:6])
    #println((Σ*D + (Σ*D)' - B)[1:6,1:6])
    #println((dr̃*Wg*Dr + dr̃*(D')*Wg + B*rootγ)[1:6,1:6])

    #E  = dr̃*( Π'*Wv*(Dr*ψ) + (D*Π)'*Ws*ψ ) #check
    #E  = (Π'*Wv*Π +  ψ'*Wv*ψ )
    E = (α.*Π)'*Wv*Π/2. +  (α.*ψ./γrr)'*Wv*ψ/2. - (βʳ.*Π)'*Wv*ψ

    #E  = dr̃*sum(Σ*( @. (frθθ*ρ - Kθθ*Sr)*4*pi*sqrt(γθθ)*drdr̃ ) )
    Ec = dr̃*sum(Σ*( @. (C^2 + Cr^2/γrr)*4*pi*sqrt(γrr)*γθθ*drdr̃ ))

    return [C, Cr, Crrr, Crθθ, C𝜙, E, Ec]

end

function custom_progress_message(dt,state::VarContainer{T},param,t) where T

    ###############################################
    # Outputs status numbers while the program runs
    ###############################################

    dtstate = param.dtstate::VarContainer{T}

    ∂ₜγrr,∂ₜγθθ,∂ₜKrr,∂ₜKθθ,∂ₜfrrr,∂ₜfrθθ,∂ₜ𝜙,∂ₜψ,∂ₜΠ = dtstate.x

    println("  ",
    rpad(string(round(t,digits=1)),10," "),
    rpad(string(round(maximum(abs.(∂ₜγrr)), digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂ₜγθθ)), digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂ₜKrr)), digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂ₜKθθ)), digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂ₜfrrr)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂ₜfrθθ)),digits=3)),12," ")
    )

    return

end


function solution_saver(T,grid,sol,param)

    ###############################################
    # Saves all of the variables in nice CSV files
    # in the choosen data folder directory
    ###############################################

    folder = string("n=",      n,
                    "_rspan=", round.(r̃span, digits=2),
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

    vars = (["grr","gtt","Krr","Ktt","frrr","frtt","phi","psi","Pi",
    #"∂ₜγrr","∂ₜγθθ","∂ₜKrr","∂ₜKθθ","∂ₜfrrr","∂ₜfrθθ","∂ₜ𝜙","∂ₜψ","∂ₜΠ",
    "C","Cr","Crrr","Crtt","Cphi","E","Ec"])

    # vars = (["γrr","γθθ","Krr","Kθθ","frrr","frθθ","𝜙","ψ","Π",
    # #"∂ₜγrr","∂ₜγθθ","∂ₜKrr","∂ₜKθθ","∂ₜfrrr","∂ₜfrθθ","∂ₜ𝜙","∂ₜψ","∂ₜΠ",
    # "H","Mr","Crrr","Crθθ","C𝜙","E","Ec"])
    varlen = length(vars)
    tlen = size(sol)[2]
    rlen = grid.ncells
    r = param.rsamp
    r̃min = param.r̃min

    init_state = param.init_state
    init_drstate = param.init_drstate

    #dtstate = [rhs_all(sol[i],param,0.) for i = 1:tlen]


    cons = [constraints(sol[i],param) for i = 1:tlen]

    # ens = [energies(sol[i],param) for i = 1:tlen]

    array = Array{T,2}(undef,tlen,rlen)

    save(string(path,"/coords.h5"), Dict("r"=>r,"t"=>sol.t[:]) )

    for j = 1:numvar

        if j in reg_list
            for i = 1:tlen
                @. array[i,:] = sol[i].x[j]*init_state.x[j]
            end
            save(string(path,"/",vars[j],".h5"), Dict(vars[j]=>array ) )
        else
            for i = 1:tlen
                @. array[i,:] = sol[i].x[j]
            end
            save(string(path,"/",vars[j],".h5"), Dict(vars[j]=>array ) )
        end

    end

    # for j = 1:numvar
    #
    #     if j in reg_list
    #         for i = 2:tlen+1
    #             array[i,1] = sol.t[i-1]
    #             @. array[i,2:end] = dtstate[i-1].x[j]*init_state.x[j]
    #         end
    #     else
    #         for i = 2:tlen+1
    #             array[i,1] = sol.t[i-1]
    #             @. array[i,2:end] = dtstate[i-1].x[j]
    #         end
    #     end
    #
    #     CSV.write(
    #         string(path,"/",vars[j+numvar],".csv"),
    #         DataFrame(array, :auto),
    #         header=false
    #     )
    #
    # end
    # println(length(cons))
    # println(size(cons[:][1]))
    # println(size(array[1,:]))
    # return

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

    # for i = 2:tlen+1
    #     array[i,1] = sol.t[i-1]
    #     array[i,2] = cons[i-1][4]
    #     array[i,3:end] .= 0
    # end
    #
    # CSV.write(
    # string("data/",folder,"/","E-",r̃min,".csv"),
    # DataFrame(array, :auto),
    # header=false
    # )

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
function error_handler(regstate,t,integrator)
    return !(typeof(global_error.error) == NothingException)
end

# Terminates the integrator when error_handler returns true
cb = DiscreteCallback(error_handler,terminate!,save_positions=(false,false))


function main()

    ###############################################
    # Main Program
    #
    # Calls each of the above functions to run a
    # simulation. Sets up the numerical grid,
    # sets the gauge conditions, sets up a new
    # computational r̃ coordinate that makes the
    # physical r coordinate step larger near the
    # outer boundary, sets the initial conditions,
    # and finally runs the numerical DiffEq
    # package to run the time integration.
    #
    # All data is saved in the folder specified to
    # the solution_saver, each in their own CSV
    # file.
    ###############################################

    r̃min, r̃max = r̃span
    rspan = T[r̃min,r̃max]

    domain = Domain{T}(r̃min, r̃max)
    grid = Grid(domain, n)

    tmin, tmax = tspan

    atol = eps(T)^(T(3) / 4)

    alg = RK4()
    #alg = Vern6()

    #printlogo()

    custom_progress_step = round(Int64, printtimes/dt)
    step_iterator = custom_progress_step

    regstate = similar(ArrayPartition,T,n)

    state = similar(ArrayPartition,T,n)
    drstate = similar(ArrayPartition,T,n)

    init_state = similar(ArrayPartition,T,n)
    init_drstate = similar(ArrayPartition,T,n)

    gauge = similar(ArrayPartition,T,n)
    speeds = similar(ArrayPartition,T,n)
    dtstate = similar(ArrayPartition,T,n)
    dissipation = similar(ArrayPartition,T,n)
    temp = similar(ArrayPartition,T,n)

    #println("Defining Problem...")
    rsamp = similar(Vector{T}(undef,n))
    drdr̃samp = similar(Vector{T}(undef,n))
    d2rdr̃samp = similar(Vector{T}(undef,n))
    #right_boundary = similar(Vector{T}(undef,numvar))

    sample!(rsamp, grid, r̃ -> r(r̃) )
    sample!(drdr̃samp, grid, r̃ -> drdr̃(r̃) )
    sample!(d2rdr̃samp, grid, r̃ -> d2rdr̃(r̃) )

    #return

    param = Param(
    r̃min,r̃max,grid,
    r,drdr̃,d2rdr̃,
    rsamp,drdr̃samp,d2rdr̃samp,gauge,speeds,
    init_state,init_drstate,
    state,drstate,
    dtstate,dissipation,temp,copy(D))

    init!(regstate, param)

    # return

    prob = ODEProblem(rhs!, regstate, tspan, param)

    #println("Starting Solution...")
    println("")
    println("| Time | max ∂ₜγrr | max ∂ₜγθθ | max ∂ₜKrr | max ∂ₜKθθ | max ∂ₜfrrr | max ∂ₜfrθθ |")
    println("|______|___________|___________|___________|___________|____________|____________|")
    println("")


    sol = solve(
        prob, alg,
        abstol = atol,
        dt = dt,
        adaptive = false,
        saveat = savetimes,
        alias_u0 = true,
        progress = true,
        progress_steps = custom_progress_step,
        progress_message = custom_progress_message,
        callback = cb
    )

    solution_saver(T,grid,sol,param)

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
