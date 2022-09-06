module GR_Spherical

using DifferentialEquations
using BoundaryValueDiffEq
using OrdinaryDiffEq
#using Fun1d
using DataFrames
using CSV
#using Plots
using Roots

using BenchmarkTools
using InteractiveUtils
using RecursiveArrayTools
using SparseArrays
using LinearAlgebra
using BandedMatrices

using Distributions
using ForwardDiff

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
    r̃min::T
    r̃max::T
    Mtot::T
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

spacing(grid::Grid) = (grid.domain.xmax - grid.domain.xmin) / (grid.ncells - 1)

function sample!(f::Vector{T}, grid::Grid{S}, fun) where {S,T}

    dr̃ = spacing(grid)
    r̃min = grid.domain.xmin

    f .= T[fun(r̃min + dr̃*(j-1)) for j in 1:(grid.ncells)]

end

#Kerr-Schild Coordinates
# sign=1 for ingoing (black hole), sign=-1 for outgoing (white hole)

sign = 1.

fᾶ(M,r,r̃) = 1/(r(r̃)^2 + 2*M(r̃)*r(r̃))
fβʳ(M,r,r̃) = sign*2*M(r̃)/(2*M(r̃)+r(r̃))
fγrr(M,r,r̃) = 1 + 2*M(r̃)/r(r̃)
fγθθ(M,r,r̃) = r(r̃)^2
fα(M,r,r̃) = fᾶ(M,r,r̃)*fγθθ(M,r,r̃)*sqrt(fγrr(M,r,r̃))
fKrr(M,∂ᵣM,r,r̃) = sign*(2*(r(r̃)*∂ᵣM(r̃)-M(r̃))/r(r̃)^3)*(r(r̃)+M(r̃))/sqrt(1+2*M(r̃)/r(r̃))
fKθθ(M,r,r̃) = sign*2*M(r̃)/sqrt((1+2*M(r̃)/r(r̃)))
ffrrr(M,∂ᵣM,r,r̃) = (7*M(r̃) + (4 + ∂ᵣM(r̃))*r(r̃))/(r(r̃)^2)
ffrθθ(M,r,r̃) = r(r̃)

fᾶ(M::Number,r,r̃) = 1/(r(r̃)^2+2*M*r(r̃))
fβʳ(M::Number,r,r̃) = sign*2*M/(2*M+r(r̃))
fγrr(M::Number,r,r̃) = 1 + 2*M/r(r̃)
fKrr(M::Number,∂ᵣM::Number,r,r̃) = sign*(2*(r(r̃)*∂ᵣM-M)/r(r̃)^3)*(r(r̃)+M)/sqrt(1+2*M/r(r̃))
ffrrr(M::Number,∂ᵣM::Number,r,r̃) = (7*M + (4 + ∂ᵣM)*r(r̃))/(r(r̃)^2)

fcp(M,r,r̃) = -fβʳ(M,r,r̃) + fα(M,r,r̃)/sqrt(fγrr(M,r,r̃))
fcm(M,r,r̃) = -fβʳ(M,r,r̃) - fα(M,r,r̃)/sqrt(fγrr(M,r,r̃))

#Painleve-Gullstrand Coordinates

# fᾶ(M,r,r̃) = 1.
# fβʳ(M,r,r̃) = sqrt(2*M(r̃)/r(r̃))
# fγrr(M,r,r̃) = 1.
# fγθθ(M,r,r̃) = r(r̃)^2
# fKrr(M,∂ᵣM,r,r̃) = -sqrt(M(r̃)/(2*r(r̃)^3))
# fKθθ(M,r,r̃) = r(r̃)*sqrt(2*M(r̃)/r(r̃))
# ffrrr(M,∂ᵣM,r,r̃) = 4/r(r̃)
# ffrθθ(M,r,r̃) = r(r̃)

# Schwarzschild
#
# r0 = 5.
# σr = 0.1
# Amp = 0.00001
# fᾶ(M,r,r̃) = sqrt(1. - 2*M(r̃)/r(r̃)) + Amp*exp(-(1/2)*((r(r̃)-r0)/σr)^2)
# fβʳ(M,r,r̃) = 0.
# fγrr(M,r,r̃) = 1/(1 - 2*M(r̃)/r(r̃))
# fγθθ(M,r,r̃) = r(r̃)^2
# fKrr(M,∂ᵣM,r,r̃) = 0.
# fKθθ(M,r,r̃) = 0.
# ffrrr(M,∂ᵣM,r,r̃) = (-17*M(r̃) + (8 + ∂ᵣM(r̃))*r(r̃))/(r(r̃)-2*M(r̃))^2
# ffrθθ(M,r,r̃) = r(r̃)

# Cartesian Minkowski

# fᾶ(M,r,r̃) = 1.
# fβʳ(M,r,r̃) = 0.
# fγrr(M,r,r̃) = 1.
# fγθθ(M,r,r̃) = 1.
# fKrr(M,∂ᵣM,r,r̃) = 0.
# fKθθ(M,r,r̃) = 0.
# ffrrr(M,∂ᵣM,r,r̃) = 0.
# ffrθθ(M,r,r̃) = 0.


f∂r̃ᾶ(M,r,r̃)         = ForwardDiff.derivative(r̃ -> fᾶ(M,r,r̃), r̃)
f∂r̃2ᾶ(M,r,r̃)        = ForwardDiff.derivative(r̃ -> f∂r̃ᾶ(M,r,r̃), r̃)
f∂r̃βʳ(M,r,r̃)        = ForwardDiff.derivative(r̃ -> fβʳ(M,r,r̃), r̃)
f∂r̃2βʳ(M,r,r̃)       = ForwardDiff.derivative(r̃ -> f∂r̃βʳ(M,r,r̃), r̃)
f∂r̃cp(M,r,r̃)        = ForwardDiff.derivative(r̃ -> fcp(M,r,r̃), r̃)
f∂r̃2cp(M,r,r̃)       = ForwardDiff.derivative(r̃ -> f∂r̃cp(M,r,r̃), r̃)
f∂r̃cm(M,r,r̃)        = ForwardDiff.derivative(r̃ -> fcm(M,r,r̃), r̃)
f∂r̃2cm(M,r,r̃)       = ForwardDiff.derivative(r̃ -> f∂r̃cm(M,r,r̃), r̃)

f∂r̃γrr(M,r,r̃)       = ForwardDiff.derivative(r̃ -> fγrr(M,r,r̃), r̃)
f∂r̃2γrr(M,r,r̃)      = ForwardDiff.derivative(r̃ -> f∂r̃γrr(M,r,r̃), r̃)
f∂r̃γθθ(M,r,r̃)       = ForwardDiff.derivative(r̃ -> fγθθ(M,r,r̃), r̃)
f∂r̃2γθθ(M,r,r̃)      = ForwardDiff.derivative(r̃ -> f∂r̃γθθ(M,r,r̃), r̃)
f∂r̃Kθθ(M,r,r̃)       = ForwardDiff.derivative(r̃ -> fKθθ(M,r,r̃), r̃)
f∂r̃2Kθθ(M,r,r̃)      = ForwardDiff.derivative(r̃ -> f∂r̃Kθθ(M,r,r̃), r̃)
f∂r̃frrr(M,∂ᵣM,r,r̃)  = ForwardDiff.derivative(r̃ -> ffrrr(M,∂ᵣM,r,r̃), r̃)
f∂r̃2frrr(M,∂ᵣM,r,r̃) = ForwardDiff.derivative(r̃ -> f∂r̃frrr(M,∂ᵣM,r,r̃), r̃)
f∂r̃frθθ(M,r,r̃)      = ForwardDiff.derivative(r̃ -> ffrθθ(M,r,r̃), r̃)
f∂r̃2frθθ(M,r,r̃)     = ForwardDiff.derivative(r̃ -> f∂r̃frθθ(M,r,r̃), r̃)
f∂r̃Krr(M,∂ᵣM,r,r̃)   = ForwardDiff.derivative(r̃ -> fKrr(M,∂ᵣM,r,r̃), r̃)
f∂r̃2Krr(M,∂ᵣM,r,r̃)  = ForwardDiff.derivative(r̃ -> f∂r̃Krr(M,∂ᵣM,r,r̃), r̃)

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
    dr̃ = spacing(grid)
    r = param.r
    drdr̃ = param.drdr̃
    d2rdr̃ = param.d2rdr̃
    r̃min = param.r̃min
    r̃max = param.r̃max

    Mtot = param.Mtot
    n = grid.ncells
    m = 0.
    r̃span = (r̃min,r̃max)

    # Mass (no real reason not to use 1 here)
    #M = 1

    fΠ(M,r,r̃) = -(f∂ₜ𝜙(M,r,r̃) - fβʳ(M,r,r̃)*fψ(M,r,r̃) )/fα(M,r,r̃)
    #fΠ(M,r,r̃) = 0.

    fρ(M,r,r̃) = ( fΠ(M,r,r̃)^2 + fψ(M,r,r̃)^2/fγrr(M,r,r̃) + m^2*f𝜙(M,r,r̃)^2 )/2.
    fSr(M,r,r̃) = fψ(M,r,r̃)*fΠ(M,r,r̃)

    f∂r̃M_KS(M,r,r̃) = 4*pi*r(r̃)^2*(fρ(M,r,r̃) - fβʳ(M,r,r̃)*fSr(M,r,r̃)/fα(M,r,r̃))*drdr̃(r̃)
    f∂ₜγrr_KS(M,r,r̃) = -8*pi*r(r̃)*fSr(M,r,r̃)/fα(M,r,r̃)
    # f∂r̃M_KS(M,r,r̃) = 4*pi*r(r̃)^2*(fρ(M,r,r̃))*drdr̃(r̃)
    # f∂ₜγrr_KS(M,r,r̃) = 0.

    # f∂r̃M_PG(M,r,r̃) = 4*pi*r(r̃)^2*(fρ(M,r,r̃) - fSr(M,r,r̃)*sqrt(2*M(r̃)/r(r̃))/2)*drdr̃(r̃)
    # f∂ₜγrr_PG(M,r,r̃) = -8*pi*r(r̃)*Sr

    # Constraint Equations

    function constraintSystem(M, param, r̃)
        r = param.r
        f∂r̃M_KS(M,r,r̃)
    end

    BVP = ODEProblem(constraintSystem, 1., r̃span, param)
    Mass = solve(BVP, Tsit5(), abstol=1e-15, dt=dr̃, adaptive=false)

    Mtot = Mass(r̃max)
    M(r̃) = Mass(r̃)
    ∂ᵣM(r̃) = f∂r̃M_KS(M,r,r̃)/drdr̃(r̃)

    println("")
    println(string("Total Mass: ",round(Mass(r̃max), digits=3)))

    #fKrri(M,∂ᵣM,r,r̃)   =  fKrr(M,∂ᵣM,r,r̃)
    fKrri(M,∂ᵣM,r,r̃)   = -f∂ₜγrr_KS(M,r,r̃)/fα(M,r,r̃)/2 + fKrr(M,∂ᵣM,r,r̃)
    f∂r̃Krri(M,∂ᵣM,r,r̃) = ForwardDiff.derivative(r̃ -> fKrri(M,∂ᵣM,r,r̃), r̃)

    # M(r̃) = 1.
    # ∂ᵣM(r̃) = 0.
    M0(r̃) = 1.
    ∂ᵣM0(r̃) = 0.
    # M0(r̃) = Mass(r̃)
    # ∂ᵣM0(r̃) = ∂ᵣM(r̃)

    sample!(γrri,   grid, r̃ -> fγrr(M0,r,r̃)                 )
    sample!(γθθi,   grid, r̃ -> fγθθ(M0,r,r̃)                 )
    sample!(Krri,   grid, r̃ -> fKrri(M0,∂ᵣM0,r,r̃)           )
    sample!(Kθθi,   grid, r̃ -> fKθθ(M0,r,r̃)                 )
    sample!(frrri,  grid, r̃ -> ffrrr(M0,∂ᵣM0,r,r̃)           )
    sample!(frθθi,  grid, r̃ -> ffrθθ(M0,r,r̃)                )
    sample!(𝜙i,     grid, r̃ -> f𝜙(M0,r,r̃)                   )
    sample!(ψi,     grid, r̃ -> fψ(M0,r,r̃)                   )
    sample!(Πi,     grid, r̃ -> fΠ(M0,r,r̃)                   )

    sample!(γrr,    grid, r̃ -> fγrr(M,r,r̃)                  )
    sample!(γθθ,    grid, r̃ -> fγθθ(M,r,r̃)                  )
    sample!(Krr,    grid, r̃ -> fKrri(M,∂ᵣM,r,r̃)             )
    sample!(Kθθ,    grid, r̃ -> fKθθ(M,r,r̃)                  )
    sample!(frrr,   grid, r̃ -> ffrrr(M,∂ᵣM,r,r̃)             )
    sample!(frθθ,   grid, r̃ -> ffrθθ(M,r,r̃)                 )
    sample!(𝜙,      grid, r̃ -> f𝜙(M,r,r̃)                   )
    sample!(ψ,      grid, r̃ -> fψ(M,r,r̃)                    )
    sample!(Π,      grid, r̃ -> fΠ(M,r,r̃)                    )

    sample!(∂ᵣγrr,  grid, r̃ -> f∂r̃γrr(M0,r,r̃)/drdr̃(r̃)       )
    sample!(∂ᵣγθθ,  grid, r̃ -> f∂r̃γθθ(M0,r,r̃)/drdr̃(r̃)       )
    sample!(∂ᵣKrr,  grid, r̃ -> f∂r̃Krri(M0,∂ᵣM0,r,r̃)/drdr̃(r̃) )
    sample!(∂ᵣKθθ,  grid, r̃ -> f∂r̃Kθθ(M0,r,r̃)/drdr̃(r̃)       )
    sample!(∂ᵣfrrr, grid, r̃ -> f∂r̃frrr(M0,∂ᵣM0,r,r̃)/drdr̃(r̃) )
    sample!(∂ᵣfrθθ, grid, r̃ -> f∂r̃frθθ(M0,r,r̃)/drdr̃(r̃)      )

    Mg(r̃) = M(r̃)

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

    s = 0*10^(-10)

    for i in 1:numvar
        if i in reg_list
            for j in 1:n
               state.x[i][j] /= init_state.x[i][j]
               state.x[i][j] += s*rand(Uniform(-1,1))
            end
            # state.x[i][1:10] .= 1.
            # state.x[i][n-9:n] .= 1.
        else
            for j in 1:n
               state.x[i][j] += s*rand(Uniform(-1,1))
            end
            # state.x[i][1:10] .= init_state.x[i][1:10]
            # state.x[i][n-9:n] .= init_state.x[i][n-9:n]
        end
    end

end

@inline function deriv!(df::Vector{T}, f::Vector{T}, n::Int64, dr̃::T) where T

    #######################################################
    # Calculates derivatives using a 4th order SBP operator
    #######################################################

    df[1:5] .= ql*f[1:7]/dr̃

    for i in 6:(n - 5)
        df[i] = (f[i-2] - 8*f[i-1] + 8*f[i+1] - f[i+2])/(12*dr̃)
    end

    df[n-4:n] .= qr*f[n-6:n]/dr̃

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

    m = 0.
    Mtot = 1.
    M = 1.

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
    ∂ᵣγrr,∂ᵣγθθ,∂ᵣKrr,∂ᵣKθθ,∂ᵣfrrr,∂ᵣfrθθ,∂ᵣ𝜙,∂ᵣψ,∂ᵣΠ = drstate.x
    ∂ₜγrr,∂ₜγθθ,∂ₜKrr,∂ₜKθθ,∂ₜfrrr,∂ₜfrθθ,∂ₜ𝜙,∂ₜψ,∂ₜΠ = dtstate.x
    ᾶ,βʳ,∂ᵣᾶ,∂ᵣβʳ,∂ᵣ2ᾶ,∂ᵣ2βʳ,α,∂ᵣlnᾶ,∂ᵣ2lnᾶ = gauge.x
    cp,cm,∂ᵣcp,∂ᵣcm,∂ᵣ2cp,∂ᵣ2cm,∂ᵣ3cp,∂ᵣ4cp,∂ᵣ5cp = speeds.x

    γrri,γθθi,Krri,Kθθi,frrri,frθθi,𝜙i,ψi,Πi = init_state.x
    ∂ᵣγrri,∂ᵣγθθi,∂ᵣKrri,∂ᵣKθθi,∂ᵣfrrri,∂ᵣfrθθi,∂ᵣ𝜙i,∂ᵣψi,∂ᵣΠi = init_drstate.x

    # Calculate first spatial derivatives by multipling D operator

    for i in 1:numvar
        mul!(drstate.x[i],D,state.x[i])
    end

    # Convert between the computational r̃ coordinate
    # and the traditional r coordinate

    for i in 1:numvar
        @. drstate.x[i] /= drdr̃
    end

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
    @. Sr = ψ*Π  # Momentum source
    @. Tt = Π^2 - ψ^2/γrr - 2*(m^2)*𝜙^2  # Trace of the Stress-Energy tensor
    @. Srr = γrr*( Π^2 + ψ^2/γrr - (m^2)*𝜙^2)/2  # Radial pressure component
    @. Sθθ = γθθ*( Π^2 - ψ^2/γrr - (m^2)*𝜙^2)/2  # Angular pressure component

    # Gauge Conditions
    # Keep radius areal and keep cp constant

    # @. ᾶ  = cp*frθθ/γθθ/(frθθ-Kθθ*sqrt(γrr))
    #
    # @. βʳ = cp*Kθθ*sqrt(γrr)/(frθθ-Kθθ*sqrt(γrr))
    #
    # mul!(∂ᵣᾶ,D,ᾶ)
    # mul!(∂ᵣ2ᾶ,D,∂ᵣᾶ)
    #
    # mul!(∂ᵣβʳ,D,βʳ)
    # mul!(∂ᵣ2βʳ,D,∂ᵣβʳ)

    # Keep both ingoing and outgoing coordinate speeds of light fixed
    # @. ᾶ    = (cp-cm)/γθθ/2
    # @. ∂ᵣᾶ  = (∂ᵣcp-∂ᵣcm)/γθθ/2 - 4*frθθ*ᾶ/γθθ/2
    # @. ∂ᵣ2ᾶ = ( (∂ᵣ2cp-∂ᵣ2cm)/γθθ/2 + 7*ᾶ*frθθ^2/γθθ^2 - ᾶ*Kθθ^2*γrr/γθθ^2
    #  - 4*∂ᵣᾶ*frθθ/γθθ - 2*ᾶ*Krr*Kθθ/γθθ - 2*ᾶ*frrr*frθθ/γrr/γθθ - ᾶ*γrr/γθθ
    #  + 8*pi*ᾶ*γrr*ρ )
    #
    # @. βʳ    = -(cp+cm)/2
    # @. ∂ᵣβʳ  = -(∂ᵣcp+∂ᵣcm)/2
    # @. ∂ᵣ2βʳ = -(∂ᵣ2cp+∂ᵣ2cm)/2

    # ρ = temp.x[5]
    # @. ρ = ( Π^2 + ψ^2/γrr + (m^2)*𝜙^2)/2
    #
    # @. βʳ    = -cp + ᾶ*γθθ
    # @. ∂ᵣβʳ  = -∂ᵣcp + ∂ᵣᾶ*γθθ + 2*ᾶ*frθθ
    # @. ∂ᵣ2βʳ = ( -∂ᵣ2cp + ∂ᵣ2ᾶ*γθθ + 4*∂ᵣᾶ*frθθ + 2*ᾶ*Krr*Kθθ + 2*ᾶ*frrr*frθθ/γrr
    #     + ᾶ*γrr - 7*ᾶ*frθθ^2/γθθ + ᾶ*Kθθ^2*γrr/γθθ - 8*pi*ᾶ*γrr*γθθ*ρ )

    # Gauge condition for preventing apparent horizon formation

    # @. βʳ = ᾶ*γθθ*γrr*( 3*frθθ^2 - 2*frθθ*Kθθ*sqrt(γrr) - γrr*Kθθ^2 - 2*γθθ*∂ᵣlnᾶ*frθθ
    #  - 2*frrr*frθθ*γθθ/γrr + 2*frrr*Kθθ*γθθ/sqrt(γrr) + 2*∂ᵣlnᾶ*Kθθ*γθθ*sqrt(γrr)
    #  + γrr*γθθ - 4*pi*γθθ^2*sqrt(γrr)*Sr + 8*pi*γθθ^2*γrr*ρ )/(
    #     frθθ^2*γrr - 2*frθθ*Kθθ*sqrt(γrr)^3 + Kθθ^2*γrr^2 - 2*frθθ*Krr*γθθ*sqrt(γrr)
    #  + 2*Krr*Kθθ*γrr*γθθ + γrr^2*γθθ + 4*pi*γθθ^2*sqrt(γrr)^3*Sr - 8*pi*γθθ^2*γrr^2*ρ )
    #
    # mul!(∂ᵣβʳ,D,βʳ)
    # mul!(∂ᵣ2βʳ,D,∂ᵣβʳ)

    #Gauge condition for preventing apparent horizon formation

    # @. βʳ = ( 2*rh*vh + ᾶ*γθθ*sqrt(γrr)*Kθθ )/frθθ
    #
    # mul!(∂ᵣβʳ,D,βʳ)
    # mul!(∂ᵣ2βʳ,D,∂ᵣβʳ)

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

    @. ∂ₜγrr  = βʳ*∂ᵣγrr + 2*∂ᵣβʳ*γrr - 2*α*Krr

    @. ∂ₜγθθ  = βʳ*∂ᵣγθθ - 2*α*Kθθ

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

    #########################################################
    # Source Terms and Source Evolution
    #
    # This currently includes the addition of source terms
    # to GR that come from a Klein-Gordon scalar field
    #
    #########################################################

    # Klein-Gordon System

    @. ∂ₜ𝜙 =   βʳ*∂ᵣ𝜙 - α*Π
    @. ∂ₜψ =   βʳ*∂ᵣψ - α*∂ᵣΠ - α*(frrr/γrr - 2*frθθ/γθθ + ∂ᵣlnᾶ)*Π + ψ*∂ᵣβʳ
    @. ∂ₜΠ = ( βʳ*∂ᵣΠ - α*∂ᵣψ/γrr + α*(Krr/γrr + 2*Kθθ/γθθ)*Π
     - α*(4*frθθ/γθθ + ∂ᵣlnᾶ)*ψ/γrr + m^2*α*𝜙 )

    # Source terms to GR

    @. ∂ₜKrr  += 4*pi*α*(γrr*Tt - 2*Srr)
    @. ∂ₜKθθ  += 4*pi*α*(γθθ*Tt - 2*Sθθ)
    @. ∂ₜfrrr += 16*pi*α*γrr*Sr

    # Calculate the Apparent Horizon

    AH = temp.x[1]
    @. AH = Kθθ - frθθ/sqrt(γrr)
    is_AH = false
    for i in 1:n-1 if AH[i]*AH[i+1] <= 0. is_AH = true; break; end end

    s = 1.

    if !(is_AH)

        ## Apply Inner Boundary Conditions

        Umθ = @part 1 ( Kθθ - frθθ/sqrt(γrr) )
        Upθ = @part 1 ( Kθθ + frθθ/sqrt(γrr) )

        Umr = @part 1 ( Krr - frrr/sqrt(γrr) )
        Upr = @part 1 ( Krr + frrr/sqrt(γrr) )

        # γrrrhs = ∂ₜγrr[1]; γθθrhs = ∂ₜγθθ[1];
        # Krrrhs = ∂ₜKrr[1]; frrrrhs = ∂ₜfrrr[1];
        # Kθθrhs = ∂ₜKθθ[1]; frθθrhs = ∂ₜfrθθ[1];


        # Mode speeds

        ∂ₜUm𝜙 = @part 1 ( ∂ₜΠ - ∂ₜψ/sqrt(γrr) + ψ*∂ₜγrr/2/sqrt(γrr)^3 )

        Up𝜙   = @part 1 ( Π + ψ/sqrt(γrr) )
        Um𝜙   = @part 1 ( Π - ψ/sqrt(γrr) )

        # Dirichlet
        ∂ₜUp𝜙 = @part 1 -(∂ₜUm𝜙*cm/cp + Um𝜙*(2*ᾶ*βʳ*∂ₜγθθ/cp^2))

        # Neumann
        # ∂ₜUp𝜙 = @part 1 -∂ₜUm𝜙*cm/cp
        #∂ₜUp𝜙 = 0.

        #∂ₜUp𝜙 = 0.

        # ∂ₜψ[1]  += s*sqrt(γrr[1])*(Up𝜙b - Up𝜙)/(dr̃*σ00)/2.
        # ∂ₜΠ[1]  += s*(Up𝜙b - Up𝜙)/(dr̃*σ00)/2.
        #∂ₜ𝜙[1]  += s*(0. - 𝜙[1])/(dr̃*σ00)

        #∂ₜ𝜙[1] = 0.
        # ∂ₜψ[1] += Πrhs/cp
        # ∂ₜΠ[1] = 0.

        γrrrhs = ∂ₜγrr[1]; γθθrhs = ∂ₜγθθ[1];
        Krrrhs = ∂ₜKrr[1]; frrrrhs = ∂ₜfrrr[1];
        Kθθrhs = ∂ₜKθθ[1]; frθθrhs = ∂ₜfrθθ[1];
        Πrhs = ∂ₜΠ[1]; ψrhs = ∂ₜψ[1];

        # @part 1 ∂ₜΠ = ∂ₜUp𝜙/2 + Πrhs/2 - ψrhs/sqrt(γrr)/2 + ψ*γrrrhs/4/sqrt(γrr)^3
        # @part 1 ∂ₜψ = ψrhs/2 + ∂ₜUp𝜙*sqrt(γrr)/2 - Πrhs*sqrt(γrr)/2 + ψ*γrrrhs/4/γrr

        ∂ᵣUmθ = @part 1 ∂ᵣKθθ - ∂ᵣfrθθ/sqrt(γrr) + frθθ*(2*frrr - 8*frθθ*γrr/γθθ)/(2*sqrt(γrr)^3)

        Upθb = @part 1 ((2*M*sqrt(γθθ) - γθθ)/Umθ)

        #Dirichlet on scalar
        Up𝜙b = @part 1 -sqrt((cm*Upθb)/(cp*Umθ))*Um𝜙

        #Uprb = @part 1 (-Umr - γrr*Umθ/γθθ - (2*∂ᵣUmθ*sqrt(γrr) + γrr)/Umθ )

        Uprb = @part 1 Krri + frrri/sqrt(γrri)

        #Dirichlet on r-mode
        #Uprb = @part 1 (cm/cp)*(Umr-(Krri - frrri/sqrt(γrri))) + Krri + frrri/sqrt(γrri)

        ∂ₜΠ[1] += s*(Up𝜙b - Up𝜙)/(dr̃*Σ[1,1])/2.
        ∂ₜψ[1] += s*sqrt(γrr[1])*(Up𝜙b - Up𝜙)/(dr̃*Σ[1,1])/2.

        ∂ₜKrr[1]  += s*(Uprb - Upr)/(dr̃*Σ[1,1])/2.
        ∂ₜfrrr[1] += s*sqrt(γrr[1])*(Uprb - Upr)/(dr̃*Σ[1,1])/2.

        ∂ₜKθθ[1]  += s*(Upθb - Upθ)/(dr̃*Σ[1,1])/2.
        ∂ₜfrθθ[1] += s*sqrt(γrr[1])*(Upθb - Upθ)/(dr̃*Σ[1,1])/2.

        #Define boundary condition
        # Dirichlet condition keeps areal radius constant.
        # Upθ = @part 1 Umθ*cm/cp
        #
        # ∂ₜUmr = @part 1 ∂ₜKrr - ∂ₜfrrr/sqrt(γrr) + frrr*∂ₜγrr/2/sqrt(γrr)^3

        #∂ₜUpr = @part 1 ∂ₜUmr*cm/cp + Umr*(2*ᾶ*βʳ*∂ₜγθθ/cp^2)

        # ∂ₜUpr = 0
        #
        # #Dirichlet on r-mode
        # #Uprb = @part 1 (cm/cp)*Umr
        #
        # @part 1 ∂ₜKrr = ∂ₜUpr/2 + Krrrhs/2 - frrrrhs/sqrt(γrr)/2 + frrr*γrrrhs/4/sqrt(γrr)^3
        # @part 1 ∂ₜfrrr = frrrrhs/2 + ∂ₜUpr*sqrt(γrr)/2 - Krrrhs*sqrt(γrr)/2 + frrr*γrrrhs/4/γrr
        #
        # # ∂ₜKrr[1]  += s*(Uprb - Upr[1])/(dr̃*Σ11)/2.
        # # ∂ₜfrrr[1] += s*sqrt(γrr[1])*(Uprb - Upr[1])/(dr̃*Σ11)/2.
        #
        # ∂ᵣUpθ = @part 1 ( (Umr + Upr)*Upθ/2/sqrt(γrr) + (1. + Upθ^2/γθθ)*sqrt(γrr)/2
        #     - 4*pi*sqrt(γrr)*γθθ*(ρ + Sr/sqrt(γrr)) )
        #
        # ∂ₜUpθ = @part 1 ( α - cp*∂ᵣUpθ + Umr*Upθ*α/γrr + (Upθ - Umθ)*Upθ*α/γθθ
        #     - α*∂ᵣlnᾶ*Upθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )
        #
        # @part 1 ∂ₜKθθ  = ∂ₜUpθ/2 + Kθθrhs/2 - frθθrhs/sqrt(γrr)/2 + frθθ*γrrrhs/4/sqrt(γrr)^3
        # @part 1 ∂ₜfrθθ = frθθrhs/2 + ∂ₜUpθ*sqrt(γrr)/2 - Kθθrhs*sqrt(γrr)/2 + frθθ*γrrrhs/4/γrr

    end

    ## Outer

    Umθ = @part n ( Kθθ - frθθ/sqrt(γrr) )
    Upθ = @part n ( Kθθ + frθθ/sqrt(γrr) )

    Umr = @part n ( Krr - frrr/sqrt(γrr) )
    Upr = @part n ( Krr + frrr/sqrt(γrr) )

    γrrrhs = ∂ₜγrr[n]; γθθrhs = ∂ₜγθθ[n];
    Krrrhs = ∂ₜKrr[n]; frrrrhs = ∂ₜfrrr[n];
    Kθθrhs = ∂ₜKθθ[n]; frθθrhs = ∂ₜfrθθ[n];
    Πrhs = ∂ₜΠ[n]; ψrhs = ∂ₜψ[n];


    dtU0r = @part n ( (2*frrr - 8*frθθ*γrr/γθθ)*βʳ + 2*∂ᵣβʳ*γrr - 2*α*Krr )
    dtU0θ = @part n ( 2*frθθ*βʳ - 2*α*Kθθ )

    # dtU0r = 0.
    # dtU0θ = 0.

    ∂ₜγrr[n] = dtU0r
    ∂ₜγθθ[n] = dtU0θ

    #∂ₜUmr = ∂ₜKrr[n] - ∂ₜfrrr[n]/sqrt(γrr[n]) + frrr[n]*∂ₜγrr[n]/2/sqrt(γrr[n])^3
    #∂ₜUmr = 0. + 4*pi*α[n]*(γrr[n]*Tt[n] - 2*Srr[n]) - 16*pi*α[n]*sqrt(γrr[n])*Sr[n]

    ∂ₜUmr = 0.
    #∂ₜUmr = @part n ∂ₜKrr - ∂ₜfrrr/sqrt(γrr) + frrr*∂ₜγrr/2/sqrt(γrr)^3

    @part n ∂ₜKrr  = ∂ₜUmr/2 + Krrrhs/2 + frrrrhs/sqrt(γrr)/2 - frrr*γrrrhs/4/sqrt(γrr)^3
    @part n ∂ₜfrrr = (frrrrhs/2 - ∂ₜUmr*sqrt(γrr)/2 + Krrrhs*sqrt(γrr)/2
     - frrr*γrrrhs/4/γrr + frrr*dtU0r/2/γrr)

    ∂ᵣUmθ = @part n ( -(Umr + Upr)*Umθ/2/sqrt(γrr) - (1. + Umθ^2/γθθ)*sqrt(γrr)/2
        + 4*pi*sqrt(γrr)*γθθ*(ρ - Sr/sqrt(γrr)) )

    ∂ₜUmθ = @part n ( α - cm*∂ᵣUmθ + Upr*Umθ*α/γrr
        - (Upθ - Umθ)*Umθ*α/γθθ + α*∂ᵣlnᾶ*Umθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )

    #∂ₜUmθ = 0.

    @part n ∂ₜKθθ  = ∂ₜUmθ/2 + Kθθrhs/2 + frθθrhs/sqrt(γrr)/2 - frθθ*γrrrhs/4/sqrt(γrr)^3
    @part n ∂ₜfrθθ = (frθθrhs/2 - ∂ₜUmθ*sqrt(γrr)/2 + Kθθrhs*sqrt(γrr)/2
     - frθθ*γrrrhs/γrr/4 + frθθ*dtU0r/γrr/2)

    Um𝜙 = @part n ( Π - ψ/sqrt(γrr) )
    Up𝜙 = @part n ( Π + ψ/sqrt(γrr) )
    U0𝜙 = @part n ( 𝜙 )

    # Neumann
    #Um𝜙b = @part n -Up𝜙*cp/cm

    #∂ₜ𝜙[n]  += s*(U0𝜙b - U0𝜙)/(dr̃*σ00)/2
    @part n ( ∂ₜ𝜙 = βʳ*ψ - α*Π )

    #∂ₜ𝜙[n]  += s*(U0𝜙b - U0𝜙)/(dr̃*σ00)
    # ∂ₜψ[n]  += -s*sqrt(γrr[n])*(Um𝜙b - Um𝜙)/(dr̃*σ00)/2.
    # ∂ₜΠ[n]  += s*(Um𝜙b - Um𝜙)/(dr̃*σ00)/2.

    #∂ₜUp𝜙 = @part n -(∂ₜUm𝜙*cm/cp)
    ∂ₜUm𝜙 = 0.

    @part n ∂ₜΠ = ∂ₜUm𝜙/2 + Πrhs/2 + ψrhs/sqrt(γrr)/2 - ψ*γrrrhs/4/sqrt(γrr)^3
    @part n ∂ₜψ = ψrhs/2 - ∂ₜUm𝜙*sqrt(γrr)/2 + Πrhs*sqrt(γrr)/2 - ψ*γrrrhs/4/γrr

    # Dirichlet
    #Um𝜙b = Up𝜙*cp/cm
    #U0𝜙b = 0.

    # ∂ₜKrr[n]  = 0.
    # ∂ₜfrrr[n] = 0.
    # ∂ₜKθθ[n]  = 0.
    # ∂ₜfrθθ[n] = 0.

    # Store the calculated state into the param
    # so that we can print it to the screen

    for i in 1:numvar
        dtstate2.x[i] .= dtstate.x[i]
    end

    # Convert back to regularized variables

    for i in reg_list
        @. dtstate.x[i] /= init_state.x[i]
    end

    # Add the numerical dissipation to dtstate

    for i in 1:numvar
        mul!(dtstate.x[i],D4,regstate.x[i],1,1)
        # syntax is equivalent to dtstate.x[i] .+= D4*regstate.x[i]
    end

    # for i in 1:numvar-3
    #     @. dtstate.x[i] = 0.
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

    m = 0.
    M = 1.
    r = param.rsamp
    drdr̃ = param.drdr̃samp
    d2rdr̃ = param.d2rdr̃samp
    temp = param.temp
    grid = param.grid

    # for i in reg_list
    #     @. state.x[i] /= init_state.x[i]
    # end

    deriv!(∂ᵣγrr,γrr,n,dr̃)
    deriv!(∂ᵣγθθ,γθθ,n,dr̃)
    deriv!(∂ᵣKθθ,Kθθ,n,dr̃)
    deriv!(∂ᵣfrθθ,frθθ,n,dr̃)
    deriv!(∂ᵣ𝜙,𝜙,n,dr̃)

    ∂ᵣγrr ./= drdr̃
    ∂ᵣγθθ ./= drdr̃
    ∂ᵣKθθ ./= drdr̃
    ∂ᵣfrθθ ./= drdr̃
    ∂ᵣ𝜙 ./= drdr̃

    reg = temp.x[1]; ∂reg = temp.x[2];

    for i in reg_list
        @. reg = state.x[i]; @. ∂reg = drstate.x[i];
        @. state.x[i] *= init_state.x[i]
        @. drstate.x[i] = (init_state.x[i]*∂reg
              + init_drstate.x[i]*reg )
    end

    α = temp.x[3]; ρ = temp.x[4]; Sr = temp.x[5]

    @. α = ᾶ*γθθ*sqrt(γrr)
    @. ρ = (Π^2 + ψ^2/γrr + (m^2)*𝜙^2)/2
    #Lower Index
    @. Sr = ψ*Π
    # @. ρ = ( (Π - βʳ*ψ)^2/α^2 + ψ^2/γrr + (m^2)*𝜙^2)/2
    # #Lower Index
    # @. Sr = -ψ*(Π - βʳ*ψ)/α

    Er = zeros(T,n)
    #; norm = ones(T,n);
    # norm[1] = 17/48; norm[2] = 59/48; norm[3] = 43/48; norm[4] = 49/48;
    # norm[n] = 17/48; norm[n-1] = 59/48; norm[n-2] = 43/48; norm[n-3] = 49/48;
    # norm[1] = 1/2; norm[n] = 1/2;

    #@. Er = norm*sqrt(γrr)*γθθ*(α*ρ - βʳ*Sr)*drdr̃

    Σ = sparse(Diagonal(fill(1.,n)))
    Σ[1:5,1:5] .= inv(Σil); Σ[n-4:n,n-4:n] .= inv(Σir);

    @. Er = 4*pi*(ρ - βʳ*Sr/α)*α*sqrt(γrr)*γθθ*drdr̃

    Er .= Σ*Er

    E = 0
    for i in 1:n
        E += dr̃*Er[i]
    end

    # Constraint Equations

    𝓗 = zeros(T,n); 𝓜r = zeros(T,n); Crrr = zeros(T,n); Crθθ = zeros(T,n);
    C𝜙 = zeros(T,n);

    @. 𝓗 = (∂ᵣfrθθ/(γθθ*γrr) + 7*frθθ^2/(2*γrr*γθθ^2) - frrr*frθθ/(γrr^2*γθθ)
     - Kθθ^2/(2*γθθ^2) - 1/(2*γθθ) - Krr*Kθθ/(γrr*γθθ) + 4*pi*ρ)

    @. 𝓜r = (∂ᵣKθθ/γθθ - frθθ*Kθθ/γθθ^2 - frθθ*Krr/(γθθ*γrr) + 4*pi*Sr)

    @. Crrr = ∂ᵣγrr + 8*frθθ*γrr/γθθ - 2*frrr

    @. Crθθ = ∂ᵣγθθ - 2*frθθ

    @. C𝜙 = ∂ᵣ𝜙 - ψ

    return [𝓗, 𝓜r, Crrr, Crθθ, C𝜙, E]

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


function solution_saver(T,grid,sol,param,folder)

    ###############################################
    # Saves all of the variables in nice CSV files
    # in the choosen data folder directory
    ###############################################

    path = string("data/",folder,"/","r-D_s-D")

    mkpath(path);

    old_files = readdir(path; join=true)
    for i in 1:length(old_files)
        rm(old_files[i])
    end

    vars = (["γrr","γθθ","Krr","Kθθ","frrr","frθθ","𝜙","ψ","Π",
    "∂ₜγrr","∂ₜγθθ","∂ₜKrr","∂ₜKθθ","∂ₜfrrr","∂ₜfrθθ","∂ₜ𝜙","∂ₜψ",
    "∂ₜΠ","H","Mr","Crrr","Crθθ","C𝜙","E","appHorizon"])
    varlen = length(vars)
    #mkdir(string("data\\",folder))
    tlen = size(sol)[2]
    rlen = grid.ncells
    r = param.rsamp
    r̃min = param.r̃min

    init_state = param.init_state
    init_drstate = param.init_drstate

    dtstate = [rhs_all(sol[i],param,0.) for i = 1:tlen]

    cons = [constraints(sol[i],param) for i = 1:tlen]

    array = Array{T,2}(undef,tlen+1,rlen+1)

    array[1,1] = 0.
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
            string(path,"/",vars[j],".csv"),
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
            string(path,"/",vars[j+numvar],".csv"),
            DataFrame(array, :auto),
            header=false
        )

    end

    for j = 1:5

        for i = 2:tlen+1
            array[i,1] = sol.t[i-1]
            @. array[i,2:end] = cons[i-1][j]
        end

        CSV.write(
            string(path,"/",vars[j+2*numvar],".csv"),
            DataFrame(array, :auto),
            header=false
        )

    end

    for j = 6:6

        for i = 2:tlen+1
            array[i,1] = sol.t[i-1]
            array[i,2] = cons[i-1][j]
            @. array[i,3:end] = 0.
        end

        CSV.write(
            string(path,"/",vars[j+2*numvar],".csv"),
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
    # string("data/",folder,"/","E-",r̃min,".csv"),
    # DataFrame(array, :auto),
    # header=false
    # )

end

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

    #rspan = T[r̃min,r̃max*10.]
    # f(x) = x*tan((r̃max-r̃min)/x) + r̃min - rspan[2]
    #
    # rs = find_zero(f, 0.64*r̃max)
    #
    # r(r̃) = rs*tan((r̃-r̃min)/rs) + r̃min
    # drdr̃(r̃) = sec((r̃-r̃min)/rs)^2
    # d2rdr̃(r̃) = (2/rs)*(sec((r̃-r̃min)/rs)^2)*tan((r̃-r̃min)/rs)

    r(r̃) = r̃
    drdr̃(r̃) = 1.
    d2rdr̃(r̃) = 0.

    domain = Domain{T}(r̃min, r̃max)
    grid = Grid(domain, n)

    tmin, tmax = tspan

    atol = eps(T)^(T(3) / 4)

    alg = RK4()

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

    Mtot = 1.

    param = Param(
    r̃min,r̃max,Mtot,grid,
    r,drdr̃,d2rdr̃,
    rsamp,drdr̃samp,d2rdr̃samp,gauge,speeds,
    init_state,init_drstate,
    state,drstate,
    dtstate,dissipation,temp)

    init!(regstate, param)

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

    solution_saver(T,grid,sol,param,folder)

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
