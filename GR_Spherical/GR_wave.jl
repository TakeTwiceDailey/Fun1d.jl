module GR_Spherical_Wave

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

using PrettyTables

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
const numvar = 3
const metricvars = 8

# Type to store all of the grid functions for the ODE Solver
VarContainer{T} = ArrayPartition{T, NTuple{numvar, Vector{T}}}
MetricContainer{T} = ArrayPartition{T, NTuple{metricvars, Vector{T}}}

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
    metric::MetricContainer{T}
    temp::MetricContainer{T}
    state::VarContainer{T}
    drstate::VarContainer{T}
    dtstate::VarContainer{T}
end

# Defines how to allocate the grid functions
@inline function Base.similar(::Type{VarContainer},::Type{T},size::Int) where T
    return ArrayPartition([similar(Vector{T}(undef,size)) for i=1:numvar]...)::VarContainer{T}
end

# Defines how to allocate the grid functions
@inline function Base.similar(::Type{MetricContainer},::Type{T},size::Int) where T
    return ArrayPartition([similar(Vector{T}(undef,size)) for i=1:metricvars]...)::MetricContainer{T}
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
# the set (ᾶ, βʳ, γrr, γθθ) as functions of (M,r,r̃),
# All of the other variables are fixed automatically based on definitions

#Kerr-Schild Coordinates
# sign=1 for ingoing (black hole), sign=-1 for outgoing (white hole)

sign = 1.

fᾶ(M,r) = 1/(r^2 + 2*M(r)*r)
fβʳ(M,r) = sign*2*M(r)/(2*M(r)+r)
fγrr(M,r) = 1 + 2*M(r)/r
fγθθ(M,r) = r^2

fᾶ(M::Number,r) = 1/(r^2+2*M*r)
fβʳ(M::Number,r) = sign*2*M/(2*M+r)
fγrr(M::Number,r) = 1 + 2*M/r

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
# fᾶ(M,r) = sqrt(1. - 2*M(r̃)/r(r̃))
# fβʳ(M,r) = 0.
# fγrr(M,r) = 1/(1 - 2*M(r̃)/r(r̃))
# fγθθ(M,r) = r(r̃)^2
#
# fᾶ(M::Number,r,r̃) = sqrt(1. - 2*M/r(r̃))
# fγrr(M::Number,r,r̃) = 1/(1 - 2*M/r(r̃))

# Cartesian Minkowski

# fᾶ(M,r) = 1.
# fβʳ(M,r) = 0.
# fγrr(M,r) = 1.
# fγθθ(M,r) = 1.

# fᾶ(M::Number,r) = 1.
# fβʳ(M::Number,r) = 0.
# fγrr(M::Number,r) = 1.

# Define derivatives, extrinsic curavture, and the f_{ijk} variables

fα(M,r) = fᾶ(M,r)*fγθθ(M,r)*sqrt(fγrr(M,r))

fcp(M,r) = -fβʳ(M,r) + fα(M,r)/sqrt(fγrr(M,r))
fcm(M,r) = -fβʳ(M,r) - fα(M,r)/sqrt(fγrr(M,r))

f∂ᵣᾶ(M,r)   = ForwardDiff.derivative(r -> fᾶ(M,r), r)
f∂ᵣβʳ(M,r)  = ForwardDiff.derivative(r -> fβʳ(M,r),  r)
f∂ᵣγrr(M,r) = ForwardDiff.derivative(r -> fγrr(M,r), r)
f∂ᵣγθθ(M,r) = ForwardDiff.derivative(r -> fγθθ(M,r), r)

fKrr(M,∂ₜγrr,r) = -(∂ₜγrr(M,r) - fβʳ(M,r)*f∂ᵣγrr(M,r) - 2*fγrr(M,r)*f∂ᵣβʳ(M,r))/(2*fα(M,r))
fKθθ(M,∂ₜγθθ,r) = -(∂ₜγθθ(M,r) - fβʳ(M,r)*f∂ᵣγθθ(M,r))/(2*fα(M,r))
ffrθθ(M,r) = f∂ᵣγθθ(M,r)/2
ffrrr(M,r) = (f∂ᵣγrr(M,r) + 8*fγrr(M,r)*ffrθθ(M,r)/fγθθ(M,r))/2

# f∂ᵣKrr(M,∂ₜγrr,r)   = ForwardDiff.derivative(r -> fKrr(M,∂ₜγrr,r), r)
# f∂ᵣfrrr(M,r)       = ForwardDiff.derivative(r -> ffrrr(M,r), r)
# f∂ᵣKθθ(M,∂ₜγθθ,r)   = ForwardDiff.derivative(r -> fKθθ(M,∂ₜγθθ,r), r)
# f∂ᵣfrθθ(M,r)       = ForwardDiff.derivative(r -> ffrθθ(M,r), r)

f∂ᵣ𝜙(M,r) = ForwardDiff.derivative(r -> f𝜙(M,r), r)

# fΠ(M,r) = f∂ₜ𝜙(M,r)
# fψr(M,r) = ((fα(M,r)^2 - fγrr(M,r)*fβʳ(M,r)^2)*f∂ᵣ𝜙(M,r)/fγrr(M,r) + fβʳ(M,r)*f∂ₜ𝜙(M,r))/fα(M,r)

fψr(M,r) = f∂ᵣ𝜙(M,r)
fΠ(M,r) = -(f∂ₜ𝜙(M,r) - fβʳ(M,r)*fψr(M,r) )/fα(M,r)

function init!(state::VarContainer{T}, param::Param{T}) where T

    ############################################
    # Specifies the Initial Conditions
    ############################################

    metric = param.metric
    temp = param.temp

    𝜙,ψr,Π = state.x
    α,βʳ,γrr,γθθ,frrr,frθθ,Krr,Kθθ = metric.x
    ∂ᵣᾶ = temp.x[1]
    ∂ᵣβʳ = temp.x[2]

    grid = param.grid

    global Mtot = copy(M0)

    f∂ₜγrri(M,r) = 0.
    f∂ₜγθθi(M,r) = 0.

    M(r) = M0

    # Sample the state initial vector
    sample!(γrr,    grid, r -> fγrr(M,r)                )
    sample!(γθθ,    grid, r -> fγθθ(M,r)                )
    sample!(frrr,   grid, r -> ffrrr(M,r)               )
    sample!(frθθ,   grid, r -> ffrθθ(M,r)               )
    sample!(Krr,    grid, r -> fKrr(M,f∂ₜγrri,r)         )
    sample!(Kθθ,    grid, r -> fKθθ(M,f∂ₜγθθi,r)         )
    sample!(𝜙,      grid, r -> f𝜙(M,r)                 )
    sample!(ψr,     grid, r -> fψr(M,r)                 )
    sample!(Π,      grid, r -> fΠ(M,r)                  )

    # Sample the gauge variables
    sample!(α,      grid, r -> fα(M,r)                  )
    sample!(βʳ,     grid, r -> fβʳ(M,r)                 )

    sample!(∂ᵣᾶ,    grid, r -> f∂ᵣᾶ(M,r)                )
    sample!(∂ᵣβʳ,   grid, r -> f∂ᵣβʳ(M,r)               )

    # global Up𝜙i1 = @part 1 ( Π + ψr/sqrt(γrr) )
    # global Um𝜙i1 = @part 1 ( Π - ψr/sqrt(γrr) )
    # global Up𝜙in = @part n ( Π + ψr/sqrt(γrr) )
    # global Um𝜙in = @part n ( Π - ψr/sqrt(γrr) )

    global ψri1 = @part 1 ψr
    global ψrin = @part n ψr

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

    rhs!(param.dtstate, state, param, 0.)

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
    rmin = grid.domain.rmin
    rmax = grid.domain.rmax

    state = param.state
    drstate = param.drstate
    dtstate2 = param.dtstate
    temp = param.temp
    #gauge = param.gauge
    metric = param.metric

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

    𝜙,ψr,Π = state.x
    ∂ᵣ𝜙,∂ᵣψr,∂ᵣΠ = drstate.x
    ∂ₜ𝜙,∂ₜψr,∂ₜΠ = dtstate.x
    α,βʳ,γrr,γθθ,frrr,frθθ,Krr,Kθθ = metric.x

    ∂ᵣᾶ = temp.x[1]; ∂ᵣβʳ = temp.x[2]; ᾶ = temp.x[3]; 

    @. ᾶ = α/γθθ/sqrt(γrr)

    # Calculate first spatial derivatives by multipling D operator
    # and convert between the computational r̃ coordinate
    # and the traditional r coordinate

    mul!(∂ᵣ𝜙,D,𝜙)
    mul!(∂ᵣψr,D,ψr)
    mul!(∂ᵣΠ,D,Π)

    ∇ᵣψr = temp.x[4]

    ∇ᵣψr .= (D*(sqrt.(γrr).*γθθ.*ψr  ))./(sqrt.(γrr).*γθθ)

    # Klein-Gordon System

    # @. ∂ₜ𝜙 = Π 
    # #+ (α^2 - γrr*βʳ^2)*∂ᵣ𝜙/γrr + βʳ*Π - α*ψr

    # @. ∂ₜψr =  βʳ*∇ᵣψr + α*∂ᵣΠ/γrr - βʳ*m^2*α*𝜙

    # @. ∂ₜΠ  = α*∇ᵣψr + βʳ*∂ᵣΠ - m^2*α*𝜙

    @. ∂ₜ𝜙 = βʳ*∂ᵣ𝜙 - α*Π
    #@. ∂ₜ𝜙 = βʳ*ψr - α*Π

    @. ∂ₜψr =  βʳ*∂ᵣψr - α*∂ᵣΠ - α*(frrr/γrr - 2*frθθ/γθθ + ∂ᵣᾶ/ᾶ)*Π + ∂ᵣβʳ*ψr 

    @. ∂ₜΠ = ( βʳ*∂ᵣΠ - α*∇ᵣψr/γrr + α*(Krr/γrr + 2*Kθθ/γθθ)*Π
     + α*(frrr/γrr - 6*frθθ/γθθ - ∂ᵣᾶ/ᾶ)*ψr/γrr + m^2*α*𝜙 )

    # Calculate the advection speeds

    # @. cp = -βʳ + ᾶ*γθθ
    # @. cm = -βʳ - ᾶ*γθθ

    # Inner Boundary Conditions

    # Um𝜙  = @part 1 ( Π + ψr*sqrt(γrr) )
    # Up𝜙  = @part 1 ( Π - ψr*sqrt(γrr) )


    # # Dirichlet
    # Up𝜙b = -Um𝜙

    # cp = @part 1 ( -βʳ + α/sqrt(γrr) )

    # s1 = cp/Σ[1,1]
    # # if (1-Π[1]/Up𝜙) == NaN
    # #     s1 = cp*(1-Π[1]/Up𝜙)/Σ[1,1]
    # # else
    # #     s1 = cp/Σ[1,1]
    # # end
    

    # ∂ₜΠ[1] += s1*(Up𝜙b - Up𝜙)/2
    # ∂ₜψr[1] += -s1*(Up𝜙b - Up𝜙)/2/sqrt(γrr[1])

######################################

    cp = @part 1 ( -βʳ + α/sqrt(γrr) )
    cm = @part 1 ( -βʳ - α/sqrt(γrr) )

    Up𝜙   = @part 1 ( Π + ψr/sqrt(γrr) )
    Um𝜙   = @part 1 ( Π - ψr/sqrt(γrr) )
    
    # Static Dirichlet
    Up𝜙b = @part 1 (cm/cp)*Um𝜙
    
    s1 = cp/Σ[1,1]

    ∂ₜΠ[1] += s1*(Up𝜙b - Up𝜙)/2
    ∂ₜψr[1] += s1*sqrt(γrr[1])*(Up𝜙b - Up𝜙)/2

######################################

    # dtUm𝜙 = @part 1 ( ∂ₜΠ - ∂ₜψr/sqrt(γrr) )

    # cp = @part 1 ( -βʳ + α/sqrt(γrr) )
    # cm = @part 1 ( -βʳ - α/sqrt(γrr) )

    # dtUp𝜙b = (cm/cp)*dtUm𝜙

    # ∂ₜΠ[1] = (dtUp𝜙b + dtUm𝜙)/2
    # ∂ₜψr[1] = sqrt(γrr[1])*(dtUp𝜙b - dtUm𝜙)/2


    ## Outer Boundary Conditions

    # Um𝜙 = @part n ( Π + ψr*sqrt(γrr) )
    # Up𝜙 = @part n ( Π - ψr*sqrt(γrr) )

    # # Neumann
    # Um𝜙b = @part n Up𝜙

    # cm = @part n ( -βʳ - α/sqrt(γrr) )

    # sn = -cm/Σ[n,n]
    # # if (Π[1]/Um𝜙) == NaN
    # #     sn = -cm*(Π[1]/Up𝜙)/Σ[n,n]
    # # else
    # #     sn = -cm/Σ[n,n]
    # # end

    # ∂ₜΠ[n] += sn*(Um𝜙b - Um𝜙)/2
    # ∂ₜψr[n] += sn*(Um𝜙b - Um𝜙)/2/sqrt(γrr[n])

    #@part n ∂ₜ𝜙 = Π

    cp = @part n ( -βʳ + α/sqrt(γrr) )
    cm = @part n ( -βʳ - α/sqrt(γrr) )

    Up𝜙 = @part n ( Π + ψr/sqrt(γrr) )
    Um𝜙 = @part n ( Π - ψr/sqrt(γrr) )

     # Static Neumann
    Um𝜙b = @part n -(cp/cm)*Up𝜙
    #Um𝜙b = Up𝜙 - 2*ψrin/sqrt(γrr[n])

    sn = -cm/Σ[n,n]

    ∂ₜΠ[n] += sn*(Um𝜙b - Um𝜙)/2
    ∂ₜψr[n] += -sn*sqrt(γrr[n])*(Um𝜙b - Um𝜙)/2

    #∂ₜΠ[n] += sn*((-Π[n]) - Π[n])/2
   # ∂ₜψr[n] += sn*(0. - ψr[n])/2

    @part n ∂ₜ𝜙 = (βʳ*ψr - α*Π)

    # dtUp𝜙 = @part n ( ∂ₜΠ + ∂ₜψr/sqrt(γrr) )

    # cp = @part n ( -βʳ + α/sqrt(γrr) )
    # cm = @part n ( -βʳ - α/sqrt(γrr) )

    # dtUm𝜙b = -(cp/cm)*dtUp𝜙

    # ∂ₜΠ[n] = (dtUp𝜙 + dtUm𝜙b)/2
    # ∂ₜψr[n] = sqrt(γrr[n])*(dtUp𝜙 - dtUm𝜙b)/2

    # Store the calculated state into the param
    # so that we can print it to the screen

    for i in 1:numvar
        dtstate2.x[i] .= dtstate.x[i]
    end

    # catch any errors, save them to print later
    catch e
        global_error.error = e
        global_error.stacktrace = stacktrace(catch_backtrace())
    end

end

function rhs_all(regstate::VarContainer{T}, param::Param{S}, t) where {S,T}

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
    metric = param.metric

    α,βʳ,γrr,γθθ,frrr,frθθ = metric.x

    for i in 1:numvar
        state.x[i] .= regstate.x[i]
    end

    𝜙,ψr,Π = state.x
    ∂ᵣ𝜙,∂ᵣψr,∂ᵣΠ = drstate.x

    for i in 1:numvar
        mul!(drstate.x[i],D,state.x[i])
    end

    # Constraint Equations

    C𝜙 = zeros(T,n);

    # @. C𝜙 = (α^2 - γrr*βʳ^2)*∂ᵣ𝜙/γrr + βʳ*Π - α*ψr

    @. C𝜙 = ∂ᵣ𝜙 - ψr

    Γ = spdiagm(sqrt.(γrr).*γθθ)
    W = Σ*Γ;

    # E = sum(W*(@. (α*Π^2 + α*γrr*ψr^2 - 2*γrr*ψr*βʳ*Π )/(α^2 - γrr*βʳ^2) + α*m^2*𝜙^2 ))

    E = (α.*Π)'*W*Π/2. +  (α.*ψr./γrr)'*W*ψr/2. - (βʳ.*Π)'*W*ψr

    return [C𝜙, E]

end

function solution_saver(T,sol,param)

    ###############################################
    # Saves all of the variables in nice CSV files
    # in the choosen data folder directory
    ###############################################

    folder = string("Static_n=",      n,
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
    
    vars = (["phi","psi","Pi","Cphi","E"])

    varlen = length(vars)
    tlen = size(sol)[2]
    grid = param.grid

    #dtstate = [rhs_all(sol[i],param,0.) for i = 1:tlen]

    cons = [constraints(sol[i],param) for i = 1:tlen]

    array = Array{T,2}(undef,tlen,n)

    r = zeros(T,n); sample!(r, grid, r -> r );

    save(string(path,"/coords.h5"), Dict("r"=>r,"t"=>sol.t[:]) )

    for j = 1:numvar

        for i = 1:tlen
            @. array[i,:] = sol[i].x[j]
        end
        save(string(path,"/",vars[j],".h5"), Dict(vars[j]=>array ) )

    end

    for j = 1:1
        for i = 1:tlen @. array[i,:] = cons[i][j] end
        save(string(path,"/",vars[j+numvar],".h5"), Dict(vars[j+numvar]=>array ) )
    end

    for j = 2:2
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
function error_handler(regstate,t,integrator)
    return !(typeof(global_error.error) == NothingException)
end

# Terminates the integrator when error_handler returns true
error_cb = DiscreteCallback(error_handler,terminate!,save_positions=(false,false))


function print_times(regstate,t,integrator)
    t in tspan[1]:print_interval:tspan[2]
end

function continuous_print(integrator)

    ###############################################
    # Outputs status numbers while the program runs
    ###############################################

    dtstate = integrator.p.dtstate

    ∂ₜ𝜙,∂ₜψr,∂ₜΠ = dtstate.x

    println("| ",
    rpad(string(round(integrator.t,digits=1)),5," "),"|   ",
    rpad(string(round(integrator.dt,digits=4)),8," "),"|   ",
    rpad(string(round(maximum(abs.(∂ₜ𝜙)), digits=3)),8," "),"|   ",
    rpad(string(round(maximum(abs.(∂ₜψr)), digits=3)),8," "),"|   ",
    rpad(string(round(maximum(abs.(∂ₜΠ)), digits=3)),8," "),"|"
    )

    return

end

function initial_print(param::Param)

    ###############################################
    # Outputs initial status numbers
    ###############################################

    dtstate = param.dtstate

    ∂ₜ𝜙,∂ₜψr,∂ₜΠ = dtstate.x

    println("| ",
    rpad(0.0,5," "),"|   ",
    rpad(0.0,8," "),"|   ",
    rpad(string(round(maximum(abs.(∂ₜ𝜙)), digits=4)),8," "),"|   ",
    rpad(string(round(maximum(abs.(∂ₜψr)), digits=3)),8," "),"|   ",
    rpad(string(round(maximum(abs.(∂ₜΠ)), digits=3)),8," "),"|"
    )

    return

end

print_cb = DiscreteCallback(print_times,continuous_print,save_positions=(false,false))

cb = CallbackSet(error_cb,print_cb)

function main()

    ###############################################
    # Main Program
    #
    # Calls each of the above functions to run a
    # simulation. Sets up the numerical grid,
    # sets the gauge conditions, sets the 
    # initial conditions,
    # and finally runs the numerical DiffEq
    # package to run the time integration.
    #
    # All data is saved in the folder specified to
    # the solution_saver, each in their own HDF5
    # file.
    ###############################################

    rmin, rmax = rspan

    domain = Domain{T}(rmin, rmax)
    grid = Grid(domain, n)

    #atol = eps(T)^(T(3) / 4)
    atol = 10e-12

    alg = RK4()
    #alg = Trapezoid(nlsolve=NLAnderson())
    #alg = ImplicitMidpoint(autodiff=false)
    #alg = SSPRK54()

    #printlogo()

    regstate = similar(VarContainer,T,n)
    
    state = similar(VarContainer,T,n)
    drstate = similar(VarContainer,T,n)
    dtstate = similar(VarContainer,T,n)

    metric = similar(MetricContainer,T,n)
    temp = similar(MetricContainer,T,n)

    #println("Defining Problem...")

    param = Param(grid,metric,temp,state,drstate,dtstate)

    init!(regstate, param)

    prob = ODEProblem(rhs!, regstate, tspan, param)

    #println("Starting Solution...")
    println("")
    println("| Time |    dt     | max ∂ₜ𝜙   | max ∂ₜψr  | max ∂ₜΠ   |")
    println("|______|___________|___________|___________|___________|")
    println("|      |           |           |           |           |")

    initial_print(param)

    # println(typeof(regstate))
    # println(typeof(param.state))

    #return

    sol = solve(
        prob, alg;
        # abstol = atol,
        # reltol = atol,
        # dtmax = dt,
        dt = dt,
        adaptive = false,
        saveat = save_interval,
        tstops = [t for t in tspan[1]:print_interval:tspan[2]],
        callback = cb
    )

    solution_saver(T,sol,param)

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

