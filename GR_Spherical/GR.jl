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
using SparseArrays
using LinearAlgebra
using BandedMatrices

using Distributions

using ForwardDiff

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
    r̃min::T
    r̃max::T
    dr̃::T
    Mtot::T
    grid::Grid{T}
    reg_list::Vector{Int64}
    r::Function
    drdr̃::Function
    d2rdr̃::Function
    rsamp::Vector{T}
    drdr̃samp::Vector{T}
    d2rdr̃samp::Vector{T}
    gauge::VarContainer{T}
    init_state::VarContainer{T}
    init_drstate::VarContainer{T}
#    init_dr2state::VarContainer{T}
    state::VarContainer{T}
    drstate::VarContainer{T}
    dtstate::VarContainer{T}
    dissipation::VarContainer{T}
    temp::VarContainer{T}
    B2::SparseMatrixCSC{T, Int64}
    D2::SparseMatrixCSC{T, Int64}
    Σi::SparseMatrixCSC{T, Int64}
end

struct test{T}
    n::T
    Mat::Array
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

r0 = 5.
σr = 0.1
Amp = 0.00000

fᾶ(M,r,r̃) = 1/(r(r̃)^2 + 2*M(r̃)*r(r̃)) + Amp*exp(-(1/2)*((r(r̃)-r0)/σr)^2)
f∂r̃ᾶ(M,r,r̃) = ForwardDiff.derivative(r̃ -> fᾶ(M,r,r̃), r̃)
f∂r̃2ᾶ(M,r,r̃) = ForwardDiff.derivative(r̃ -> f∂r̃ᾶ(M,r,r̃), r̃)

fβr(M,r,r̃) = 2*M(r̃)/(2*M(r̃)+r(r̃))
f∂r̃βr(M,r,r̃) = ForwardDiff.derivative(r̃ -> fβr(M,r,r̃), r̃)
f∂r̃2βr(M,r,r̃) = ForwardDiff.derivative(r̃ -> f∂r̃βr(M,r,r̃), r̃)

fγrr(M,r,r̃) = 1 + 2*M(r̃)/r(r̃)
f∂r̃γrr(M,r,r̃) = ForwardDiff.derivative(r̃ -> fγrr(M,r,r̃), r̃)
f∂r̃2γrr(M,r,r̃) = ForwardDiff.derivative(r̃ -> f∂r̃γrr(M,r,r̃), r̃)

fγθθ(M,r,r̃) = r(r̃)^2
f∂r̃γθθ(M,r,r̃) = ForwardDiff.derivative(r̃ -> fγθθ(M,r,r̃), r̃)
f∂r̃2γθθ(M,r,r̃) = ForwardDiff.derivative(r̃ -> f∂r̃γθθ(M,r,r̃), r̃)

fKrr(M,∂rM,r,r̃) = (2*(r(r̃)*∂rM(r̃)-M(r̃))/r(r̃)^3)*(r(r̃)+M(r̃))/sqrt((1+2*M(r̃)/r(r̃)))
f∂r̃Krr(M,∂rM,r,r̃) = ForwardDiff.derivative(r̃ -> fKrr(M,∂rM,r,r̃), r̃)
f∂r̃2Krr(M,∂rM,r,r̃) = ForwardDiff.derivative(r̃ -> f∂r̃Krr(M,∂rM,r,r̃), r̃)

fKθθ(M,r,r̃) = 2*M(r̃)/sqrt((1+2*M(r̃)/r(r̃)))
f∂r̃Kθθ(M,r,r̃) = ForwardDiff.derivative(r̃ -> fKθθ(M,r,r̃), r̃)
f∂r̃2Kθθ(M,r,r̃) = ForwardDiff.derivative(r̃ -> f∂r̃Kθθ(M,r,r̃), r̃)

ffrrr(M,∂rM,r,r̃) = (7*M(r̃) + (4 + ∂rM(r̃))*r(r̃))/(r(r̃)^2)
f∂r̃frrr(M,∂rM,r,r̃) = ForwardDiff.derivative(r̃ -> ffrrr(M,∂rM,r,r̃), r̃)
f∂r̃2frrr(M,∂rM,r,r̃) = ForwardDiff.derivative(r̃ -> f∂r̃frrr(M,∂rM,r,r̃), r̃)

ffrθθ(M,r,r̃) = r(r̃)
f∂r̃frθθ(M,r,r̃) = ForwardDiff.derivative(r̃ -> ffrθθ(M,r,r̃), r̃)
f∂r̃2frθθ(M,r,r̃) = ForwardDiff.derivative(r̃ -> f∂r̃frθθ(M,r,r̃), r̃)

function init!(state::VarContainer{T}, param) where T

    ############################################
    # Specifies the Initial Conditions
    ############################################

    init_state = param.init_state
    init_drstate = param.init_drstate
    gauge = param.gauge

    γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,ψ,Π = state.x
    ᾶ,βr,∂rᾶ,∂rβr,∂r2ᾶ,∂r2βr,∂r3βr,∂r4βr,∂r5βr = gauge.x
    γrri,γθθi,Krri,Kθθi,frrri,frθθi,𝜙i,ψi,Πi = init_state.x
    ∂rγrr,∂rγθθ,∂rKrr,∂rKθθ,∂rfrrr,∂rfrθθ,∂r𝜙,∂rψ,∂rΠ = init_drstate.x

    grid = param.grid
    dr̃ = spacing(grid)
    r = param.r
    drdr̃ = param.drdr̃
    d2rdr̃ = param.d2rdr̃
    r̃min = param.r̃min
    r̃max = param.r̃max
    reg_list = param.reg_list

    n = grid.ncells
    m = 0.
    r̃span = (r̃min,r̃max)

    # Mass (no real reason not to use 1 here)
    #M = 1

    #min = 5
    # f𝜙(M,r,r̃) = 0.
    # fψ(M,r,r̃) = 0.
    # fΠ(M,r,r̃) = 0.
    # f𝜙(r̃) = Amp*(1/r(r̃))*exp(-(1/2)*((r(r̃)-r0)/σr)^2)
    # fψ(r̃) = Amp*exp(-(1/2)*((r(r̃)-r0)/σr)^2)*(r(r̃)*r0-r(r̃)^2-σr^2)/(r(r̃)^2*σr^2)
    # fΠ(r̃) = 0.

    r0 = 6.
    σr = 0.1
    Amp = 0.

    Fᾶ(M,r,r̃) = 1/(r(r̃)^2+2*(1)*r(r̃))
    Fβr(M,r,r̃) = 2*(1)/(2*(1)+r(r̃))
    Fγrr(M,r,r̃) = 1 + 2*M/r(r̃)
    Fγθθ(M,r,r̃) = r(r̃)^2

    f𝜙(M,r,r̃) = Amp*(1/r(r̃))*exp(-(1/2)*((r(r̃)-r0)/σr)^2)
    fψ(M,r,r̃) = Amp*exp(-(1/2)*((r(r̃)-r0)/σr)^2)*(r(r̃)*r0-r(r̃)^2-σr^2)/(r(r̃)^2*σr^2)
    fΠ(M,r,r̃) = Fβr(M,r,r̃)*fψ(M,r,r̃)

    fρ(M,r,r̃) = ((fΠ(M,r,r̃) - Fβr(M,r,r̃)*fψ(M,r,r̃))^2/(Fᾶ(M,r,r̃)^2*Fγθθ(M,r,r̃)^2*Fγrr(M,r,r̃))
        + fψ(M,r,r̃)^2/Fγrr(M,r,r̃) + m^2*f𝜙(M,r,r̃)^2)/2

    f∂r̃M(M,r,r̃) = 4*pi*r(r̃)^2*fρ(M,r,r̃)*drdr̃(r̃)

    # Constraint Equations

    function constraintSystem(M, param, r̃)
        r = param.r
        f∂r̃M(M,r,r̃)
    end

    BVP = ODEProblem(constraintSystem, 1., r̃span, param)
    M = solve(BVP, Tsit5(), abstol=1e-15, dt=dr̃, adaptive=false)

    #∂rM(r̃) = 4*pi*r(r̃)^2*fρ(M(r̃),r,r̃)
    M(r̃) = 1.
    ∂rM(r̃) = 0.

    sample!(γrri,   grid, r̃ -> fγrr(M,r,r̃)      )
    sample!(γθθi,   grid, r̃ -> fγθθ(M,r,r̃)      )
    sample!(Krri,   grid, r̃ -> fKrr(M,∂rM,r,r̃)  )
    sample!(Kθθi,   grid, r̃ -> fKθθ(M,r,r̃)      )
    sample!(frrri,  grid, r̃ -> ffrrr(M,∂rM,r,r̃) )
    sample!(frθθi,  grid, r̃ -> ffrθθ(M,r,r̃)     )
    sample!(𝜙i,     grid, r̃ -> f𝜙(M,r,r̃)        )
    sample!(ψi,     grid, r̃ -> fψ(M,r,r̃)        )
    sample!(Πi,     grid, r̃ -> fΠ(M(r̃),r,r̃)     )

    sample!(∂rγrr,  grid, r̃ -> f∂r̃γrr(M,r,r̃)/drdr̃(r̃)      )
    sample!(∂rγθθ,  grid, r̃ -> f∂r̃γθθ(M,r,r̃)/drdr̃(r̃)      )
    sample!(∂rKrr,  grid, r̃ -> f∂r̃Krr(M,∂rM,r,r̃)/drdr̃(r̃)  )
    sample!(∂rKθθ,  grid, r̃ -> f∂r̃Kθθ(M,r,r̃)/drdr̃(r̃)      )
    sample!(∂rfrrr, grid, r̃ -> f∂r̃frrr(M,∂rM,r,r̃)/drdr̃(r̃) )
    sample!(∂rfrθθ, grid, r̃ -> f∂r̃frθθ(M,r,r̃)/drdr̃(r̃)     )

    Mg(rt) = 1.

    sample!(ᾶ,     grid, r̃ -> fᾶ(Mg,r,r̃)        )
    sample!(βr,    grid, r̃ -> fβr(Mg,r,r̃)       )
    sample!(∂rᾶ,   grid, r̃ -> f∂r̃ᾶ(Mg,r,r̃)/drdr̃(r̃)        )
    sample!(∂rβr,  grid, r̃ -> f∂r̃βr(Mg,r,r̃)/drdr̃(r̃)       )
    sample!(∂r2ᾶ,  grid, r̃ -> (f∂r̃2ᾶ(Mg,r,r̃) - d2rdr̃(r̃)*f∂r̃ᾶ(Mg,r,r̃)/drdr̃(r̃))/drdr̃(r̃)^2   )
    sample!(∂r2βr, grid, r̃ -> (f∂r̃2βr(Mg,r,r̃) - d2rdr̃(r̃)*f∂r̃βr(Mg,r,r̃)/drdr̃(r̃))/drdr̃(r̃)^2 )

    𝜙i[1]=0.; 𝜙i[n]=0.; Πi[1]=0.; Πi[n]=0.;

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

    # for i in 1:numvar
    #     for j in 1:n
    #        state.x[i][j] = init_state.x[i][j] + s*rand(Uniform(-1,1))
    #     end
    #     state.x[i][1] = init_state.x[i][1]
    #     state.x[i][n] = init_state.x[i][n]
    # end

end

Σ11 = 4.186595370392226897362216859769846226369
Σ21 = 0.; Σ31 = 0.; Σ41 = 0.; Σ51 = 0.;

Σ22 = 0.6725191921225620731888714836983116420871
Σ32 = 0.3613418181134949259370502966736306984367
Σ42 = -0.2021316117293899791481674539631879662707
Σ52 = 0.03455320708729270824077678274955265350304

Σ33 = 0.7206133711630147057720442098623847362950
Σ43 = 0.1376472340546569368321616389764958792591
Σ53 = -0.04136405531324488624637892257286207044784

Σ44 = 0.9578653607931026822074133441449909689509
Σ54 = 0.02069353627247161734563597102894256809696

Σ55 = 0.9908272703370861473007798925906968380654

Σil = [ Σ11 Σ21 Σ31 Σ41 Σ51;
        Σ21 Σ22 Σ32 Σ42 Σ52;
        Σ31 Σ32 Σ33 Σ43 Σ53;
        Σ41 Σ42 Σ43 Σ44 Σ54;
        Σ51 Σ52 Σ53 Σ54 Σ55 ]

Σir = Σil[end:-1:1,end:-1:1];

q11 = -2.0932976346634987158873300;   q21 =  4.0398572053206615302160000;
q31 = -3.0597858079809922953240000;   q41 =  1.3731905386539948635493300;
q51 = -0.2599643013301653825540000;   q61 =  0.; q71 = 0.;

q12 = -0.3164158528594044527229700;   q22 = -0.5393078897398042232738800;
q32 =  0.9851773202864434338329700;   q42 = -0.0526466598929757814670900;
q52 = -0.1138072517506242350132580;   q62 =  0.0398797678898499118031030;
q72 = -0.0028794339334846531588787;

q13 =  0.1302691618502116452445200;   q23 = -0.8796685899505924925689000;
q33 =  0.3860964096110007000013400;   q43 =  0.3135836907243558874598800;
q53 =  0.0853189419136783846335110;   q63 = -0.0390466157927346402746410;
q73 =  0.0034470016440805155042908;

q14 = -0.0172451219382464791217200;   q24 =  0.1627228822712750438113400;
q34 = -0.8134981024864881302921700;   q44 =  0.1383326926647983321564500;
q54 =  0.5974385432854805339961600;   q64 = -0.0660264343462998876193240;
q74 = -0.0017244594505194129307249;

q15 = -0.0088356946855219296506100;   q25 =  0.0305607475920320385728400;
q35 =  0.0502116827453085423227800;   q45 = -0.6630736465244492953406800;
q55 =  0.0148787874640051911160880;   q65 =  0.6588270638170747195382000;
q75 = -0.0825689404084492665586150;

ql =  [ q11 q21 q31 q41 q51 q61 q71;
        q12 q22 q32 q42 q52 q62 q72;
        q13 q23 q33 q43 q53 q63 q73;
        q14 q24 q34 q44 q54 q64 q74;
        q15 q25 q35 q45 q55 q65 q75 ]

qr = -ql[end:-1:1,end:-1:1];

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

@inline function deriv2!(df::Vector{T}, f::Vector{T}, n::Int64, dx::T) where T

    #@inbounds

    df[1] = (2*f[1] - 5*f[2] + 4*f[3] - f[4])/(dx^2)

    df[2] = (f[1] - 2*f[2] + f[3])/(dx^2)

    df[3] = (-4*f[1] + 59*f[2] - 110*f[3] + 59*f[4] - 4*f[5])/(43*dx^2)

    df[4] = (-f[1] + 59*f[3] - 118*f[4] + 64*f[5] - 4*f[6])/(49*dx^2)

    for i in 5:(n - 4)
        df[i] = (-f[i-2] + 16*f[i-1] - 30*f[i] + 16*f[i+1] - f[i+2])/(12*dx^2)
    end

    df[n-3] = (-f[n] + 59*f[n-2] - 118*f[n-3] + 64*f[n-4] - 4*f[n-5])/(49*dx^2)

    df[n-2] = (-4*f[n] + 59*f[n-1] - 110*f[n-2] + 59*f[n-3] - 4*f[n-4])/(43*dx^2)

    df[n-1] = (f[n] - 2*f[n-1] + f[n-2])/(dx^2)

    df[n] = (2*f[n] - 5*f[n-1] + 4*f[n-2] - f[n-3])/(dx^2)

end

@inline function dissipation!(df::Vector{T}, f::Vector{T}, param::Param) where T

    ############################################
    # Calculates the numerical dissipation terms
    ############################################

    B2 = param.B2; D2 = param.D2; Σi = param.Σi; n = param.grid.ncells;

    # df[1:7] .= Σil*(D2l')*B2l(dr̃)*D2l*f[1:7]
    #
    # for i in 7:(n-5)
    #     df[i] = -(-f[i-2] + 4*f[i-1] - 6*f[i] + 4*f[i+1] - f[i+2])
    # end
    #
    # df[n-6:n] .= Σir*(D2r')*B2r(dr̃)*D2r*f[n-6:n]

    df .= Σi*(D2')*B2*D2*f
    #df[n] = 0.

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

    #sqrt(γrr) handler?

    # Unpack the parameters

    m = 0.
    Mtot = 1.
    M = 1.

    grid = param.grid
    dr̃ = param.dr̃
    r = param.rsamp
    drdr̃ = param.drdr̃samp
    d2rdr̃ = param.d2rdr̃samp
    r̃min = param.r̃min
    r̃max = param.r̃max
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

    n = grid.ncells

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
    ∂rγrr,∂rγθθ,∂rKrr,∂rKθθ,∂rfrrr,∂rfrθθ,∂r𝜙,∂rψ,∂rΠ = drstate.x
    ∂tγrr,∂tγθθ,∂tKrr,∂tKθθ,∂tfrrr,∂tfrθθ,∂t𝜙,∂tψ,∂tΠ = dtstate.x
    ∂4γrr,∂4γθθ,∂4Krr,∂4Kθθ,∂4frrr,∂4frθθ,∂4𝜙,∂4ψ,∂4Π = dissipation.x
    ᾶ,βr,∂rᾶ,∂rβr,∂r2ᾶ,∂r2βr,α,∂rlnᾶ,∂r2lnᾶ = gauge.x

    γrri,γθθi,Krri,Kθθi,frrri,frθθi,𝜙i,ψi,Πi = init_state.x
    ∂rγrri,∂rγθθi,∂rKrri,∂rKθθi,∂rfrrri,∂rfrθθi,∂r𝜙i,∂rψi,∂rΠi = init_drstate.x

    # for i in reg_list
    #     @. state.x[i] /= init_state.x[i]
    # end

    # Calculate first spatial derivatives

    for i in 1:numvar
        deriv!(drstate.x[i],state.x[i],n,dr̃)
    end

    # Calculate numerical dissipation

    for i in 1:numvar
        dissipation!(dissipation.x[i],state.x[i],param)
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
        @. drstate.x[i] = ∂reg*init_state.x[i] + reg*init_drstate.x[i]
    end

    # Calculated lapse and derivatives of densitized lapse

    @. α = ᾶ*γθθ*sqrt(γrr)
    @. ∂rlnᾶ = ∂rᾶ/ᾶ
    @. ∂r2lnᾶ = (∂r2ᾶ*ᾶ - ∂rᾶ^2)/ᾶ^2

    # Gauge Conditions

    # Keep radius areal

    @. βr = ᾶ*γθθ*sqrt(γrr)*Kθθ/frθθ

    @. ∂rβr = (∂rᾶ*γθθ*sqrt(γrr)*Kθθ/frθθ + ᾶ*∂rγθθ*sqrt(γrr)*Kθθ/frθθ
     + ᾶ*γθθ*sqrt(γrr)*∂rKθθ/frθθ + ᾶ*γθθ*∂rγrr*Kθθ/sqrt(γrr)/frθθ/2
     - ᾶ*γθθ*sqrt(γrr)*Kθθ*∂rfrθθ/frθθ^2)
    #deriv!(∂rβr,βr,n,dr̃)
    deriv!(∂r2βr,∂rβr,n,dr̃)

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

    Γt = temp.x[5]; Γr = temp.x[6];

    @. Γt = (βr*∂rlnᾶ - ∂rβr)/α^2
    @. Γr = 2*βr*∂rβr/α^2 - (1/γrr + (βr/α)^2)*∂rlnᾶ - 4*frθθ/(γrr*γθθ)

    @. ∂t𝜙 = Π
    @. ∂tψ = ∂rΠ
    @. ∂tΠ = (α^2)*((1/γrr-(βr/α)^2)*∂rψ + 2*(βr/α^2)*∂rΠ - Γr*ψ - Γt*Π - m^2*𝜙)

    # @. ∂t𝜙 = βr*ψ - α*Π
    # @. ∂tψ = βr*∂rψ - α*∂rΠ - α*(frrr/γrr - 2*frθθ/γθθ + ∂rlnᾶ)*Π + ψ*∂rβr
    # @. ∂tΠ = (βr*∂rΠ - α*∂rψ/γrr + α*(Krr/γrr + 2*Kθθ/γθθ)*Π
    #  - α*(4*frθθ/γθθ + ∂rlnᾶ)*ψ/γrr)#- m^2*𝜙)

    ρ = temp.x[5]; Sr = temp.x[6]; Tt = temp.x[7]; Srr = temp.x[8]; Sθθ = temp.x[9];

    @. ρ = ( (Π - βr*ψ)^2/α^2 + ψ^2/γrr + (m^2)*𝜙^2)/2
    @. Sr = -ψ*(Π - βr*ψ)/α
    @. Tt = (Π - βr*ψ)^2/α^2 - ψ^2/γrr - 2*(m^2)*𝜙^2
    @. Srr = γrr*( (Π - βr*ψ)^2/α^2 + ψ^2/γrr - (m^2)*𝜙^2)/2
    @. Sθθ = γθθ*( (Π - βr*ψ)^2/α^2 - ψ^2/γrr - (m^2)*𝜙^2)/2

    @. ∂tKrr += 4*pi*α*(γrr*Tt - 2*Srr)
    @. ∂tKθθ += 4*pi*α*(γθθ*Tt - 2*Sθθ)
    @. ∂tfrrr += 16*pi*α*γrr*Sr

    # Specify the inner and outer temporal boundary conditions
    # for metric variables

    # for i in 1:numvar
    #     dtstate.x[i][n] = 0.
    # end
    #σ00 = 17/48.
    σ00 = 0.23885757
    s = 1.

    Upθ = temp.x[8]
    Umθ = temp.x[9]

    # cp = -βr[1] + α[1]/sqrt(γrr[1])
    #
    # Crrr = ∂rγrr[1] + 8*frθθ[1]*γrr[1]/γθθ[1] - 2*frrr[1]

    @. Umθ = Kθθ - frθθ/sqrt(γrr)

    #@. Upθ = Kθθ + frθθ/sqrt(γrr)

    @. Upθ = ((-βr - α/sqrt(γrr))/(-βr + α/sqrt(γrr)))*Umθ

    #@. Upθ = (2*sqrt(γθθ) - γθθ)/Umθ

    #Upθ[1] = (2*sqrt(γθθ[1]) - γθθ[1])/Umθ[1]

    ∂tKθθ[1]  += s*( (Upθ[1] - Kθθ[1])/2 - frθθ[1]/sqrt(γrr[1])/2 )/(dr̃*σ00)
    ∂tfrθθ[1] += s*( (Upθ[1] - Kθθ[1])*sqrt(γrr[1])/2 - frθθ[1]/2 )/(dr̃*σ00)

    Umr = Krr[1] - frrr[1]/sqrt(γrr[1])
    #Upθ = Kθθ[1] + frθθ[1]/sqrt(γrr[1])

    #∂rUmθ = ∂rKθθ[1] - ∂rfrθθ[1]/sqrt(γrr[1]) + frθθ[1]*∂rγrr[1]/(2*sqrt(γrr[1])^3)
    #
    # ∂rUpθ = (1/sqrt(γθθ[1]) - 1)*∂rγθθ[1]/Umθ[1] - (2*sqrt(γθθ[1]) - γθθ[1])*∂rUmθ/Umθ[1]^2

    cp = -βr[1] + α[1]/sqrt(γrr[1])
    cm = -βr[1] - α[1]/sqrt(γrr[1])

    ∂rcp = -∂rβr[1] + ∂rᾶ[1]*γθθ[1] + ᾶ[1]*∂rγθθ[1]
    ∂rcm = -∂rβr[1] - ∂rᾶ[1]*γθθ[1] - ᾶ[1]*∂rγθθ[1]

    ∂rUmθ = ∂rKθθ[1] - ∂rfrθθ[1]/sqrt(γrr[1]) + frθθ[1]*∂rγrr[1]/(2*sqrt(γrr[1])^3)

    ∂rUpθ = cm*∂rUmθ/cp + ∂rcm*Umθ[1]/cp - cm*Umθ[1]*∂rcp/cp^2

    #∂rUpθ = (q1 ⋅ Upθ[1:7])/dr̃/drdr̃[1]
    #∂rUmθ = ∂rKθθ[1] - ∂rfrθθ[1]/sqrt(γrr[1]) + frθθ[1]*∂rγrr[1]/(2*sqrt(γrr[1])^3)
    #∂rUpθ = (1/sqrt(γθθ[1]) - 1.)*∂rγθθ[1]/Umθ[1] - (2*sqrt(γθθ[1]) - γθθ[1])*∂rUmθ/Umθ[1]^2
    #∂rUpθ = ∂rKθθ[1] + ∂rfrθθ[1]/sqrt(γrr[1]) - frθθ[1]*∂rγrr[1]/(2*sqrt(γrr[1])^3)
    #∂rUpθ = (-25*Upθ[1] + 48*Upθ[2] - 36*Upθ[3] + 16*Upθ[4] - 3*Upθ[5])/(12*dr̃)/drdr̃[1]
    #∂rUpθ = (-137*Upθ[1] + 300*Upθ[2] - 300*Upθ[3] + 200*Upθ[4] - 75*Upθ[5] + 12*Upθ[6])/(60*dr̃)/drdr̃[1]

    Upr = -Umr - γrr[1]*Upθ[1]/γθθ[1] + (2*∂rUpθ*sqrt(γrr[1]) - γrr[1])/Upθ[1]

    ∂tKrr[1]  += s*( (Upr - Krr[1])/2 - frrr[1]/sqrt(γrr[1])/2 )/(dr̃*σ00)
    ∂tfrrr[1] += s*( (Upr - Krr[1])*sqrt(γrr[1])/2 - frrr[1]/2 )/(dr̃*σ00)


    # Outer boundary


    @. Upθ = Kθθ + frθθ/sqrt(γrr)

    @. Umθ = ((-βr + α/sqrt(γrr))/(-βr - α/sqrt(γrr)))*Upθ
    #@. Umθ = Kθθ - frθθ/sqrt(γrr)

    cp = -βr[n] + α[n]/sqrt(γrr[n])
    cm = -βr[n] - α[n]/sqrt(γrr[n])

    ∂rcp = -∂rβr[n] + ∂rᾶ[n]*γθθ[n] + ᾶ[n]*∂rγθθ[n]
    ∂rcm = -∂rβr[n] - ∂rᾶ[n]*γθθ[n] - ᾶ[n]*∂rγθθ[n]

    ∂rUpθ = ∂rKθθ[n] + ∂rfrθθ[n]/sqrt(γrr[n]) - frθθ[n]*∂rγrr[n]/(2*sqrt(γrr[n])^3)
    #∂rUmθ = ∂rKθθ[n] - ∂rfrθθ[n]/sqrt(γrr[n]) + frθθ[n]*∂rγrr[n]/(2*sqrt(γrr[n])^3)

    ∂rUmθ = cp*∂rUpθ/cm + ∂rcp*Upθ[n]/cm - cp*Upθ[n]*∂rcm/cm^2

    #∂rUmθ = ∂rKθθ[n] - ∂rfrθθ[n]/sqrt(γrr[n]) + frθθ[n]*∂rγrr[n]/(2*sqrt(γrr[n])^3)

    Upr = Krr[n] + frrr[n]/sqrt(γrr[n])

    Umr = -Upr - γrr[n]*Umθ[n]/γθθ[n] - (2*∂rUmθ*sqrt(γrr[n]) + γrr[n])/Umθ[n]

    # U0r = (2*frrr[n]-∂rγrr[n])*γθθ[n]/frθθ[n]/8
    # U0θ = 8*γrr[n]*frθθ[n]/(2*frrr[n]-∂rγrr[n])

    U0r = γrr[n]
    U0θ = γθθ[n]

    #Umθ[n] = Kθθi[n] - frθθi[n]/sqrt(γrri[n])

    #Umr = Krri[n] - frrri[n]/sqrt(γrri[n])

    ∂tKθθ[n]  += s*(  (Umθ[n] - Kθθ[n])/2 + frθθ[n]/sqrt(γrr[n])/2 )/(dr̃*σ00)
    ∂tfrθθ[n] += s*( -(Umθ[n] - Kθθ[n])*sqrt(γrr[n])/2 - frθθ[n]/2 )/(dr̃*σ00)
    #+ frθθ[n]*U0r/γrr[n]/2


    # ∂tγrr[n]  += s*( U0r - γrr[n] )/(dr̃*σ00)
    # ∂tγθθ[n]  += s*( U0θ - γθθ[n] )/(dr̃*σ00)

    ∂tγrr[n] = (2*frrr[n] - 8*frθθ[n]*γrr[n]/γθθ[n])*βr[n] + 2*∂rβr[n]*γrr[n] - 2*α[n]*Krr[n]
    ∂tγθθ[n] = 0.
    #∂tγθθ[n] = 2*frθθ[n]*βr[n] - 2*α[n]*Kθθ[n]

    ∂tKrr[n]  += s*(  (Umr - Krr[n])/2 + frrr[n]/sqrt(γrr[n])/2 )/(dr̃*σ00)
    ∂tfrrr[n] += s*( -(Umr - Krr[n])*sqrt(γrr[n])/2 - frrr[n]/2 )/(dr̃*σ00)
    #+ frrr[n]*U0r/γrr[n]/2

    # ∂tγrr[n] = 0.
    # ∂tγθθ[n] = 0.
    #
    # ∂tKrr[n] = 0.
    # ∂tfrrr[n] = 0.
    #
    # ∂tKθθ[n] = 0.
    # ∂tfrθθ[n] = 0.

    # ∂tKrr[n] += s*( (frrr[n]-frrri[n])/sqrt(γrr[n]) )/(dr̃*σ00)
    # ∂tKθθ[n] += s*( -(frθθi[n]-frθθ[n])/(2*sqrt(γrr[n])) + (Kθθi[n]-Kθθ[n])/2 )/(dr̃*σ00)
    # ∂tfrrr[n] += s*( -(frrr[n] - frrri[n]) )/(dr̃*σ00)
    # ∂tfrθθ[n] += s*( (frθθi[n]-frθθ[n])/2 - sqrt(γrr[n])*(Kθθi[n]-Kθθ[n])/2 )/(dr̃*σ00)

    # ∂tγrr[n] += s*( -γrr[n] )/(dr̃*σ00)
    # ∂tγθθ[n] += s*( -γθθ[n] )/(dr̃*σ00)
    # ∂tKrr[n] += s*( frrr[n]/sqrt(γrr[n]) )/(dr̃*σ00)
    # ∂tKθθ[n] += s*( frθθ[n]/sqrt(γrr[n]) )/(dr̃*σ00)
    # ∂tfrrr[n] += s*( -frrr[n] )/(dr̃*σ00)
    # ∂tfrθθ[n] += s*( -frθθ[n] )/(dr̃*σ00)

    # ∂tψ[n] += s*( Π[n]/cm )/(dr̃*σ00)
    # ∂tΠ[n] += s*( -Π[n] )/(dr̃*σ00)

    # ∂tKθθ =  α + α*Krr*Kθθ/γrr - 2*α*frθθ^2/(γrr*γθθ) - α*∂rfrθθ/γrr
    #  - α*frθθ*∂rlnᾶ/γrr + frθθ*Kθθ*βr/γθθ + frθθ*Krr*βr/γrr - 4*pi*γθθ*βr*Sr
    #
    # ∂tfrθθ =  ∂rβr*frθθ - α*∂rKθθ - α*frrr*Kθθ/γrr + 2*α*frθθ*Kθθ/γθθ
    #  - α*Kθθ*∂rlnᾶ - 7*βr*frθθ^2/(2*γθθ) + βr*frrr*frθθ/γrr
    #  + βr*Kθθ^2*γrr/(2*γθθ) + βr*γrr/2 + βr*Krr*Kθθ - 4*pi*βr*γθθ*γrr*ρ

    # Convert back to regularized variables

    for i in reg_list
        @. dtstate.x[i] /= init_state.x[i]
    end

    # Add the numerical dissipation to the regularized state

    σ = 1.
    for i in 1:numvar
        @. dtstate.x[i] -= σ*dissipation.x[i]/16.
    end

    # for i in 1:numvar-3
    #     @. dtstate.x[i] = 0.
    # end

    # Store the calculated state into the param
    # so that we can print it to the screen

    for i in 1:numvar
        dtstate2.x[i] .= dtstate.x[i]
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
    ∂rγrr,∂rγθθ,∂rKrr,∂rKθθ,∂rfrrr,∂rfrθθ,∂r𝜙,∂rψ,∂rΠ = drstate.x
    ᾶ,βr,∂rᾶ,∂rβr,∂r2ᾶ,∂r2βr,∂r3βr,∂r4βr,∂r5βr = gauge.x

    init_state = param.init_state
    init_drstate = param.init_drstate

    m = 0.
    M = 1.
    n = param.grid.ncells
    dr̃ = param.dr̃
    r = param.rsamp
    drdr̃ = param.drdr̃samp
    d2rdr̃ = param.d2rdr̃samp
    temp = param.temp
    grid = param.grid
    reg_list = param.reg_list

    # for i in reg_list
    #     @. state.x[i] /= init_state.x[i]
    # end

    deriv!(∂rKθθ,Kθθ,n,dr̃)
    deriv!(∂rfrθθ,frθθ,n,dr̃)
    deriv!(∂r𝜙,𝜙,n,dr̃)

    ∂rKθθ ./= drdr̃
    ∂rfrθθ ./= drdr̃
    ∂r𝜙 ./= drdr̃

    reg = temp.x[1]; ∂reg = temp.x[2];

    for i in reg_list
        @. reg = state.x[i]; @. ∂reg = drstate.x[i];
        @. state.x[i] *= init_state.x[i]
        @. drstate.x[i] = ∂reg*init_state.x[i] + reg*init_drstate.x[i]
    end

    α = temp.x[3]; ρ = temp.x[4]; Sr = temp.x[5]

    @. α = ᾶ*γθθ*sqrt(γrr)
    # @. ρ = (Π^2 + ψ^2/γrr + (m^2)*𝜙^2)/2
    # #Lower Index
    # @. Sr = ψ*Π
    @. ρ = ( (Π - βr*ψ)^2/α^2 + ψ^2/γrr + (m^2)*𝜙^2)/2
    #Lower Index
    @. Sr = -ψ*(Π - βr*ψ)/α

    Er = zeros(T,n); norm = ones(T,n);
    norm[1] = 17/48; norm[2] = 59/48; norm[3] = 43/48; norm[4] = 49/48;
    norm[n] = 17/48; norm[n-1] = 59/48; norm[n-2] = 43/48; norm[n-3] = 49/48;
    # norm[1] = 1/2; norm[n] = 1/2;

    #@. Er = norm*sqrt(γrr)*γθθ*(α*ρ - βr*Sr)*drdr̃

    Σ = sparse(Diagonal(fill(1.,n)))
    Σ[1:5,1:5] .= inv(Σil); Σ[n-4:n,n-4:n] .= inv(Σir);

    @. Er = (sqrt(γrr)*γθθ*
        (2*Krr^2 + 2*Kθθ^2 + 2*frrr^2/γrr + 2*frθθ^2/γrr + γrr^2 + γθθ^2)*drdr̃)

    Er .= Σ*Er

    E = 0
    for i in 1:n
        E += dr̃*Er[i]
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
    rlen = grid.ncells
    r = param.rsamp
    r̃min = param.r̃min
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
    # string("data/",folder,"/","E-",r̃min,".csv"),
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

    for i = 9:9


        T = Float64

        #r̃span = T[2.,22.] .+ (1.0 - 0.1*i)
        r̃span = T[3.5,9.0]
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

        println("Mirror: ",r̃min)

        domain = Domain{T}(r̃min, r̃max)
        grid = Grid(domain, points)

        n = grid.ncells

        dr̃ = spacing(grid)
        dt = dr̃/4.

        tspan = T[0., 6.]
        tmin, tmax = tspan

        printtimes = 0.1

        v = 1.

        m = 0.

        Mtot = 1.

        # γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,, = state.x
        #reg_list = Int64[]
        #reg_list = [2]
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
        drdr̃samp = similar(Vector{T}(undef,n))
        d2rdr̃samp = similar(Vector{T}(undef,n))

        sample!(rsamp, grid, r̃ -> r(r̃) )
        sample!(drdr̃samp, grid, r̃ -> drdr̃(r̃) )
        sample!(d2rdr̃samp, grid, r̃ -> d2rdr̃(r̃) )

        Bvec = ones(T,n)
        Bvec[1:5] .= [5*dr̃,(1+4*dr̃),(2+3*dr̃),(3+2*dr̃),(4+dr̃)]/5.
        Bvec[n:-1:n-4] .= Bvec[1:5]
        B2 = sparse(Diagonal(Bvec))

        D2 = sparse(Tridiagonal(fill(1.,n-1),fill(-2.,n),fill(1.,n-1)))
        D2[1,1:3] .= D2[2,1:3]; D2[n,n-2:n] .= D2[n-1,n-2:n];

        Σi = sparse(Diagonal(fill(1.,n)))
        Σi[1:5,1:5] .= Σil; Σi[n-4:n,n-4:n] .= Σir;

        param = Param(
        r̃min,r̃max,dr̃,Mtot,grid,reg_list,
        r,drdr̃,d2rdr̃,
        rsamp,drdr̃samp,d2rdr̃samp,gauge,
        init_state,init_drstate,
        state,drstate,
        dtstate,dissipation,temp,
        B2,D2,Σi)

        init!(regstate, param)

        prob = ODEProblem(rhs!, regstate, tspan, param)

        #println("Starting Solution...")

        println("")
        println("| Time | max ∂tγrr | max ∂tγθθ | max ∂tKrr | max ∂tKθθ | max ∂tfrrr | max ∂tfrθθ |")
        println("|______|___________|___________|___________|___________|____________|____________|")
        println("")


        sol = solve(
            prob, alg,
            abstol = atol,
            dt = dt,
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
