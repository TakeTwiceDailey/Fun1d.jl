module GR_Spherical_wave

using DifferentialEquations
using DiffEqGPU
using DataFrames
using CSV
using CUDA

using SparseArrays
using InteractiveUtils
using RecursiveArrayTools
using LinearAlgebra
using ForwardDiff

numvar = 3

USEGPU = false

if USEGPU
    Vec{T} = CuArray{T,1}
    save{T} = CuArray{T,2}
    Mat{T} = CUDA.CUSPARSE.CuSparseMatrixCSC{T}
    smat(x) = CUDA.CUSPARSE.CuSparseMatrixCSC(x)
else
    Vec{T} = Vector{T}
    save{T} = Array{T,2}
    Mat{T} = SparseMatrixCSC{T, Int64}
    smat(x) = x
end

VarContainer{T} = ArrayPartition{T, NTuple{numvar, Vec{T}}}

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
    dr̃::T
    Mtot::T
    grid::Grid{T}
    reg_list::Vector{Int64}
    r::Function
    drdr̃::Function
    d2rdr̃::Function
    rsamp::Vec{T}
    drdr̃samp::Vec{T}
    d2rdr̃samp::Vec{T}
    metric_vars::ArrayPartition{T, NTuple{6, Vec{T}}}
    gauge_vars::ArrayPartition{T, NTuple{4, Vec{T}}}
    temp::VarContainer{T}
    init_state::VarContainer{T}
    init_drstate::VarContainer{T}
    state::VarContainer{T}
    drstate::VarContainer{T}
    dtstate::VarContainer{T}
    D::Mat{T}
    D4::Mat{T}
end

@inline function Base.similar(::Type{ArrayPartition},::Type{T},size::Int) where T
    return ArrayPartition([similar(Vec{T}(undef,size)) for i=1:numvar]...)::VarContainer{T}
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

function sample!(f::Vec{T}, grid::Grid{S}, fun) where {S,T}

    dr̃ = spacing(grid)
    rtmin = grid.domain.xmin

    f .= Vec{T}(T[fun(rtmin + dr̃*(j-1)) for j in 1:(grid.ncells)])

end

sign = 1.

fᾶ(M,r,r̃) = 1/(r(r̃)^2 + 2*M(r̃)*r(r̃))
fβr(M,r,r̃) = sign*2*M(r̃)/(2*M(r̃)+r(r̃))
fγrr(M,r,r̃) = 1 + 2*M(r̃)/r(r̃)
fγθθ(M,r,r̃) = r(r̃)^2
fα(M,r,r̃) = fᾶ(M,r,r̃)*fγθθ(M,r,r̃)*sqrt(fγrr(M,r,r̃))
fKrr(M,∂rM,r,r̃) = sign*(2*(r(r̃)*∂rM(r̃)-M(r̃))/r(r̃)^3)*(r(r̃)+M(r̃))/sqrt(1+2*M(r̃)/r(r̃))
fKθθ(M,r,r̃) = sign*2*M(r̃)/sqrt((1+2*M(r̃)/r(r̃)))
ffrrr(M,∂rM,r,r̃) = (7*M(r̃) + (4 + ∂rM(r̃))*r(r̃))/(r(r̃)^2)
ffrθθ(M,r,r̃) = r(r̃)

fᾶ(M::Number,r,r̃) = 1/(r(r̃)^2+2*M*r(r̃))
fβr(M::Number,r,r̃) = sign*2*M/(2*M+r(r̃))
fγrr(M::Number,r,r̃) = 1 + 2*M/r(r̃)
fKrr(M::Number,∂rM::Number,r,r̃) = sign*(2*(r(r̃)*∂rM-M)/r(r̃)^3)*(r(r̃)+M)/sqrt(1+2*M/r(r̃))
ffrrr(M::Number,∂rM::Number,r,r̃) = (7*M + (4 + ∂rM)*r(r̃))/(r(r̃)^2)

f∂r̃ᾶ(M,r,r̃)         = ForwardDiff.derivative(r̃ -> fᾶ(M,r,r̃), r̃)
f∂r̃βr(M,r,r̃)        = ForwardDiff.derivative(r̃ -> fβr(M,r,r̃), r̃)

function init!(state::VarContainer{T}, param) where T

    ############################################
    # Specifies the Initial Conditions
    ############################################

    init_state = param.init_state
    init_drstate = param.init_drstate
    rtmin = param.rtmin
    rtmax = param.rtmax
    drdr̃ = param.drdr̃

    𝜙,ψ,Π = state.x
    𝜙i,ψi,Πi = init_state.x

    grid = param.grid
    dr̃ = spacing(grid)
    r = param.r

    n = grid.ncells
    m = 1.
    rtspan = (rtmin,rtmax)

    # Mass (no real reason not to use 1 here)
    M = 1

    r0 = 5.
    σr = 0.1
    A = 0.1

    cp(M,r,r̃) = -fβr(M,r,r̃) + fα(M,r,r̃)/sqrt(fγrr(M,r,r̃))
    cm(M,r,r̃) = -fβr(M,r,r̃) - fα(M,r,r̃)/sqrt(fγrr(M,r,r̃))

    f𝜙(M,r,r̃) = (r0-σr)<r(r̃)<(r0+σr) ? (A/r(r̃))*(r(r̃)-(r0-σr))^4*(r(r̃)-(r0+σr))^4/σr^8 : 0.
    fψ(M,r,r̃) = (r0-σr)<r(r̃)<(r0+σr) ? (A/r(r̃)^2)*((r(r̃)-r0)^2-σr^2)^3*((r(r̃)-r0)*(7*r(r̃)+r0)+σr^2)/σr^8 : 0.
    f∂t𝜙(M,r,r̃) = (r0-σr)<r(r̃)<(r0+σr) ? -(8*A*cm(M,r,r̃)/r(r̃))*((r(r̃)-r0)^2-σr^2)^3*(r(r̃)-r0)/σr^8 : 0.
    fΠ(M,r,r̃) = -(f∂t𝜙(M,r,r̃) - fβr(M,r,r̃)*fψ(M,r,r̃) )/fα(M,r,r̃)

    γrr,γθθ,Krr,Kθθ,frrr,frθθ = param.metric_vars.x
    ᾶ,βr,∂rᾶ,∂rβr = param.gauge_vars.x

    M0(r̃) = 1.
    ∂rM0(r̃) = 0.

    sample!(γrr,   grid, r̃ -> fγrr(M0,r,r̃)                 )
    sample!(γθθ,   grid, r̃ -> fγθθ(M0,r,r̃)                 )
    sample!(Krr,   grid, r̃ -> fKrr(M0,∂rM0,r,r̃)            )
    sample!(Kθθ,   grid, r̃ -> fKθθ(M0,r,r̃)                 )
    sample!(frrr,  grid, r̃ -> ffrrr(M0,∂rM0,r,r̃)           )
    sample!(frθθ,  grid, r̃ -> ffrθθ(M0,r,r̃)                )

    Mg(r̃) = M0(r̃)

    sample!(ᾶ,      grid, r̃ -> fᾶ(Mg,r,r̃)                  )
    sample!(βr,     grid, r̃ -> fβr(Mg,r,r̃)                 )
    sample!(∂rᾶ,    grid, r̃ -> f∂r̃ᾶ(Mg,r,r̃)/drdr̃(r̃)        )
    sample!(∂rβr,   grid, r̃ -> f∂r̃βr(Mg,r,r̃)/drdr̃(r̃)       )

    sample!(𝜙i, grid,r̃ -> f𝜙(M,r,r̃) )
    sample!(ψi, grid, r̃ -> fψ(M,r,r̃) )
    sample!(Πi, grid, r̃ -> fΠ(M,r,r̃) )

    for i in 1:numvar
        state.x[i] .= init_state.x[i]
    end

end


Σ11 =  4.186595370392226897362216859769846226369
Σ21 = 0.; Σ31 = 0.; Σ41 = 0.; Σ51 = 0.;

Σ22 =  0.6725191921225620731888714836983116420871
Σ32 =  0.3613418181134949259370502966736306984367
Σ42 = -0.2021316117293899791481674539631879662707
Σ52 =  0.03455320708729270824077678274955265350304

Σ33 =  0.7206133711630147057720442098623847362950
Σ43 =  0.1376472340546569368321616389764958792591
Σ53 = -0.04136405531324488624637892257286207044784

Σ44 =  0.9578653607931026822074133441449909689509
Σ54 =  0.02069353627247161734563597102894256809696

Σ55 =  0.9908272703370861473007798925906968380654

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

    #grid = param.grid
    #dr̃ = param.dr̃
    #r = param.rsamp
    #drdr̃ = param.drdr̃samp
    #d2rdr̃ = param.d2rdr̃samp
    #rtmin = param.rtmin
    #rtmax = param.rtmax
    #reg_list = param.reg_list

    #fr = param.r

    state = param.state
    drstate = param.drstate
    dtstate2 = param.dtstate
    D = param.D
    D4 = param.D4

    init_state = param.init_state
    init_drstate = param.init_drstate
    temp = param.temp

    #n = grid.ncells

    # Give names to individual variables

    𝜙,ψ,Π = state.x
    ∂r𝜙,∂rψ,∂rΠ = drstate.x
    ∂t𝜙,∂tψ,∂tΠ = dtstate.x

    𝜙i,ψi,Πi = init_state.x
    ∂r𝜙i,∂rψi,∂rΠi = init_drstate.x

    γrr,γθθ,Krr,Kθθ,frrr,frθθ = param.metric_vars.x
    ᾶ,βr,∂rᾶ,∂rβr = param.gauge_vars.x


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

    # Dirichlet boundary conditions on scalar field

    # 𝜙[1] = 0.
    # Π[1] = 0.

    # Calculate first spatial derivatives

    mul!(∂r𝜙,D,𝜙)
    mul!(∂rψ,D,ψ)
    mul!(∂rΠ,D,Π)

    α = temp.x[1]
    @. α = ᾶ*γθθ*sqrt(γrr)

    # Klein-Gordon System

    @. ∂t𝜙 =  βr*∂r𝜙 - α*Π
    @. ∂tψ =  βr*∂rψ - α*∂rΠ - α*(frrr/γrr - 2*frθθ/γθθ + ∂rᾶ/ᾶ)*Π + ψ*∂rβr
    @. ∂tΠ = (βr*∂rΠ - α*∂rψ/γrr + α*(Krr/γrr + 2*Kθθ/γθθ)*Π
     - α*(4*frθθ/γθθ + ∂rᾶ/ᾶ)*ψ/γrr)

    # σ00 = 17/48.
    # s = 1.

    # ∂tψ[1] += s*( Π[1] )/(dr̃*σ00)
    # ∂tΠ[1] += s*(0. - Π[1])/(dr̃*σ00)
    # ∂t𝜙[1] += s*(0. - 𝜙[1])/(dr̃*σ00)
    #
    # ∂tψ[n] += s*( -Π[n] )/(dr̃*σ00)
    # ∂tΠ[n] += s*(0. - Π[n])/(dr̃*σ00)
    # ∂t𝜙[n] += s*(0. - 𝜙[n])/(dr̃*σ00)


    # Add the numerical dissipation to the regularized state
    for i in 1:numvar
        mul!(dtstate.x[i],D4,state.x[i],1,1)
        #dtstate.x[i] .+= D4*state.x[i]
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

    for i in 1:numvar
        state.x[i] .= regstate.x[i]
    end

    𝜙,ψ,Π = state.x
    ∂r𝜙,∂rψ,∂rΠ = drstate.x

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


    Er = zeros(T,n); norm = ones(T,n);
    norm[1] = 17/48; norm[2] = 59/48; norm[3] = 43/48; norm[4] = 49/48;
    norm[n] = 17/48; norm[n-1] = 59/48; norm[n-2] = 43/48; norm[n-3] = 49/48;
    norm = Vec{T}(norm); Er = Vec{T}(Er);
    # norm[1] = 1/2; norm[n] = 1/2;

    #@. Er = norm*sqrt(γrr)*γθθ*(α*ρ - βr*Sr)*drdr̃

    #@. Er = norm*(r^2)*( Π^2 + ψ^2)/2
    @. Er = norm*( Π^2 + ψ^2)/2

    E = 0
    for i in 1:n
        E += dr̃*Er[i]
    end

    # Constraint Equations

    return E

end

function custom_progress_message(dt,state::VarContainer{T},param,t) where T

    ###############################################
    # Outputs status numbers while the program runs
    ###############################################

    dtstate = param.dtstate::VarContainer{T}

    ∂t𝜙,∂tψ,∂tΠ = dtstate.x

    println("  ",
    rpad(string(round(t,digits=1)),10," "),
    rpad(string(round(maximum(abs.(∂t𝜙)), digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂tψ)), digits=3)),12," "),
    rpad(string(round(maximum(abs.(∂tΠ)), digits=3)),12," ")
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

    vars = (["𝜙","ψ","Π","∂t𝜙","∂tψ","∂tΠ","E"])
    varlen = length(vars)
    #mkdir(string("data\\",folder))
    tlen = size(sol)[2]
    rlen = grid.ncells
    r = param.rsamp
    rtmin = param.rtmin
    reg_list = param.reg_list

    init_state = param.init_state
    init_drstate = param.init_drstate

    dtstate = [rhs_all(sol[i],param,0.) for i = 1:tlen]

    cons = [constraints(sol[i],param) for i = 1:tlen]

    array = save{T}(undef,tlen+1,rlen+1)

    array[1,1] = 0
    array[1,2:end] .= r

    for j = 1:numvar

        for i = 2:tlen+1
            array[i,1] = sol.t[i-1]
            @. array[i,2:end] = sol[i-1].x[j]
        end

        CSV.write(
            string("data/",folder,"/",vars[j],".csv"),
            DataFrame(array, :auto),
            header=false
        )

    end

    for j = 1:numvar

        for i = 2:tlen+1
            array[i,1] = sol.t[i-1]
            @. array[i,2:end] = dtstate[i-1].x[j]
        end

        CSV.write(
            string("data/",folder,"/",vars[j+numvar],".csv"),
            DataFrame(array, :auto),
            header=false
        )

    end

    for i = 2:tlen+1
        array[i,1] = sol.t[i-1]
        array[i,2] = cons[i-1]
        @. array[i,3:end] = 0.
    end

    CSV.write(
        string("data/",folder,"/",vars[1+2*numvar],".csv"),
        DataFrame(array, :auto),
        header=false
    )


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
        rtspan = T[3.0,7.0]
        rtmin, rtmax = rtspan
        rspan = T[rtmin,rtmax]

        r(rt) = rt
        drdr̃(rt) = 1.
        d2rdr̃(rt) = 0.

        USEGPU ? println("GPU") : println("CPU")

        domain = Domain{T}(rtmin, rtmax)
        grid = Grid(domain, points)

        n = grid.ncells

        dr̃ = spacing(grid)
        dt = dr̃/4.

        tspan = T[0., 5.]
        tmin, tmax = tspan

        printtimes = 1.

        Mtot = T(1.)

        # γrr,γθθ,Krr,Kθθ,frrr,frθθ,𝜙,, = state.x
        reg_list = Int64[]
        #reg_list = [4,7,8]
        #reg_list = [1,2,3,4,5,6]
        #reg_list = [7,8,9,10]
        #reg_list = [10]

        atol = eps(T)^(T(3) / 4)

        #alg = Tsit5()
        alg = RK4()

        #printlogo()

        custom_progress_step = round(Int, printtimes/dt)
        step_iterator = custom_progress_step

        regstate = similar(ArrayPartition,T,n)

        state = similar(ArrayPartition,T,n)
        drstate = similar(ArrayPartition,T,n)

        init_state = similar(ArrayPartition,T,n)
        init_drstate = similar(ArrayPartition,T,n)

        dtstate = similar(ArrayPartition,T,n)
        temp = similar(ArrayPartition,T,n)

        metric_vars = ArrayPartition([similar(Vec{T}(undef,n)) for i=1:6]...)
        gauge_vars  = ArrayPartition([similar(Vec{T}(undef,n)) for i=1:4]...)

        #println("Defining Problem...")
        rsamp = similar(Vec{T}(undef,n))
        drdr̃samp = similar(Vec{T}(undef,n))
        d2rdr̃samp = similar(Vec{T}(undef,n))

        sample!(rsamp, grid, rt -> r(rt) )
        sample!(drdr̃samp, grid, rt -> drdr̃(rt) )
        sample!(d2rdr̃samp, grid, rt -> d2rdr̃(rt) )

        Dc = spdiagm(-2=>ones(T,n-2),-1=>-8*ones(T,n-1),1=>8*ones(T,n-1),2=>-ones(T,n-2))/12
        Dc[1:5,1:7] .= ql; Dc[n-4:n,n-6:n] .= qr
        D = smat(Dc)

        Bvec = ones(T,n)
        tr = round(Int64, n/20)
        for i in 1:tr
            Bvec[i] = ((i-1) + (tr-(i-1))*dr̃)/tr
        end
        Bvec[n:-1:n-(tr-1)] .= Bvec[1:tr]
        B2 = spdiagm(0=>Bvec)

        D2 = spdiagm(-1=>ones(T,n-1),0=>-2*ones(T,n),1=>ones(T,n-1))
        D2[1,1:3] .= D2[2,1:3]; D2[n,n-2:n] .= D2[n-1,n-2:n];

        Σi = spdiagm(0=>ones(T,n))
        Σi[1:5,1:5] .= Σil; Σi[n-4:n,n-4:n] .= Σir;
        #
        # Σ = diagm(0=>ones(n))
        # Σ[1:5,1:5] .= inv(Σil); Σ[n-4:n,n-4:n] .= inv(Σir);
        # Σ = smat(Σ)

        σ = T(0.5/16)

        D4 = smat(-σ*Σi*(D2')*B2*D2)

        param = Param(
        rtmin,rtmax,dr̃,Mtot,grid,reg_list,
        r,drdr̃,d2rdr̃,
        rsamp,drdr̃samp,d2rdr̃samp,
        metric_vars, gauge_vars,temp,
        init_state,init_drstate,
        state,drstate,dtstate,
        D,D4)

        init!(regstate, param)

        prob = ODEProblem(rhs!, regstate, tspan, param)

        #return
        #println("Starting Solution...")
        #return CUDA.@time rhs!(dtstate,regstate,param,0.)

        # println("")
        # println("| Time | max ∂t𝜙   | max ∂tψ | max ∂tΠ |")
        # println("|______|___________|_________|_________|")
        # println("")

        CUDA.@time sol = solve(
            prob, alg,
            abstol = atol,
            dt = dt,
            adaptive = false,
            saveat = printtimes,
            alias_u0 = true
            # progress = true,
            # progress_steps = custom_progress_step,
            # progress_message = custom_progress_message
        );

        return

        # k1=similar(ArrayPartition,T,n)
        # k2=similar(ArrayPartition,T,n)
        # k3=similar(ArrayPartition,T,n)
        # k4=similar(ArrayPartition,T,n)
        #
        # k1.=dtstate
        # rhs!(k1,regstate,param,0.)
        # k1 *= dt
        #
        # k2.=dtstate
        # rhs!(k1,regstate,param,0.)
        # k1 *= dt
        #
        # k2 .= dt*rhs!(dtstate + k1/2, regstate, param, 0.)
        # k3 .= dt*rhs!(dtstate + k2/2, regstate, param, 0.)
        # k4 .= dt*rhs!(dtstate + k3, regstate, param, 0.)
        #
        # return t0+h,x+(hk1+2*hk2+2*hk3+hk4)/6



        return

        solution_saver(T,grid,sol,param,folder)


    end

    return

end


end
