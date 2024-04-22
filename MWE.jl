module MWE

const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using BenchmarkTools
using Plots
#using PyPlot
#using GR
using RecursiveArrayTools
using StaticArrays
using InteractiveUtils
using Traceur

using HDF5
using FileIO

using ForwardDiff

ParallelStencil.@reset_parallel_stencil()

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

# Include the input parameter file

include("GR_Spherical/inputfile.jl")


@parallel_indices (x,y) function rhs!(∂tU,U,Hm,∂Hm,rm,θm,ns,_ds)

    ##############################################################################
    # Calculates the right-hand-side of the evolution equations
    # Greek indices run over space and time (i.e. μ in 1:4)
    # whereas Latin indices run over space only (i.e. j in 2:4).
    # The position indices, which are parallelized over, are x and y. 
    # At some points in this code, advantage of axisymmetry is taken, so
    # generalizations to 3D may require more calculations, such
    # as when inverting the metric or in the bounds of the for loops.
    #############################################################################

    # Position to be evaluated on this kernel
    r = rm[x,y]; θ = θm[x,y];

    # Give names to stored arrays from the state vector
    ( gtt,  gtr,  gtθ,  grr,  grθ,  gθθ,  gϕϕ,
      Ptt,  Ptr,  Ptθ,  Prr,  Prθ,  Pθθ,  Pϕϕ,
     drtt, drtr, drtθ, drrr, drrθ, drθθ, drϕϕ,
     dθtt, dθtr, dθtθ, dθrr, dθrθ, dθθθ, dθϕϕ  ) = U.x

    # Give names to stored arrays from the time derivative of the state vector
    ( ∂tgtt,  ∂tgtr,  ∂tgtθ,  ∂tgrr,  ∂tgrθ,  ∂tgθθ,  ∂tgϕϕ,
      ∂tPtt,  ∂tPtr,  ∂tPtθ,  ∂tPrr,  ∂tPrθ,  ∂tPθθ,  ∂tPϕϕ,
      ∂tdrtt, ∂tdrtr, ∂tdrtθ, ∂tdrrr, ∂tdrrθ, ∂tdrθθ, ∂tdrϕϕ,
      ∂tdθtt, ∂tdθtr, ∂tdθtθ, ∂tdθrr, ∂tdθrθ, ∂tdθθθ, ∂tdθϕϕ  ) = ∂tU.x

    # Unpack gauge functions
    (Ht,Hr,Hθ) = Hm.x
    (∂tHt,∂tHr,∂tHθ,∂rHt,∂rHr,∂rHθ,∂θHt,∂θHr,∂θHθ) = ∂Hm.x

    # Define variables to hold the (x,y) slice of the state vector to be evaluated in this CPU kernel. 
    # Slicing now instead of each time it is needed is far more efficient, and this does not allocate memory.  
    gttxy  = U.x[ 1][x,y]; gtrxy  = U.x[ 2][x,y]; gtθxy  = U.x[ 3][x,y];
    grrxy  = U.x[ 4][x,y]; grθxy  = U.x[ 5][x,y]; gθθxy  = U.x[ 6][x,y]; gϕϕxy  = U.x[ 7][x,y];
    Pttxy  = U.x[ 8][x,y]; Ptrxy  = U.x[ 9][x,y]; Ptθxy  = U.x[10][x,y];
    Prrxy  = U.x[11][x,y]; Prθxy  = U.x[12][x,y]; Pθθxy  = U.x[13][x,y]; Pϕϕxy  = U.x[14][x,y];
    drttxy = U.x[15][x,y]; drtrxy = U.x[16][x,y]; drtθxy = U.x[17][x,y];
    drrrxy = U.x[18][x,y]; drrθxy = U.x[19][x,y]; drθθxy = U.x[20][x,y]; drϕϕxy = U.x[21][x,y];
    dθttxy = U.x[22][x,y]; dθtrxy = U.x[23][x,y]; dθtθxy = U.x[24][x,y];
    dθrrxy = U.x[25][x,y]; dθrθxy = U.x[26][x,y]; dθθθxy = U.x[27][x,y]; dθϕϕxy = U.x[28][x,y];

    Htxy = Ht[x,y]; Hrxy = Hr[x,y]; Hθxy = Hθ[x,y];

    ∂tHtxy = ∂tHt[x,y]; ∂tHrxy = ∂tHr[x,y]; ∂tHθxy = ∂tHθ[x,y];
    ∂rHtxy = ∂rHt[x,y]; ∂rHrxy = ∂rHr[x,y]; ∂rHθxy = ∂rHθ[x,y];
    ∂θHtxy = ∂θHt[x,y]; ∂θHrxy = ∂θHr[x,y]; ∂θHθxy = ∂θHθ[x,y];

    # Sliced version of lapse and shift for calculations
    det  = grrxy*gθθxy - grθxy^2
    βrxy = (gtrxy*gθθxy-gtθxy*grθxy)/det
    βθxy = (gtθxy*grrxy-gtrxy*grθxy)/det
    αxy  = sqrt(-gttxy + grrxy*βrxy^2 + 2*grθxy*βrxy*βθxy + gθθxy*βθxy^2)
    # 680μs

    ∂tgttxy = βrxy*drttxy + βθxy*dθttxy - αxy*Pttxy; ∂tgtrxy = βrxy*drtrxy + βθxy*dθtrxy - αxy*Ptrxy;
    ∂tgtθxy = βrxy*drtθxy + βθxy*dθtθxy - αxy*Ptθxy; ∂tgrrxy = βrxy*drrrxy + βθxy*dθrrxy - αxy*Prrxy;
    ∂tgrθxy = βrxy*drrθxy + βθxy*dθrθxy - αxy*Prθxy; ∂tgθθxy = βrxy*drθθxy + βθxy*dθθθxy - αxy*Pθθxy;
    ∂tgϕϕxy = βrxy*drϕϕxy + βθxy*dθϕϕxy - αxy*Pϕϕxy; 
    
    # Metric determinant and inverse components
    γirr = gθθxy/det; γirθ = -grθxy/det; γiθθ = grrxy/det; γiϕϕ = 1.0/gϕϕxy;
    nt = 1.0/αxy; nx = -βrxy/αxy; ny = -βθxy/αxy; 

    gitt = -nt^2; 
    gitr = -nt*nx; girr = γirr-nx^2;
    gitθ = -nt*ny; girθ = γirθ-nx*ny; giθθ = γiθθ-ny^2;
    giϕϕ = γiϕϕ

    # Put gauge functions into index notation
    @inline H(μ) = μ==1 ? Htxy : μ==2 ? Hrxy : μ==3 ? Hθxy : μ==4 ? 0. : @assert false

    # # Same with the derivatives
    
    @inline function ∂H(μ,ν) 
    μ==1 ? (ν==1 ? ∂tHtxy : ν==2 ? ∂tHrxy : ν==3 ? ∂tHθxy : ν==4 ? 0. : @assert false) :
    μ==2 ? (ν==1 ? ∂rHtxy : ν==2 ? ∂rHtxy : ν==3 ? ∂rHθxy : ν==4 ? 0. : @assert false) :
    μ==3 ? (ν==1 ? ∂θHtxy : ν==2 ? ∂θHrxy : ν==3 ? ∂θHθxy : ν==4 ? 0. : @assert false) :
    μ==4 ? (ν==1 ? 0.     : ν==2 ? 0.     : ν==3 ? 0.     : ν==4 ? 0. : @assert false) : @assert false
    end

    @inline β(i) = i==2 ? βrxy : i==3 ? βθxy : i==4 ? 0. : @assert false

    @inline function gi(μ,ν) 
    μ==1 ? (ν==1 ? gitt : ν==2 ? gitr : ν==3 ? gitθ : ν==4 ? 0.   : @assert false) :
    μ==2 ? (ν==1 ? gitr : ν==2 ? girr : ν==3 ? girθ : ν==4 ? 0.   : @assert false) :
    μ==3 ? (ν==1 ? gitθ : ν==2 ? girθ : ν==3 ? giθθ : ν==4 ? 0.   : @assert false) :
    μ==4 ? (ν==1 ? 0.   : ν==2 ? 0.   : ν==3 ? 0.   : ν==4 ? giϕϕ : @assert false) : @assert false
    end

    @inline function g(μ,ν) 
    μ==1 ? (ν==1 ? gttxy : ν==2 ? gtrxy : ν==3 ? gtθxy : ν==4 ? 0.    : @assert false) :
    μ==2 ? (ν==1 ? gtrxy : ν==2 ? grrxy : ν==3 ? grθxy : ν==4 ? 0.    : @assert false) :
    μ==3 ? (ν==1 ? gtθxy : ν==2 ? grθxy : ν==3 ? gθθxy : ν==4 ? 0.    : @assert false) :
    μ==4 ? (ν==1 ? 0.    : ν==2 ? 0.    : ν==3 ? 0.    : ν==4 ? gϕϕxy : @assert false) : @assert false
    end

    @inline function P(μ,ν) 
    μ==1 ? (ν==1 ? Pttxy : ν==2 ? Ptrxy : ν==3 ? Ptθxy : ν==4 ? 0.    : @assert false) :
    μ==2 ? (ν==1 ? Ptrxy : ν==2 ? Prrxy : ν==3 ? Prθxy : ν==4 ? 0.    : @assert false) :
    μ==3 ? (ν==1 ? Ptθxy : ν==2 ? Prθxy : ν==3 ? Pθθxy : ν==4 ? 0.    : @assert false) :
    μ==4 ? (ν==1 ? 0.    : ν==2 ? 0.    : ν==3 ? 0.    : ν==4 ? Pϕϕxy : @assert false) : @assert false
    end

    @inline function d(i,μ,ν) 
    i==4 ? 0. :
    i==2 ? (μ==1 ? (ν==1 ? drttxy : ν==2 ? drtrxy : ν==3 ? drtθxy : ν==4 ? 0.     : @assert false) :
            μ==2 ? (ν==1 ? drtrxy : ν==2 ? drrrxy : ν==3 ? drrθxy : ν==4 ? 0.     : @assert false) :
            μ==3 ? (ν==1 ? drtθxy : ν==2 ? drrθxy : ν==3 ? drθθxy : ν==4 ? 0.     : @assert false) :
            μ==4 ? (ν==1 ? 0.     : ν==2 ? 0.     : ν==3 ? 0.     : ν==4 ? drϕϕxy : @assert false) : @assert false) :
    i==3 ? (μ==1 ? (ν==1 ? dθttxy : ν==2 ? dθtrxy : ν==3 ? dθtθxy : ν==4 ? 0.     : @assert false) :
            μ==2 ? (ν==1 ? dθtrxy : ν==2 ? dθrrxy : ν==3 ? dθrθxy : ν==4 ? 0.     : @assert false) :
            μ==3 ? (ν==1 ? dθtθxy : ν==2 ? dθrθxy : ν==3 ? dθθθxy : ν==4 ? 0.     : @assert false) :
            μ==4 ? (ν==1 ? 0.     : ν==2 ? 0.     : ν==3 ? 0.     : ν==4 ? dθϕϕxy : @assert false) : @assert false) : @assert false
    end

    @inline function f∂tg(μ,ν)
    μ==1 ? (ν==1 ? ∂tgttxy : ν==2 ? ∂tgtrxy : ν==3 ? ∂tgtθxy : ν==4 ? 0.      : @assert false) :
    μ==2 ? (ν==1 ? ∂tgtrxy : ν==2 ? ∂tgrrxy : ν==3 ? ∂tgrθxy : ν==4 ? 0.      : @assert false) :
    μ==3 ? (ν==1 ? ∂tgtθxy : ν==2 ? ∂tgrθxy : ν==3 ? ∂tgθθxy : ν==4 ? 0.      : @assert false) :
    μ==4 ? (ν==1 ? 0.      : ν==2 ? 0.      : ν==3 ? 0.      : ν==4 ? ∂tgϕϕxy : @assert false) : @assert false 
    end

    # Define all metric derivatives in terms of the state vector in index notation
    @inline ∂g(σ,μ,ν) = σ==1 ? f∂tg(μ,ν) : σ==4 ? 0. : d(σ,μ,ν)

    for k in 1:7

        (μ,ν) = ((1,1),(2,1),(3,1),(2,2),(3,2),(3,3),(4,4))[k]
        
        #temporaries
        ∂tgμν = 0.; ∂tPμν = 0.; ∂tdrμν = 0.; ∂tdθμν = 0.; 

        #fast
        for ϵ in 1:3, σ in 1:3
            ∂tPμν += gi(ϵ,σ)*(∂g(μ,ν,σ)+∂g(ν,μ,σ))*H(ϵ)
        end
        # Benchmark shows 670μs with 8 threads

        #slow
        # for ϵ in 1:4, σ in 1:4
        #     ∂tPμν += gi(ϵ,σ)*(∂g(μ,ν,σ)+∂g(ν,μ,σ))*H(ϵ)
        # end
        # Benchmark shows 6ms with 8 threads

    end 
    
    # 210ms
    
    return
    
end

# Sample analytic functions to the grid
# function sample!(f, fun, ns, r, θ, μ...)

#     f .= Data.Array([fun(r[i,j],θ[i,j],μ...) for i in 1:ns[1], j in 1:ns[2]])

# end

#Sample analytic tensors to the grid
function sample!(f, fun, ns, r, θ)

    for x in ns[1], y in ns[2]
        f[x,y,1] = fun(r[x,y],θ[x,y],1,1)
        f[x,y,2] = fun(r[x,y],θ[x,y],1,2)
        f[x,y,3] = fun(r[x,y],θ[x,y],1,3)
        f[x,y,4] = fun(r[x,y],θ[x,y],2,2)
        f[x,y,5] = fun(r[x,y],θ[x,y],2,3)
        f[x,y,6] = fun(r[x,y],θ[x,y],3,3)
        f[x,y,7] = fun(r[x,y],θ[x,y],4,4)
    end

end

@CellType Axisymmetric4DTensor fieldnames=(tt, tr, tθ, rr, rθ, θθ, ϕϕ)

##################################################
function main()
    # Physics

    #return @macroexpand @sum (ϵ,σ,λ,ρ) ($a += gi(ϵ,σ)*gi(λ,ρ)*(∂g(λ,ϵ,μ)*∂g(ρ,σ,ν) - Γ(μ,ϵ,λ)*Γ(ν,σ,ρ)))


    numvar=4*7

    # domains
    rmin, rmax = 5.0, 10.0
    #θmin, θmax = 0.0, pi
    tmin, tmax = 0.0, 50.

    t         = tmin         # physical time
    # Numerics
    #scale = 20 # normal amount to test with
    scale = 20

    #nr, nθ    = 10,10
    nr, nθ    = 32*scale-1, 32*scale  # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf

    ns = (nr,nθ)

    # Derived numerics
    dr = (rmax-rmin)/(nr-1)
    dθ = pi/(nθ) # cell sizes
    _dr, _dθ   = 1.0/dr, 1.0/dθ
    _ds = (_dr,_dθ)

    dt        = min(dr,rmin*dθ)/4.1
    #dt       = min(dr,rmin*dθ)/4.0 #CFL

    #nt = (tmax-tmin)/dt+1
    nt = 100

    nout = 2#round(nt/100)          # plotting frequency
    nsave = Int(ceil(nt/nout))
    #nt=10

    nd = 4

    r  = @zeros(nr,nθ)
    θ  = @zeros(nr,nθ)

    r .= Data.Array([rmin + dr*(i-1) for i in 1:nr, j in 1:nθ])
    θ .= Data.Array([dθ/2 + dθ*(j-1) for i in 1:nr, j in 1:nθ])

    #Un    = @zeros(nr,nθ,numvar,nd,nd)
    # Un1   = @zeros(nr,nθ,numvar,nd,nd)
    # k     = @zeros(nr,nθ,numvar,nd,nd)
    # ∂ₜU    = @zeros(nr,nθ,numvar,nd,nd)

    #Un    = ArrayPartition([@zeros(nr,nθ) for i in 1:numvar]...)
    # Un1   = ArrayPartition([@zeros(nr,nθ) for i in 1:numvar]...)
    # k     = ArrayPartition([@zeros(nr,nθ) for i in 1:numvar]...)
    # ∂ₜU    = ArrayPartition([@zeros(nr,nθ) for i in 1:numvar]...)

    # Hm     = ArrayPartition([@zeros(nr,nθ) for i in 1:(nd-1)]...)
    # ∂Hm    = ArrayPartition([@zeros(nr,nθ) for i in 1:(3*nd-1)]...)

    r  = @zeros(nr,nθ)
    θ  = @zeros(nr,nθ)

    U  = @zeros(4,nr,nθ,celltype=Axisymmetric4DTensor)

    g = @view U[1,:,:]


    M = 1.
    sign = 1.

    @inline g_init(r,θ,μ,ν) =  (( -(1 - 2*M/r) , sign*2*M/r  , 0.  ,        0.      ),
                                ( sign*2*M/r   , (1 + 2*M/r) , 0.  ,        0.      ),
                                (      0.      ,      0.     , r^2 ,        0.      ),
                                (      0.      ,      0.     , 0.  , (r^2)*sin(θ)^2 ))[μ][ν]

    sample!(g, g_init, ns, r, θ)

    return g[1,1]



    # Define initial conditions


    # Note: Assumes initial 3-metric is diagonal                           
    @inline β(r,θ,i) = g_init(r,θ,1,i)/g_init(r,θ,i,i)  #+ g_init(r,θ,1,4)/g_init(r,θ,4,i)
    @inline α(r,θ)   = sqrt(-g_init(r,θ,1,1) + g_init(r,θ,2,2)*β(r,θ,2)^2 + g_init(r,θ,3,3)*β(r,θ,3)^2 )

    @inline ∂ₜg_init(r,θ,μ,ν) =  ((  0. ,  0.  ,  0.  ,  0.   ),
                                 (  0. ,  0.  ,  0.  ,  0.   ),
                                 (  0. ,  0.  ,  0.  ,  0.   ),
                                 (  0. ,  0.  ,  0.  ,  0.   ))[μ][ν]
     
    @inline d_init(r,θ,i,μ,ν) = (ForwardDiff.derivative(r -> g_init(r,θ,μ,ν), r),
                                 ForwardDiff.derivative(θ -> g_init(r,θ,μ,ν), θ),0.)[i-1]
                                   
    @inline P_init(r,θ,μ,ν) = -(∂ₜg_init(r,θ,μ,ν) - β(r,θ,2)*d_init(r,θ,2,μ,ν) - β(r,θ,3)*d_init(r,θ,3,μ,ν))/α(r,θ)

    # This is annoying to calculate on the fly
    @inline H(r,θ,μ) = (-2*M/r,-2*(r+M)/r^2,-cos(θ)/sin(θ),0.)[μ]

    @inline ∂H(r,θ,μ,ν) = ((0.,0.,0.,0.)[ν],
                           ForwardDiff.derivative(r -> H(r,θ,ν), r),
                           ForwardDiff.derivative(θ -> H(r,θ,ν), θ),
                           (0.,0.,0.,0.)[ν])[μ]

    # Sample initial state vector onto the grid
    sample!(gtt, g_init, ns, r, θ, 1, 1)
    sample!(gtr, g_init, ns, r, θ, 1, 2)
    sample!(gtθ, g_init, ns, r, θ, 1, 3)
    sample!(grr, g_init, ns, r, θ, 2, 2)
    sample!(grθ, g_init, ns, r, θ, 2, 3)
    sample!(gθθ, g_init, ns, r, θ, 3, 3)
    sample!(gϕϕ, g_init, ns, r, θ, 4, 4)

    sample!(Ptt, P_init, ns, r, θ, 1, 1)
    sample!(Ptr, P_init, ns, r, θ, 1, 2)
    sample!(Ptθ, P_init, ns, r, θ, 1, 3)
    sample!(Prr, P_init, ns, r, θ, 2, 2)
    sample!(Prθ, P_init, ns, r, θ, 2, 3)
    sample!(Pθθ, P_init, ns, r, θ, 3, 3)
    sample!(Pϕϕ, P_init, ns, r, θ, 4, 4)

    sample!(drtt, d_init, ns, r, θ, 2, 1, 1)
    sample!(drtr, d_init, ns, r, θ, 2, 1, 2)
    sample!(drtθ, d_init, ns, r, θ, 2, 1, 3)
    sample!(drrr, d_init, ns, r, θ, 2, 2, 2)
    sample!(drrθ, d_init, ns, r, θ, 2, 2, 3)
    sample!(drθθ, d_init, ns, r, θ, 2, 3, 3)
    sample!(drϕϕ, d_init, ns, r, θ, 2, 4, 4)

    sample!(dθtt, d_init, ns, r, θ, 3, 1, 1)
    sample!(dθtr, d_init, ns, r, θ, 3, 1, 2)
    sample!(dθtθ, d_init, ns, r, θ, 3, 1, 3)
    sample!(dθrr, d_init, ns, r, θ, 3, 2, 2)
    sample!(dθrθ, d_init, ns, r, θ, 3, 2, 3)
    sample!(dθθθ, d_init, ns, r, θ, 3, 3, 3)
    sample!(dθϕϕ, d_init, ns, r, θ, 3, 4, 4)

    sample!(Ht, H, ns, r, θ, 1)
    sample!(Hr, H, ns, r, θ, 2)
    sample!(Hθ, H, ns, r, θ, 3)

    sample!(∂rHt, ∂H, ns, r, θ, 2, 1)
    sample!(∂rHr, ∂H, ns, r, θ, 2, 2)
    sample!(∂rHθ, ∂H, ns, r, θ, 2, 3)
    sample!(∂θHt, ∂H, ns, r, θ, 3, 1)
    sample!(∂θHr, ∂H, ns, r, θ, 3, 2)
    sample!(∂θHθ, ∂H, ns, r, θ, 3, 3)

    return @benchmark @parallel (1:$nr,1:$nθ) rhs!($∂ₜU,$Un,$Hm,$∂Hm,$r,$θ,$ns,$_ds)

    
end

end