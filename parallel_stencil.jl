module GR_Axial_parallel

const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
const VISUAL = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using BenchmarkTools
using Plots
#using PyPlot
#using GR
using RecursiveArrayTools
using TensorOperations
using StaticArrays
using InteractiveUtils
using Traceur

using StructArrays

using HDF5
using FileIO

using Tensorial

using ForwardDiff

ParallelStencil.@reset_parallel_stencil()

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics

import ParallelStencil.INDICES
import ParallelStencil.WITHIN_DOC

# Macro for applying get_index to an expression
# helps to clean up everything so that you don't 
# have to write so many indices
macro part(indices, expr)
    esc(parse_index(expr,indices))
end

function parse_index(expr,indices)
    # Do nothing to numbers
    expr isa Number && return expr
    # Add the `expr_index` call to symbols
    expr isa Symbol && return :(expr_index($(expr), $indices...))
    if expr isa Expr
        # if the expr is an assignment, assign that index
        expr.head ≡ :(=) && return :(setindex!($(expr.args[1]),
                               $(parse_index(expr.args[2],indices)),
                               $indices...))
       # Handle function calls recursively for each argument.
       # `args[1]` is the function itself, which isn't modified.
       # `args[2:end]` are the function arguments.
       expr.head ≡ :call && return Expr(expr.head,expr.args[1],
                parse_index.( (@view expr.args[2:end]), Ref(indices))...)
    end
    # Abort if we find something unexpected
    println(typeof(expr))
    @assert false
end

expr_index(x::Number, i...) = x
expr_index(a::AbstractArray, i...) = a[i...]
expr_index(a::Data.Array, i...) = a[i...]

# Include the input parameter file

#include("GR_Spherical/inputfile.jl")

# Alias for SymmetricSecondOrderTensor 4x4
const StateTensor{T} = SymmetricSecondOrderTensor{4,T,10}

# Alias for SymmetricSecondOrderTensor 3x3
const ThreeTensor{T} = SymmetricSecondOrderTensor{3,T,6}

# Alias for non-symmetric 4 tensor
const FourTensor{T} = SecondOrderTensor{4,T,16}

# This struct type is used to package the state vector into 
# spherical components. This is used to store the state vector
# into memory, and does not store ϕ derivatives as those vanish in axisymmetry.
# Another 30% memory savings is possible if we exclude ϕ coponents of tensors,
# but that is not implemented.
struct SphericalStateVector{T}
    # φ::T
    # ψr::T
    # ψθ::T
    # Π::T
    g::StateTensor{T}
    dr::StateTensor{T}
    dθ::StateTensor{T}
    P::StateTensor{T}
end

# This struct type is used to package the state vector into 
# Cartesian components. Math is done to this struct when integrating
# the equations, so it needs to have math operators defined.
struct CartesianStateVector{T}
    # φ::T
    # ψx::T
    # ψy::T
    # ψz::T
    # Π::T
    g::StateTensor{T}
    dx::StateTensor{T}
    dy::StateTensor{T}
    dz::StateTensor{T}
    P::StateTensor{T}
end

const Storage{T} = StructArray{SphericalStateVector{T}, 2, NamedTuple{(:g, :dr, :dθ, :P), NTuple{4, Matrix{StateTensor{T}}}}, Int64}

# Define math operators for CartesianStateVector
math_operators = [:+, :-, :*, :/, :^]
for op in math_operators
    @eval import Base.$op
    @eval @inline function $op(A::CartesianStateVector{T},B::CartesianStateVector{T}) where T
        g  = @. $op(A.g,B.g)
        dx = @. $op(A.dx,B.dx)
        dy = @. $op(A.dy,B.dy)
        dz = @. $op(A.dz,B.dz)
        P  = @. $op(A.P,B.P)
        return CartesianStateVector{T}(g,dx,dy,dz,P)
    end

    @eval @inline function $op(a::Number,B::CartesianStateVector{T}) where T
        g  = $op(a,B.g)
        dx = $op(a,B.dx)
        dy = $op(a,B.dy)
        dz = $op(a,B.dz)
        P  = $op(a,B.P)
        return CartesianStateVector{T}(g,dx,dy,dz,P)
    end
end

for op in math_operators
    @eval import Base.$op
    @eval @inline function $op(A::SphericalStateVector{T},B::SphericalStateVector{T}) where T
        g  = @. $op(A.g,B.g)
        dr = @. $op(A.dr,B.dr)
        dθ = @. $op(A.dθ,B.dθ)
        P  = @. $op(A.P,B.P)
        return SphericalStateVector{T}(g,dr,dθ,P)
    end

    @eval @inline function $op(a::Number,B::SphericalStateVector{T}) where T
        g  = $op(a,B.g)
        dr = $op(a,B.dr)
        dθ = $op(a,B.dθ)
        P  = $op(a,B.P)
        return SphericalStateVector{T}(g,dr,dθ,P)
    end
end

# Define functions to return Stuct components for finite 
# differencing in constraint calculations
@inline fg(x,y,U::SphericalStateVector)  = U[x,y].g
@inline fdr(x,y,U::SphericalStateVector) = U[x,y].dr
@inline fdθ(x,y,U::SphericalStateVector) = U[x,y].dθ
@inline fP(x,y,U::SphericalStateVector)  = U[x,y].P

const parity  = StateTensor((1,1,-1,1,1,-1,1,1,-1,1))
const parityC = Vec{4}((1,1,-1,1))

# @inline function Aθ2(f::Function,U,x,y) 
#     f(x,y,U)
#     # n = size(U,2)
#     # if y in 2:n-1
#     #     0.5*f(x,y-1,U) + 0.5*f(x,y+1,U)
#     # elseif y==1
#     #     # axis included
#     #     #f(x,2,U)
#     #     # straddled 
#     #     0.5*f(x,1,U) + 0.5*f(x,2,U)
#     # elseif y==n 
#     #     0.5*f(x,n-1,U) + 0.5*f(x,n,U)
#     # end
# end

# @inline function Divr2(f::Function,U,ns,x,y) 
#     n = ns[1]
#     if x in 2:n-1
#         -0.5*f(x-1,y,U) + 0.5*f(x+1,y,U)
#     elseif x==1
#         -f(1,y,U) + f(2,y,U)
#     elseif x==n 
#         -f(n-1,y,U) + f(n,y,U)
#     end
# end

@inline function DrC(f::Function,U,ns,x,y) 
    n = ns[1]
    if x in 2:n-1
        -0.5*f(U[x-1,y])+0.5*f(U[x+1,y])
    elseif x==1
        -f(U[1,y]) + f(U[2,y])
    else#if x==n 
        -f(U[n-1,y]) + f(U[n,y])
    end
    #else @assert false end
end

@inline function DθC(f::Function,U,ns,x,y)
    n = ns[2]
    if y in 2:n-1
        -0.5*f(U[x,y-1]) + 0.5*f(U[x,y+1])
    elseif y==1
        #-(3/2)*f(x,1,U) + 2*f(x,2,U) - (1/2)*f(x,3,U)
        -0.5*parityC.*f(U[x,1]) + 0.5*f(U[x,2])
        #-f(x,1,U) + f(x,2,U)
    else#if y==n
        -0.5*f(U[x,n-1]) + 0.5*parityC.*f(U[x,n]) 
        #-f(x,n-1,U) + f(x,n,U)
    end
end

@inline function Dr2(f::Function,U,ns,x,y)
    n = ns[1]
    if x in 2:n-1
        -0.5*f(U[x-1,y])+0.5*f(U[x+1,y])
    elseif x==1
        -f(U[1,y]) + f(U[2,y])
    else#if x==n 
        -f(U[n-1,y]) + f(U[n,y])
    end
    #else @assert false end
end

@inline function Dθ2(f::Function,U,ns,x,y)
    n = ns[2]
    if y in 2:n-1
        -0.5*f(U[x,y-1]) + 0.5*f(U[x,y+1])
    elseif y==1
        #-(3/2)*f(x,1,U) + 2*f(x,2,U) - (1/2)*f(x,3,U)
        -0.5*parity.*f(U[x,1]) + 0.5*f(U[x,2])
        #-f(x,1,U) + f(x,2,U)
    else#if y==n
        -0.5*f(U[x,n-1]) + 0.5*parity.*f(U[x,n]) 
        #-f(x,n-1,U) + f(x,n,U)
    end
    #else @assert false end
end

@inline function Divθ2(f::Function,U,ns,x,y)
    n = ns[2]
    if y in 2:n-1
        -0.5*f(U[x,y-1]) + 0.5*f(U[x,y+1])
    elseif y==1
        0.5*parity.*f(U[x,1]) + 0.5*f(U[x,2])
    else#if y==n
        -0.5*f(U[x,n-1]) - 0.5*parity.*f(U[x,n]) 
    end
    #else @assert false end
end

# @inline function Dx2(f::Function,U,r,θ,ns,_ds,x,y) 
#     Dr2(f,U,ns,x,y)*_ds[1]*sin(θ[x,y]) + Dθ2(f,U,ns,x,y)*_ds[2]*cos(θ[x,y])/r[x,y]
# end

# @inline function Dz2(f::Function,U,r,θ,ns,_ds,x,y)
#     Dr2(f,U,ns,x,y)*_ds[1]*cos(θ[x,y]) - Dθ2(f,U,ns,x,y)*_ds[2]*sin(θ[x,y])/r[x,y]
# end

# @inline function Divx2(f::Function,U,r,θ,ns,_ds,x,y) 
#     Dr2(f,U,ns,x,y)*_ds[1]*sin(θ[x,y]) + Divθ2(f,U,ns,x,y)*_ds[2]*cos(θ[x,y])/r[x,y]
# end

# @inline function Divz2(f::Function,U,r,θ,ns,_ds,x,y)
#     Dr2(f,U,ns,x,y)*_ds[1]*cos(θ[x,y]) - Divθ2(f,U,ns,x,y)*_ds[2]*sin(θ[x,y])/r[x,y]
# end

# @inline function Divxz(vx::Function,vz::Function,U,r,θ,ns,_ds,x,y)
#     (Divx2(vx,U,r,θ,ns,_ds,x,y) + Divz2(vz,U,r,θ,ns,_ds,x,y))/rootγ(U[x,y])
# end

@inline function Div(vr::Function,vθ::Function,U,ns,_ds,x,y)::StateTensor
    (Dr2(vr,U,ns,x,y)*_ds[1] + Divθ2(vθ,U,ns,x,y)*_ds[2])/rootγ(U[x,y])
end

#@inline function Div(vx::Function,vz::Function,U,r,θ,ns,_ds,x,y)
#     (Dr2(vx,U,r,θ,ns,_ds,x,y) + Divz2(vz,U,r,θ,ns,_ds,x,y))/rootγ(x,y,U)
# end

# @inline function DxC(f::Function,U,r,θ,ns,_ds,x,y) 
#     DrC(f,U,ns,x,y)*_ds[1]*sin(θ[x,y]) + DθC(f,U,ns,x,y)*_ds[2]*cos(θ[x,y])/r[x,y]
# end

# @inline function DzC(f::Function,U,r,θ,ns,_ds,x,y)
#     DrC(f,U,ns,x,y)*_ds[1]*cos(θ[x,y]) - DθC(f,U,ns,x,y)*_ds[2]*sin(θ[x,y])/r[x,y]
# end

# @inline function Div(vr::Function,vθ::Function,U,ns,_ds,x,y)
#     if y in 2:ns[2]-1
#     (Divr2(vr,U,ns,x,y)*_ds[1] + Divθ2(vθ,U,ns,_ds,x,y)*_ds[2])/Aθ2(rootγ,U,x,y)
#     elseif y==1
#         #Divr2(vr,U,ns,x,y)*_ds[1]/Aθ2(rootγ,U,x,y) + 
#         testST
#     elseif y==ns[2]
#         testST
#         #Divr2(vr,U,ns,x,y)*_ds[1]/Aθ2(rootγ,U,x,y) + testST
#     end

# end

@inline function u(U::SphericalStateVector) # Scalar gradient-flux

    # Give names to stored arrays from the state vector
    g  = U.g 
    dr = U.dr
    dθ = U.dθ  
    P  = U.P 

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    return βr*dr + βθ*dθ - α*P

end

@inline function vr(U::SphericalStateVector) # r component of the divergence-flux

    # Give names to stored arrays from the state vector
    g  = U.g 
    dr = U.dr 
    dθ = U.dθ
    P  = U.P 

    _,_,_,_,γs... = g.data

    γ = ThreeTensor(γs)

    γi = inv(γ)

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]

    return rootγ(U)*(βr*P - α*(γi[1,1]*dr + γi[1,2]*dθ))
    
end

@inline function vθ(U::SphericalStateVector) # θ component of the divergence-flux

    # Give names to stored arrays from the state vector
    g  = U.g 
    dr = U.dr
    dθ = U.dθ  
    P  = U.P 

    _,_,_,_,γs... = g.data

    γ = ThreeTensor(γs)

    γi = inv(γ)

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βθ = -gi[1,3]/gi[1,1]

    return rootγ(U)*(βθ*P - α*(γi[2,1]*dr + γi[2,2]*dθ))
    
end


@inline function rootγ(U::SphericalStateVector)

    # Give names to stored arrays from the state vector
    g  = U.g 

    # Unpack the metric into indiviual components
    _,_,_,_,γs... = g.data

    γ = ThreeTensor(γs)

    detγ = det(γ)

    if detγ < 0
        println(x," ",y," ",γ[1,1]," ",γ[2,2]," ",γ[3,3]," ")
    end

    return sqrt(detγ)
end

@inline function SphericalToCartesian(U::SphericalStateVector{T},r,θ) where T

    # Give names to stored arrays from the state vector
    g  = U.g 
    dr = U.dr   
    dθ = U.dθ 
    P  = U.P

    # Transformation matrices from spherical to Cartesian evaluated at ϕ=0
    Λi   = FourTensor{T}((1.,0.,0.,0.,0.,sin(θ),cos(θ)/r,0.,0.,0.,0.,1/sin(θ)/r,0.,cos(θ),-sin(θ)/r,0.))
    ∂rΛi = FourTensor{T}((0.,0.,0.,0.,0.,0.,-cos(θ)/r^2,0.,0.,0.,0.,-1/sin(θ)/r^2,0.,0.,sin(θ)/r^2,0.))
    ∂θΛi = FourTensor{T}((0.,0.,0.,0.,0.,cos(θ),-sin(θ)/r,0.,0.,0.,0.,-cos(θ)/r/sin(θ)^2,0.,-sin(θ),-cos(θ)/r,0.))
    ∂ϕΛi = FourTensor{T}((0.,0.,0.,0.,0.,0.,0.,-1/sin(θ)/r,0.,sin(θ),cos(θ)/r,0.,0.,0.,0.,0.)) 
    ∂Λi  = Tensor{Tuple{4,4,4},T}(
        (σ,μ,ν) -> (σ==1 ? 0. : σ==2 ? ∂rΛi[μ,ν] : σ==3 ? ∂θΛi[μ,ν] : σ==4 ? ∂ϕΛi[μ,ν] : @assert false)
        )

    #52ns

    # convert to Cartesian derivatives
    ∂Λi = @einsum (σ,μ,ν) -> Λi[σp,σ]*∂Λi[σp,μ,ν]

    # 50ns

    # Calculate inverse components in Cartesian
    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    # 50ns

    # Calculate time derivative of the metric using only the state vector
    ∂tgs = symmetric(βr*dr + βθ*dθ - α*P)
    
    ∂g = Tensor{Tuple{4,@Symmetry{4,4}},T}(
        (σ,μ,ν) -> (σ==1 ? ∂tgs[μ,ν] : σ==2 ? dr[μ,ν] : σ==3 ? dθ[μ,ν] : σ==4 ? 0. : @assert false)
        )

    #50ns 

    #return

    # Conversion of spherical metric derivatives to Cartesian ones

    # You have to break up long expressions into multiple parts
    # or else performance gets trashed 
    ∂gc  = @einsum (σ,μ,ν) -> Λi[μp,μ]*Λi[νp,ν]*∂g[σ,μp,νp]
    ∂gc  = @einsum (σ,μ,ν) -> Λi[σp,σ]*∂gc[σp,μ,ν]
    ∂gc += @einsum (σ,μ,ν) -> ∂Λi[σ,μp,μ]*Λi[νp,ν]*g[μp,νp]
    ∂gc += @einsum (σ,μ,ν) -> ∂Λi[σ,νp,ν]*Λi[μp,μ]*g[μp,νp]

    #50ns 

    # Convert to Spherical Components on the state vector
    g   = symmetric(@einsum Λi[μp,μ]*Λi[νp,ν]*g[μp,νp])
    ∂tg = symmetric(∂gc[1,:,:])
    dx  = symmetric(∂gc[2,:,:])
    dy  = symmetric(∂gc[3,:,:])
    dz  = symmetric(∂gc[4,:,:])

    # Calculate inverse components in spherical
    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βx = -gi[1,2]/gi[1,1]
    βy = -gi[1,3]/gi[1,1]
    βz = -gi[1,4]/gi[1,1]
    
    P = symmetric(-(∂tg/α - (βx/α)*dx - (βy/α)*dy - (βz/α)*dz))

    Uc = CartesianStateVector{T}(g,dx,dy,dz,P)

    return Uc

end

@inline function CartesianToSpherical(U::CartesianStateVector{T},r,θ) where T

    # Give names to stored arrays from the state vector
    g  = U.g 
    dx = U.dx   
    dy = U.dy
    dz = U.dz  
    P  = U.P 

    # Transformation matrices from Cartesian to spherical evaluated at ϕ=0
    Λ   = FourTensor{T}((1.,0.,0.,0.,0.,sin(θ),0.,cos(θ),0.,r*cos(θ),0.,-r*sin(θ),0.,0.,r*sin(θ),0.))
    ∂rΛ = FourTensor{T}((0.,0.,0.,0.,0.,0.,0.,0.,0.,cos(θ),0.,-sin(θ),0.,0.,sin(θ),0.)) 
    ∂θΛ = FourTensor{T}((0.,0.,0.,0.,0.,cos(θ),0.,-sin(θ),0.,-r*sin(θ),0.,-r*cos(θ),0.,0.,r*cos(θ),0.)) 
    ∂ϕΛ = FourTensor{T}((0.,0.,0.,0.,0.,0.,sin(θ),0.,0.,0.,r*cos(θ),0.,0.,-r*sin(θ),0.,0.)) 
    ∂Λ  = Tensor{Tuple{4,4,4},T}(
        (σ,μ,ν) -> (σ==1 ? 0. : σ==2 ? ∂rΛ[μ,ν] : σ==3 ? ∂θΛ[μ,ν] : σ==4 ? ∂ϕΛ[μ,ν] : @assert false)
        )

    # Calculate inverse components in Cartesian
    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βx = -gi[1,2]/gi[1,1]
    βy = -gi[1,3]/gi[1,1]
    βz = -gi[1,4]/gi[1,1]

    # Calculate time derivative of the metric using only the state vector
    ∂tgc = βx*dx + βy*dy + βz*dz - α*P
    
    ∂gc = Tensor{Tuple{4,@Symmetry{4,4}},T}(
        (σ,μ,ν) -> (σ==1 ? ∂tgc[μ,ν] : σ==2 ? dx[μ,ν] : σ==3 ? dy[μ,ν] : σ==4 ? dz[μ,ν] : @assert false)
        )

    # Conversion of Cartesian metric derivatives to spherical ones

    # You have to break up long expressions into multiple parts
    # or else performance gets trashed 
    ∂gs  = @einsum (σ,μ,ν) -> Λ[μp,μ]*Λ[νp,ν]*∂gc[σ,μp,νp]
    ∂gs  = @einsum (σ,μ,ν) -> Λ[σp,σ]*∂gs[σp,μ,ν]
    ∂gs += @einsum (σ,μ,ν) -> ∂Λ[σ,μp,μ]*Λ[νp,ν]*g[μp,νp]
    ∂gs += @einsum (σ,μ,ν) -> ∂Λ[σ,νp,ν]*Λ[μp,μ]*g[μp,νp]

    # ∂g  = @einsum Λ[σp,σ]*Λ[μp,μ]*Λ[νp,ν]*∂g[σp,μp,νp] 
    # ∂g += @einsum ∂Λ[σ,μp,μ]*Λ[νp,ν]*g[μp,νp]
    # ∂g += @einsum ∂Λ[σ,νp,ν]*Λ[μp,μ]*g[μp,νp]

    # Convert to Spherical Components on the state vector
    g   = symmetric(@einsum Λ[μp,μ]*Λ[νp,ν]*g[μp,νp])
    ∂tg = symmetric(∂gs[1,:,:])
    dr  = symmetric(∂gs[2,:,:])
    dθ  = symmetric(∂gs[3,:,:])

    # Calculate inverse components in spherical
    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]
    
    P = symmetric(-(∂tg - βr*dr - βθ*dθ)/α)

    return SphericalStateVector{T}(g,dr,dθ,P)

end

# Convert time derivatives of the state vector to Cartesian ones
@inline function TimeSphericalToCartesian(∂tU::SphericalStateVector{T},U::SphericalStateVector{T},r,θ) where T

    g  = U.g 
    dr = U.dr   
    dθ = U.dθ 
    P  = U.P

    ∂tg  = ∂tU.g 
    ∂tdr = ∂tU.dr   
    ∂tdθ = ∂tU.dθ 
    ∂tP  = ∂tU.P

    # Transformation matrix from Cartesian to spherical evaluated at ϕ=0
    Λ   = SecondOrderTensor{4,T}((1.,0.,0.,0.,0.,sin(θ),0.,cos(θ),0.,r*cos(θ),0.,-r*sin(θ),0.,0.,r*sin(θ),0.))

    # Transformation matrices from spherical to Cartesian evaluated at ϕ=0
    Λi   = SecondOrderTensor{4,T}((1.,0.,0.,0.,0.,sin(θ),cos(θ)/r,0.,0.,0.,0.,1/sin(θ)/r,0.,cos(θ),-sin(θ)/r,0.))
    ∂rΛi = SecondOrderTensor{4,T}((0.,0.,0.,0.,0.,0.,-cos(θ)/r^2,0.,0.,0.,0.,-1/sin(θ)/r^2,0.,0.,sin(θ)/r^2,0.))
    ∂θΛi = SecondOrderTensor{4,T}((0.,0.,0.,0.,0.,cos(θ),-sin(θ)/r,0.,0.,0.,0.,-cos(θ)/r/sin(θ)^2,0.,-sin(θ),-cos(θ)/r,0.))
    ∂ϕΛi = SecondOrderTensor{4,T}((0.,0.,0.,0.,0.,0.,0.,-1/sin(θ)/r,0.,sin(θ),cos(θ)/r,0.,0.,0.,0.,0.)) 
    ∂Λi  = Tensor{Tuple{4,4,4},T}(
        (σ,μ,ν) -> (σ==1 ? 0. : σ==2 ? ∂rΛi[μ,ν] : σ==3 ? ∂θΛi[μ,ν] : σ==4 ? ∂ϕΛi[μ,ν] : @assert false)
        )

    # convert to Cartesian derivatives
    ∂Λi = @einsum (σ,μ,ν) -> Λi[σp,σ]*∂Λi[σp,μ,ν]

    #return #130ns up from 74 ns

    # Calculate inverse components in Cartesian
    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    β = @Vec [0.,βr,βθ,0.]

    # Unpack the metric into indiviual components
    _,_,_,_,γs... = g.data

    γ = ThreeTensor(γs)

    γi3 = inv(γ)

    γi = StateTensor((0.,0.,0.,0.,γi3.data...))

    nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 

    n = @Vec [nt,nr,nθ,0.0]

    n_ = @Vec [-α,0.0,0.0,0.0]

    ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])
    ∂tβ =   α*(@einsum σ -> γi[σ,μ]*n[ν]*∂tg[μ,ν]) # result is a 4-vector

    ∂t∂tgc  = βr*∂tdr + βθ*∂tdθ - α*∂tP + ∂tβ[2]*dr + ∂tβ[3]*dθ - ∂tα*P

    ∂t∂g = Tensor{Tuple{4,@Symmetry{4,4}},T}(
        (σ,μ,ν) -> (σ==1 ? ∂t∂tgc[μ,ν] : σ==2 ? ∂tdr[μ,ν] : σ==3 ? ∂tdθ[μ,ν] : σ==4 ? 0. : @assert false)
        )

    #return # 2.2μs

    # Conversion of metric derivatives to spherical ones
    # This only works if the transformation matrices are independent of time

    ∂t∂gc  = @einsum (σ,μ,ν) -> Λi[μp,μ]*Λi[νp,ν]*∂t∂g[σ,μp,νp]
    ∂t∂gc  = @einsum (σ,μ,ν) -> Λi[σp,σ]*∂t∂gc[σp,μ,ν]
    ∂t∂gc += @einsum (σ,μ,ν) -> ∂Λi[σ,μp,μ]*Λi[νp,ν]*∂tg[μp,νp]
    ∂t∂gc += @einsum (σ,μ,ν) -> ∂Λi[σ,νp,ν]*Λi[μp,μ]*∂tg[μp,νp]

    ∂tg   = symmetric(@einsum Λi[μ,μp]*Λi[ν,νp]*∂tg[μ,ν])
    ∂t∂tg = symmetric(∂t∂gc[1,:,:])
    ∂tdx  = symmetric(∂t∂gc[2,:,:])
    ∂tdy  = symmetric(∂t∂gc[3,:,:])
    ∂tdz  = symmetric(∂t∂gc[4,:,:])

    ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])
    ∂tβ = α*(@einsum γi[σ,μ]*n[ν]*∂tg[μ,ν]) # result is a 4-vector

    Uc = SphericalToCartesian(U,r,θ)

    g  = Uc.g 
    dx = Uc.dx   
    dy = Uc.dy
    dz = Uc.dz  
    P  = Uc.P 

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βx = -gi[1,2]/gi[1,1]
    βy = -gi[1,3]/gi[1,1]
    βz = -gi[1,4]/gi[1,1]

    # Unpack the metric into indiviual components
    _,_,_,_,γs... = g.data

    γ = ThreeTensor(γs)

    γi3 = inv(γ)

    γi = StateTensor((0.,0.,0.,0.,γi3.data...))

    nt = 1.0/α; nx = -βx/α; ny = -βy/α;  nz = -βz/α; 

    n = @Vec [nt,nx,ny,nz]

    n_ = @Vec [-α,0.0,0.0,0.0]

    ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])
    ∂tβ =   α*(@einsum γi[σ,μ]*n[ν]*∂tg[μ,ν]) # result is a 4-vector

    ∂tP  =  -(∂t∂tg - βx*∂tdx - βy*∂tdy - βz*∂tdz)/α
    ∂tP += ∂tα*(∂tg - βx*dx - βy*dy - βz*dz)/α^2
    ∂tP += (∂tβ[2]*dx + ∂tβ[3]*dy + ∂tβ[4]*dz)/α

    return CartesianStateVector(∂tg,∂tdx,∂tdy,∂tdz,∂tP)

end

function constraints(U::SphericalStateVector{T}) where T

    # Give names to stored arrays from the state vector
    g  = U.g 
    dr = U.dr   
    dθ = U.dθ  
    P  = U.P 

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    #β = @Vec (-[gi[1,2],gi[1,3],gi[1,4]]/gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    # Calculate time derivative of the metric
    ∂tg = βr*dr + βθ*dθ - α*P

    ∂g = Tensor{Tuple{4,@Symmetry{4,4}},T}(
        (σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dr[μ,ν] : σ==3 ? dθ[μ,ν] : σ==4 ? 0. : @assert false)
        )

    Γ  = Tensor{Tuple{4,@Symmetry{4,4}},T}(
        (σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν])
        )    

    C_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ]

    return C_

end

function integrate_constraints(U,H,ns,_ds)

    int = 0

    for x in ns[1], y in ns[2]

        Uxy = U[x,y]
        Hxy = H[x,y]

        g  = Uxy.g 

        gi = inv(g)

        Cxy = constraints(Uxy) - Hxy

        int += sqrt(abs((@einsum gi[μ,ν]*Cxy[μ]*Cxy[ν])*rootγ(Uxy)/_ds[1]/_ds[2]))

    end

    return int

end

@parallel_indices (x,y) function rhs!(S,U1,U2,U3,H,∂H,rm,θm,ns,dt,_ds,iter)

    #S = Float64
    #Explicit slices from main memory
    if iter == 1
        U = U1
        Uxy = U1[x,y]#::StateTensor{S}
    elseif iter == 2
        U = U2
        Uxy = U2[x,y]#::StateTensor{S}
    else
        U = U3
        Uxy = U3[x,y]#::StateTensor{S}
    end

    Hxy = H[x,y]; ∂Hxy = ∂H[x,y];

    r = rm[x,y]; θ = θm[x,y];

    # Give names to stored arrays from the state vector
    g  = Uxy.g 
    dr = Uxy.dr   
    dθ = Uxy.dθ  
    P  = Uxy.P 

    # Calculate inverse components in spherical
    gi = inv(g)

    # Calculate lapse and shift
    α = 1/sqrt(-gi[1,1])
    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    ∂tgs = βr*dr + βθ*dθ - α*P

    # Unpack the metric into indiviual components
    _,_,_,_,γs... = g.data

    γ = ThreeTensor{S}(γs)

    detγ = det(γ)

    if detγ < 0
        println(x," ",y," ",gxx," ",gyy," ",gzz," ")
    end

    γi3 = inv(γ)

    γi = StateTensor{S}((0.,0.,0.,0.,γi3.data...))

    nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 

    n = @Vec [nt,nr,nθ,0.0]

    n_ = @Vec [-α,0.0,0.0,0.0]

    ∂g = Tensor{Tuple{4,@Symmetry{4,4}},S}(
        (σ,μ,ν) -> (σ==1 ? ∂tgs[μ,ν] : σ==2 ? dr[μ,ν] : σ==3 ? dθ[μ,ν] : σ==4 ? 0. : @assert false)
        )

    Γ  = Tensor{Tuple{4,@Symmetry{4,4}},S}(
        (σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν])
        )

    C_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ] - Hxy[μ]


    # Cr = (Dr2(fg,U,r,θ,ns,_ds,x,y) - dr)
    # Cθ = (Dθ2(fg,U,r,θ,ns,_ds,x,y) - dθ)

    # C2 = Tensor{Tuple{4,@Symmetry{4,4}},S}((σ,μ,ν) -> (σ==1 ? 0.0 : σ==2 ? Cr[μ,ν] : σ==3 ? Cθ[μ,ν] : σ==4 ? 0. : @assert false))

    # if (x == 100 && y == 3 && iter==4) 
    #     display(C) 
    # end

    c1 = (x==1); c2 = (x==ns[1]);

    # Define Stress energy tensor and trace 
    T = zero(StateTensor{S})
    Tt = 0.

    δ = one(StateTensor{S})
    γ0 = 1.
    #γ1 = -1.
    γ2 = 1.

    ∂tP = 8*pi*Tt*g - 16*pi*T 

    ∂tP += 2*symmetric(@einsum (μ,ν) -> ∂Hxy[μ,ν])

    ∂tP -= 2*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*Hxy[ϵ]*∂g[μ,ν,σ])

    #∂tP -= @einsum (μ,ν) -> gi[ϵ,σ]*Hxy[ϵ]*∂g[ν,μ,σ]

    ∂tP += @einsum (μ,ν) -> 2*gi[ϵ,σ]*gi[λ,ρ]*∂g[λ,ϵ,μ]*∂g[ρ,σ,ν]

    ∂tP -= @einsum (μ,ν) -> 2*gi[ϵ,σ]*gi[λ,ρ]*Γ[μ,ϵ,λ]*Γ[ν,σ,ρ]

    ∂tP += γ0*(@einsum (μ,ν) -> C_[μ]*n_[ν] + C_[ν]*n_[μ] - g[μ,ν]*n[ϵ]*C_[ϵ]) # Constraint damping term for C_

    ∂tP *= α

    ∂tP -= @einsum (μ,ν) -> 0.5*γi[i,j]*∂tgs[i,j]*P[μ,ν]

    #∂tP -= γ2*βx*Cx + γ2*βz*Cz

    ###########################################
    # All finite differencing occurs here

    ∂tP += Div(vr,vθ,U,ns,_ds,x,y) # 

    ∂tdr = symmetric(Dr2(u,U,ns,x,y)*_ds[1]) # + α*γ2*Cr 

    ∂tdθ = symmetric(Dθ2(u,U,ns,x,y)*_ds[2]) # + α*γ2*Cθ

    #########################################
    ∂tP = symmetric(∂tP)

    #Boundary conditions

    if (c1 || c2) && false

        if c1 p=-1 else p=1 end

        s_ = @Vec [0.0,p/sqrt(gi[2,2]),0.,0.]
    
        s = @einsum gi[μ,ν]*s_[ν]
    
        rhat_ = @Vec [p/sqrt(γi[2,2]),0.,0.]
    
        rhat = @einsum γi3[i,j]*rhat_[j]
    
        θhat = @Vec [0.,1/sqrt(γ[2,2]),0.]
    
        θhat_ = @einsum γ[i,j]*θhat[j]
    
        #cp =  α - βx*r_hat[1] - βz*r_hat[3]
        cm = -α - βr*rhat_[1]
        c0 =    - βr*rhat_[1]
    
        #βθ = βx*θ_hat[1] + βz*θ_hat[3]
    
        # Up = P + rhat[1]*dx + rhat[3]*dz
        # U0 = θhat[1]*dx + θhat[3]*dz
    
        # Boundary Condition:
        # You get to choose the incoming 
        # characteristic modes (Um)
        # Pick a function Um = f(Up,U0)
    
        l = @einsum (n[α] + s[α])/sqrt(2)
        k = @einsum (n[α] - s[α])/sqrt(2)
    
        l_  = @einsum g[μ,α]*l[α]
        #Θ_  = @einsum g[μ,α]*Θ[α]
        #k_ = @einsum g[μ,α]*k[α]
    
        #σ = StateTensor((μ,ν) -> gi[μ,ν] + k[μ]*l[ν] + l[μ]*k[ν])
    
        σ_ = StateTensor((μ,ν) -> g[μ,ν] + n_[μ]*n_[ν] - s_[μ]*s_[ν])
    
        σ = @einsum gi[μ,α]*gi[ν,β]*σ_[α,β]
    
        σm = @einsum gi[μ,α]*σ_[ν,α] # mixed indices (raised second index)
    
        #δ4 = one(SymmetricFourthOrderTensor{4})
        δ = one(SymmetricSecondOrderTensor{4})

        γp = @einsum δ[μ,ν] + n_[μ]*n[ν] 

        Q4 = SymmetricFourthOrderTensor{4,S}(
            (μ,ν,α,β) -> σ_[μ,ν]*σ[α,β]/2 - 2*l_[μ]*σm[ν,α]*k[β] + l_[μ]*l_[ν]*k[α]*k[β]
        ) # Four index constraint projector (indices down down up up)
    
        Q3 = Tensor{Tuple{@Symmetry{4,4},4},S}(
            (μ,ν,α) -> l_[μ]*σm[ν,α] - σ_[μ,ν]*l[α]/2 - l_[μ]*l_[ν]*k[α]/2
        ) # Three index constraint projector (indices down down up)

        #Pij = @einsum δ3[i,j] - rhat[i]*r_hat[j]

        # O = SymmetricFourthOrderTensor{4}(
        #     (μ,ν,α,β) -> σm[μ,α]*σm[ν,β] - σ_[μ,ν]*σ[α,β]/2
        # ) # Gravitational wave projector
    
        # Pl = Tensor{Tuple{@Symmetry{4,4},4}}((μ,ν,α) -> l[μ]*δ[ν,α] - l_[α]*gi[μ,ν]/2)
    
        # Pθ = Tensor{Tuple{@Symmetry{4,4},4}}((μ,ν,α) -> Θ[μ]*δ[ν,α] - Θ_[α]*gi[μ,ν]/2)
    
        #Um1 = @einsum (sqrt(2)/2)*Pl[μ,ν,α]*Up[μ,ν] + Pθ[μ,ν,α]*U0[μ,ν] - Hxy[α]
    
        # Condition ∂tgμν = 0 on the boundary
        #Um2 = (cp/cm)*Up - 2*(βθ/cm)*U0
    
        #Um2 = P - rhat[1]*dx - rhat[3]*dz
        #-sqrt(2)*Q3[μ,ν,α]*Um1[α]
    
        #Um = @einsum -sqrt(2)*Q3[μ,ν,α]*Um1[α]# + δ4[μ,ν,α,β]*Um2[α,β] - Q4[μ,ν,α,β]*Um2[α,β]
        #Um = Um2

        #SAT type boundary conditions

        #ε = 2*abs(cm)*_ds[1]
    
        # Pb  = 0.5*(Up + Um)
        # dxb = 0.5*(Up - Um)*r_hat[1] + U0*θ_hat[1] 
        # dzb = 0.5*(Up - Um)*r_hat[3] + U0*θ_hat[3] 
    
        # ∂tPμν  += ε*(Pb - P)
        # ∂tdxμν += ε*(dxb - dx)
        # ∂tdzμν += ε*(dzb - dz)

        ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tgs[μ,ν])
    
        ∂tβ = α*(@einsum γi[α,μ]*n[ν]*∂tgs[μ,ν]) # result is a 4-vector
    
        ∂t∂tg = (βr*∂tdr + βθ*∂tdθ - α*∂tP) + (∂tβ[2]*dr + ∂tβ[3]*dθ - ∂tα*P)
    
        ∂t∂g = Tensor{Tuple{4,@Symmetry{4,4}},S}((σ,μ,ν) -> (σ==1 ? ∂t∂tg[μ,ν] : σ==2 ? ∂tdr[μ,ν] : σ==3 ? ∂tdθ[μ,ν] : σ==4 ? 0. : @assert false))
    

        ∂tΓ  = Tensor{Tuple{4,@Symmetry{4,4}},S}((σ,μ,ν) -> 0.5*(∂t∂g[ν,μ,σ] + ∂t∂g[μ,ν,σ] - ∂t∂g[σ,μ,ν]))   

        ∂tH = Vec{4}((∂Hxy[1,:]...))
        ∂rH = Vec{4}((∂Hxy[2,:]...))
        ∂θH = Vec{4}((∂Hxy[3,:]...))
    
        ∂tC = (@einsum gi[ϵ,σ]*∂tΓ[λ,ϵ,σ] - gi[μ,ϵ]*gi[ν,σ]*Γ[λ,μ,ν]*∂tgs[ϵ,σ]) - ∂tH
    
        # set up finite differencing for the constraints, by defining a function
        # that calculates the constraints for any x and y index. This
        # might not be the best idea, but should work.
    
        ∂rC = DrC(constraints,U,ns,x,y)*_ds[1] - ∂rH# + 0.5*γ2*(@einsum (n_[σ]*gi[μ,ν]*Cr[μ,ν] - n[ν]*Cr[σ,ν]))
        ∂θC = DθC(constraints,U,ns,x,y)*_ds[2] - ∂θH# + 0.5*γ2*(@einsum (n_[σ]*gi[μ,ν]*Cθ[μ,ν] - n[ν]*Cθ[σ,ν]))
    
        F = (∂tC - βr*∂rC - βθ*∂θC)/α# + γ2*(@einsum γi[μ,ν]*C2[μ,ν,λ] - 0.5*γp[λ,σ]*gi[μ,ν]*C2[σ,μ,ν])

        ∂Cm = F + rhat[1]*∂rC + rhat[2]*∂θC

        #c4rθ = Dr2(fdr,U,r,θ,ns,_ds,x,y) - Dθ2(fdθ,U,r,θ,ns,_ds,x,y)
        #c4θr = -c4rθ

        ∂tUp = ∂tP + rhat[1]*∂tdr + rhat[2]*∂tdθ# - γ2*∂tg   
        ∂tUm = ∂tP - rhat[1]*∂tdr - rhat[2]*∂tdθ# - γ2*∂tg
        ∂tU0 = θhat[1]*∂tdr + θhat[2]*∂tdθ

        #∂tU0 = ()∂tdx + ∂tdz

        ∂tUmb = @einsum Q4[μ,ν,α,β]*∂tUm[α,β]
        ∂tUmb -= sqrt(2)*cm*(@einsum Q3[μ,ν,α]*∂Cm[α]) # Constraint preserving BCs

        ∂tU0b = ∂tU0# + c0*(rhat[1]*θhat[2]*c4θr + rhat[2]*θhat[1]*c4rθ)

        #∂tUmb = @einsum O[μ,ν,α,β]*∂th[α,β] # Incoming Gravitational waveform

        # Time derivatives are OVERWRITTEN here, but still depends on evolution values
        ∂tP  = 0.5*(∂tUp + ∂tUmb)
        ∂tdr = 0.5*(∂tUp - ∂tUmb)*rhat_[1] + ∂tU0b*θhat_[1] 
        ∂tdθ = 0.5*(∂tUp - ∂tUmb)*rhat_[2] + ∂tU0b*θhat_[2] 

    end

    #############################

    ∂tP = symmetric(∂tP)

    ∂tU = SphericalStateVector(∂tgs,∂tdr,∂tdθ,∂tP)

    #return 58ns

    #display(∂tU.P)

    #∂tUc = TimeSphericalToCartesian(∂tU,Uxy,r,θ)

    # return 147ns

    #display(∂tUc.P)


    if iter == 1
        U2[x,y] = Uxy + dt*∂tU
    elseif iter == 2
        U3[x,y] = (3/4)*U1[x,y] + (1/4)*Uxy + (1/4)*dt*∂tU
    elseif iter == 3
        U1[x,y] = (1/3)*U1[x,y] + (2/3)*Uxy + (2/3)*dt*∂tU
    end

    # # End bit for SSP RK3
    # if iter == 1

    #     Uc1     = SphericalToCartesian(Uxy,r,θ)

    #     # 1.5 μs

    #     Uc2     = Uc1 + dt*∂tUc

    #     # 1.5 μs

    #     U2[x,y] = CartesianToSpherical(Uc2,r,θ)

    #     #return 2μs

    # elseif iter == 2

    #     Us1 = U1[x,y]
    #     Uc1     = SphericalToCartesian(Us1,r,θ)
    #     Uc2     = SphericalToCartesian(Uxy,r,θ)

    #     Uc3     = (3/4)*Uc1 + (1/4)*Uc2 + (1/4)*dt*∂tUc

    #     U3[x,y] = CartesianToSpherical(Uc3,r,θ)

    # elseif iter == 3

    #     Us1 = U1[x,y]
    #     Uc1     = SphericalToCartesian(Us1,r,θ)
    #     Uc3     = SphericalToCartesian(Uxy,r,θ)

    #     Uc1     = (1/3)*Uc1 + (2/3)*Uc3 + (2/3)*dt*∂tUc

    #     U1[x,y] = CartesianToSpherical(Uc1,r,θ)

    # end

    return
    
end

function RK4!(S,A,B,C,H,∂H,r,θ,ns,dt,_ds)

    nr,nθ = ns

    bulk = (1:nr,1:nθ)
    θs = (1:nr)

    #################################################
    # 4th order Runge-Kutta algoritm with only 
    # 3 main-memory registers (A,B,C)

    # First stage (iter=1)

    # @parallel θs   Symmetry_Conditions!(A,_ds)
    # @parallel bulk rhs!(C,A,H,∂H,r,θ,ns,dt,_ds,1) 

    # # Second stage (iter=2)

    # @parallel bulk add!(A,C,0.5)
    # @parallel bulk copy!(B,C)

    # @parallel θs   Symmetry_Conditions!(A,_ds)
    # @parallel bulk rhs!(C,A,H,∂H,r,θ,ns,dt,_ds,2) 

    # # Third stage (iter=3)

    # @parallel bulk update!(A,B,C,-0.5,0.5)

    # @parallel θs   Symmetry_Conditions!(A,_ds)
    # @parallel bulk rhs!(C,A,H,∂H,r,θ,ns,dt,_ds,3) 

    # # Fourth stage (iter=4)

    # @parallel bulk add!(A,C,1)
    # @parallel bulk combine!(B,B,C,(1/6),-1)

    # @parallel θs   Symmetry_Conditions!(A,_ds)
    # @parallel bulk rhs!(C,A,H,∂H,r,θ,ns,dt,_ds,4) 

    # @parallel bulk update!(A,B,C,1,(1/6))

    # result is stored in A

    # 48ms benchmark
    #################################################

    # Third order Strong Stability Preserving Runge-Kutta

    # First stage (iter=1)

    @parallel bulk rhs!(S,A,B,C,H,∂H,r,θ,ns,dt,_ds,1) 

    # Second stage (iter=2)

    @parallel bulk rhs!(S,A,B,C,H,∂H,r,θ,ns,dt,_ds,2) 

    # Third stage (iter=3)

    @parallel bulk rhs!(S,A,B,C,H,∂H,r,θ,ns,dt,_ds,3) 

    # Main bottleneck is update!(...)
    # perhaps because it has more main memory accesses
    # Is there any way we can improve its performance?

    return

end

@inline function P_init(g_init::Function,∂g_init::Function,r,θ,x,y)

    g   = StateTensor((μ,ν)->  g_init(r[x,y],θ[x,y],μ,ν)  )
    ∂tg = StateTensor((μ,ν)-> ∂g_init(r[x,y],θ[x,y],1,μ,ν))
    ∂rg = StateTensor((μ,ν)-> ∂g_init(r[x,y],θ[x,y],2,μ,ν))
    ∂θg = StateTensor((μ,ν)-> ∂g_init(r[x,y],θ[x,y],3,μ,ν))

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    return -(∂tg - βr*∂rg - βθ*∂θg)/α
    
end

function sample!(f, g, ∂g, ns, r, θ, T)

    for x in 1:ns[1], y in 1:ns[2]
        Us = SphericalStateVector{T}(
        StateTensor{T}((μ,ν) ->  g(r[x,y],θ[x,y]  ,μ,ν)),
        StateTensor{T}((μ,ν) -> ∂g(r[x,y],θ[x,y],2,μ,ν)),
        StateTensor{T}((μ,ν) -> ∂g(r[x,y],θ[x,y],3,μ,ν)),
        P_init(g,∂g,r,θ,x,y)
        )

        f[x,y] = Us
        #SphericalToCartesian(Us,r[x,y],θ[x,y])
    end

end

##################################################
function main()
    # Physics


    T = Data.Number


    rmin, rmax = 5.0, 10.0 
    tmin, tmax = 0.0, 50.


    numvar=4*7

    t         = 0.0      # physical time
    # Numerics
    #scale = 20 # normal amount to test with
    scale = 1

    nr, nθ    = scale*100, scale*100
    #nr, nθ    = 32*scale-1, 32*scale  # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf

    ns = (nr,nθ)

    # Derived numerics
    dr = (rmax-rmin)/(nr-1)
    dθ = pi/(nθ) # cell size for straddled axis, no cells at θ = 0,π

    # domains
    θmin, θmax = 10*dθ, pi- 10*dθ

    #dθ = (θmax-θmin)/(nθ-1) # cell size for included axis
    _dr, _dθ   = 1.0/dr, 1.0/dθ
    _ds = (_dr,_dθ)

    dt        = min(dr,rmin*dθ)/4.1

    #dts = (dt,dt/2,dt/3,dt/6)
    #dt       = min(dr,rmin*dθ)/4.0 #CFL

    #nt = (tmax-tmin)/dt+1
    nt = 100

    #nsave = Int(ceil(nt/nout))
    #nt=10

    nd = 4

    r  = @zeros(nr,nθ)
    θ  = @zeros(nr,nθ)

    r .= Data.Array([rmin + dr*(i-1) for i in 1:nr, j in 1:nθ])
    #θ .= Data.Array([θmin + dθ*(j-1) for i in 1:nr, j in 1:nθ])

    # xx .= Data.Array([r[i,j]*sin(θ[i,j]) for i in 1:nr, j in 1:nθ])
    # zz .= Data.Array([r[i,j]*cos(θ[i,j]) for i in 1:nr, j in 1:nθ])

    θ .= Data.Array([dθ/2 + dθ*(j-1) for i in 1:nr, j in 1:nθ])

    #return [r[100,2],θ[100,2]]

    # Un    = ArrayPartition([@zeros(nr,nθ) for i in 1:numvar]...)
    # Un1   = ArrayPartition([@zeros(nr,nθ) for i in 1:numvar]...)
    # k     = ArrayPartition([@zeros(nr,nθ) for i in 1:numvar]...)
    # ∂tU   = ArrayPartition([@zeros(nr,nθ) for i in 1:numvar]...)

    A  = StructArray{SphericalStateVector{T}}(undef,nr,nθ)
    B  = StructArray{SphericalStateVector{T}}(undef,nr,nθ)
    C  = StructArray{SphericalStateVector{T}}(undef,nr,nθ)

    # A  = StructArray{SphericalStateVector{T}}(undef,nr,nθ)
    # B  = StructArray{SphericalStateVector{T}}(undef,nr,nθ)
    # C  = StructArray{SphericalStateVector{T}}(undef,nr,nθ)

    # Hm     = ArrayPartition([@zeros(nr,nθ) for i in 1:(nd-1)]...)
    # ∂Hm    = ArrayPartition([@zeros(nr,nθ) for i in 1:(3*nd-1)]...)
    

    H  = StructArray{Tensor{Tuple{4}, T, 1, 4}}(undef,nr,nθ)
    ∂H = StructArray{SecondOrderTensor{4,T,16}}(undef,nr,nθ)


    # Define initial conditions

    M = 1
    sign = 1.

    # @inline e(r,θ,μ,ν) =  ((      1.      ,      0.     , 0.  ,        0.      ),
    #                        (      0.      ,      1.     , 0.  ,        0.      ),
    #                        (      0.      ,      0.     , r   ,        0.      ),
    #                        (      0.      ,      0.     , 0.  ,     r*sin(θ)   ))[μ][ν]
     
    @inline g_init(r,θ,μ,ν) =  (( -(1 - 2*M/r) , sign*2*M/r  , 0.  ,        0.        ),
                                (  sign*2*M/r  , (1 + 2*M/r) , 0.  ,        0.        ),
                                (      0.      ,      0.     , r^2 ,        0.        ),
                                (      0.      ,      0.     , 0.  ,    r^2*sin(θ)^2  ))[μ][ν]

    # Fully Harmonic
    #@inline ϵ(r) = (2*M)/(r*(1+M/r))
    # @inline g_init(r,θ,μ,ν) =  (( -1/(r^4*sin(θ)^2) , 0.  , 0.  ,        0.      ),
    #                             (  0.  , 1. , 0.  ,        0.      ),
    #                             (      0.      ,      0.             , r^2  ,        0.      ),
    #                             (      0.      ,      0.     , 0.  ,       r^2*sin(θ)^2      ))[μ][ν]

    ##################################################################################
    # Cartesian
    # @inline g_init(r,θ,μ,ν) =  (( -(1. - 2*M/r) , 2*M*sin(θ)/r       , 0.  , 2*M*cos(θ)/r        ),
    #                             (  2*M*sin(θ)/r , 1. +2*M*sin(θ)^2/r , 0.  , M*sin(2*θ)/r        ),
    #                             (      0.       ,      0.            , 1.  ,       0.            ),
    #                             (  2*M*cos(θ)/r ,  M*sin(2*θ)/r      , 0.  , 1. +M*(1+cos(2*θ))/r))[μ][ν]

    @inline ∂tg_init(r,θ,μ,ν) =  ((  0. ,  0.  ,  0.  ,  0.   ),
                                  (  0. ,  0.  ,  0.  ,  0.   ),
                                  (  0. ,  0.  ,  0.  ,  0.   ),
                                  (  0. ,  0.  ,  0.  ,  0.   ))[μ][ν]

    # Note: Assumes initial 3-metric is diagonal                           
    # @inline β(r,θ,i) = g_init(r,θ,1,i)/g_init(r,θ,i,i)  #+ g_init(r,θ,1,4)/g_init(r,θ,4,i)
    # @inline α(r,θ)   = sqrt(-g_init(r,θ,1,1) + g_init(r,θ,2,2)*β(r,θ,2)^2 + g_init(r,θ,3,3)*β(r,θ,3)^2 )

    @inline ∂rg(r,θ,μ,ν) = ForwardDiff.derivative(r -> g_init(r,θ,μ,ν), r)
    @inline ∂θg(r,θ,μ,ν) = ForwardDiff.derivative(θ -> g_init(r,θ,μ,ν), θ)
    
    @inline ∂g_init(r,θ,σ,μ,ν) = (∂tg_init(r,θ,μ,ν),∂rg(r,θ,μ,ν),∂θg(r,θ,μ,ν),0.0)[σ]
                                   
    # @inline P_init(r,θ,μ,ν) = -(∂ₜg_init(r,θ,μ,ν) - β(r,θ,2)*d_init(r,θ,2,μ,ν) - β(r,θ,3)*d_init(r,θ,3,μ,ν))/α(r,θ)

    # @inline ∂g(r,θ,σ,μ,ν) = if σ==1 β(r,θ,2)*d_init(r,θ,2,μ,ν) + β(r,θ,3)*d_init(r,θ,3,μ,ν) - α(r,θ)*P(r,θ,μ,ν) else d(r,θ,σ,μ,ν) end

    # # Define completely covariant Christoffel symbols
    # @inline Γ(r,θ,σ,μ,ν) = (∂g(r,θ,ν,μ,σ) + ∂g(r,θ,μ,ν,σ) - ∂g(r,θ,σ,μ,ν))/2.

    # # This is annoying to calculate on the fly
    # @inline fH(r,θ,μ)  = (-2*M/r^2, 2*(M-r)/r^2,-cos(θ)/sin(θ)/r^2,0.)[μ] # upper index

    # fr(x,z) = sqrt(x^2+z^2)
    # fθ(x,z) = acos(x/fr(x,z))
    #fHt(x,z) = 0.0
    #fHt(x,z) = (10^-3)*exp(-(1/2)*((x-7.5)/0.1)^2)*exp(-(1/2)*(z/0.1)^2)

    # Fully Harmonic
    #@inline fH_(x,z,μ) = (fHt(x,z),0.,0.,0.)[μ] # lower index

    #@inline fH(r,θ,μ)  = (, 0.,0.,0.)[μ] # upper index

    # @inline f∂H(r,θ,μ,ν) = ((0.,0.,0.,0.)[ν],
    #                         (0.,0.,0.,0.)[ν],
    #                         (0.,0.,0.,0.)[ν],
    #                         (0.,0.,0.,0.)[ν])[μ]

    # @inline f∂H_(x,z,μ,ν) = ((0.,0.,0.,0.)[ν],
    #                         (ForwardDiff.derivative(x -> fHt(x,z), x),0.0,0.0,0.0)[ν],
    #                         (0.,0.,0.,0.)[ν],
    #                         (ForwardDiff.derivative(z -> fHt(x,z), z),0.0,0.0,0.0)[ν])[μ]

    @inline fH_(r,θ,μ) = (-2*M/r^2,-2*(M+r)/r^2,-cos(θ)/sin(θ),0.)[μ] # lower index

    @inline f∂H_(r,θ,μ,ν) = ((0.,0.,0.,0.)[ν],
                            ForwardDiff.derivative(r -> fH_(r,θ,ν), r),
                            ForwardDiff.derivative(θ -> fH_(r,θ,ν), θ),
                            (0.,0.,0.,0.)[ν])[μ]

    #@inline f∂H_sym(r,θ,μ,ν) = 0.5*(∂H(r,θ,μ,ν)+∂H(r,θ,ν,μ))

    sample!(A, g_init, ∂g_init, ns, r, θ, T)

    B .= A
    C .= A

    #return (A[1,1].dθ)[4,4] - 2*r[1,1]^2*sin(θ[1,1])*cos(θ[1,1])

    for i in 1:ns[1], j in 1:ns[2]
        #Λ = StateTensor([e(r,θ,μ,ν)])
        H[i,j] = @Vec [fH_(r[i,j],θ[i,j],μ) for μ in 1:4]
        ∂H[i,j] = SecondOrderTensor{4,T,16}((μ,ν) -> f∂H_(r[i,j],θ[i,j],μ,ν))
    end

    # x=75;y=75;

    # display(A[x,y].g)
    # display(A[x,y].dx)
    # display(A[x,y].dy)
    # display(A[x,y].dz)
    # display(A[x,y].P)

    # return
    #return rhs!(A,B,C,H,∂H,r,θ,ns,dt,_ds,1)

    #println(SphericalToCartesian(A[75,75],r[75,75],θ[75,75]))

    #Uc = SphericalToCartesian(A[75,75],r[75,75],θ[75,75])

    #return @benchmark SphericalToCartesian($A[75,75],$r[75,75],$θ[75,75])

    #return @benchmark CartesianToSpherical($Uc,$r[75,75],$θ[75,75])

    #return rhs!(A,B,C,H,∂H,r,θ,ns,dt,_ds,1)

    #return @code_warntype SphericalToCartesian(A[75,75],r[75,75],θ[75,75])

    #return @code_warntype rhs!(A,B,C,H,∂H,r,θ,ns,dt,_ds,1)

    #return @benchmark rhs!($A,$B,$C,$H,$∂H,$r,$θ,$ns,$dt,$_ds,1)

    # println("r = ", r[75,75])
    # println("θ = ", θ[75,75])

    # Ui= A[75,75]

    # Uc = SphericalToCartesian(Ui,r[75,75],θ[75,75])

    # Us = CartesianToSpherical(Uc,r[75,75],θ[75,75])

    # display(Us.g - Ui.g)
    # display(Us.dr - Ui.dr)
    # display(Us.dθ - Ui.dθ)
    # display(Us.P - Ui.P)

    #return 

    # Umi1 = [zero(StateTensor) for i in 1:nθ]
    # Umin = [zero(StateTensor) for i in 1:nθ]

    #return display(H[100,1])

    #@parallel (1:10,1:10) rhs!(A,B,H,∂H_sym,r,θ,ns,_ds,1)

    xout, yout = 50,50

    # temp = A[xout,yout].P

    μi,νi = 1,1

    temp_array = getindex.(A.P,μi,νi)

    nout = 1 #round(nt/100)          # plotting frequency

    ENV["GKSwstype"]="nul";
    path = string("viz2D_out")
    mkpath(path);
    old_files = readdir(path; join=true)
    for i in 1:length(old_files) rm(old_files[i]) end
    loadpath = string("./",path,"/")
    anim = Animation(loadpath,String[])
    println("Animation directory: $(anim.dir)")
    #if isopen(file) close(file) end
    old_files = readdir(path; join=true)
    for i in 1:length(old_files) 
        rm(old_files[i]) 
    end

    ints = zeros(0)

    for i in 1:10


        if mod(i,nout)==0 || i == 1
            res=1;

            # println("")
            # #display(A[xout,yout].P -temp)
            # display(Dθ2(u,A,ns,xout,yout))
            # #println(A[xout,yout].P -temp)
            # println("")

            fg(x,y,U) = U[x,y].g
            
            # 2D slice
            #data = [constraints(A[x,50]) - H[x,50] for x in 1:nr]

            # r slice
            #data = [constraints(x,50,A)- H[x,50] for x in 1:nr]
            #data = [ [getindex(Dx2(fg,A,r,θ,ns,_ds,x,50) - A[x,50].dr,1,i) for i in 1:4] for x in 1:nr]

            # θ slice
            data = [constraints(A[50,y]) - H[50,y] for y in 1:nθ]

            #return typeof(Array(data))
            labels = ["Ct" "Cr" "Cθ" "Cϕ"]

            #data = getindex.(A[:,50].g,μi,νi) - temp_array[:,50]

            #reduce(hcat,data)'

            plot(Array(r[:,50]), reduce(hcat,data)', label=labels, title = "Time = "*string(round(t; digits=2)) )
            ylims!(-10^-4, 10^-4)
            frame(anim)

            #println(t)

            # plot(Array(θ[1,1:res:end]), Array(getindex.(A[100,:].P,μi,νi).-temp_array[100,:]))
            # ylims!(-0.001, 0.001)
            # frame(anim)
            # println(mean(∂ₜU.x[var][2,1:res:end]))

            # c = @zeroes(nr,nθ)
            # @. c = sqrt(ψr^2 + ψθ^2)
            #φ[1:res:end,1:res:end]
            # nr = ψr/c
            # heatmap(Array(r[1:res:end,1]), Array(θ[1,1:res:end]), Array(∂ₜU.x[3][1:res:end,1:res:end])', 
            # aspect_ratio=1, xlims=(rmin,rmax), ylims=(θmin,θmax),clim=(-1,1), c=:viridis); 

            # 2D Constraints
            # heatmap(Array(r[1:res:end,1]), Array(θ[1,1:res:end]), Array(data)', 
            # aspect_ratio=1, xlims=(rmin,rmax), ylims=(0,pi),clim=(-10^(-3),10^(-3)), c=:viridis); 

            # frame(anim)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection="polar")
            # ρ = LinRange(0., 7, 200)
            # θ = LinRange(0., 2π, 360)
            # funp(ρ,θ) =  sin(2ρ) * cos(θ)
            # pc = pcolormesh(θ, ρ, funp.(ρ,θ'))
            # cbar = plt.colorbar(pc)
            # cbar.set_label("Intensity")
            # ax[:grid](true)

            # plot(Array(r[1:res:end,1]), Array(-ψr[1:res:end,1]./Π[1:res:end,1])); 
            # ylims!(-1.5, 1.5)

            #

            # φdata[iter,:,:] = Array(φ)
            # iter += 1

        end

        append!(ints,integrate_constraints(A,H,ns,_ds))

        RK4!(T,A,B,C,H,∂H,r,θ,ns,dt,_ds)

        t += dt
    end

    plot(ints, yaxis=:log)
    ylims!(10^-10, 10)
    frame(anim)

    gif(anim, "tests.gif", fps = 30)

    return #A[100,100].P - temp

    #return @benchmark @parallel (1:10,1:10) rhs!($A,$B,$H,$∂H,$r,$θ,$ns,$_ds,1)

    # @parallel (1:10,1:10) RK4!(Un,Un1,H,∂H_sym,r,θ,ns,dt,_ds)

    # return Un1[1,1]

    # return

    # return @code_warntype @parallel (1:10,1:10) RK4!($Un,$Un1,$H,$∂H_sym,$r,$θ,$ns,$dt,$_ds)

    #return @code_warntype @parallel (1:10,1:10) RK4!($A,$B,$C,$H,$∂H,$r,$θ,$ns,$dt,$_ds)

    #return @benchmark RK4!($A,$B,$C,$H,$∂H,$r,$θ,$ns,$dt,$_ds)

    #return @benchmark @parallel (1:$nr,1:$nθ) RK4!($Un,$Un1,$H,$∂H_sym,$r,$θ,$ns,$dt,$_ds)

    #@parallel rhs!(∂ₜU,U,r,θ,ql,qr,dt,dr,dθ)

    #return @macroexpand @part (1,2) (φ + α)

    # @part (1,2) P = 1

    # return  P[1:10,1:10]
    #return @benchmark @parallel compute_P!($P, $Vx, $Vy, $qr, $ql, $dt, $k, $dx, $dy)
    #return
    path = string("2dData")
    mkpath(path);
    # file = h5open(path*"/file.h5","cw")
    # close(file)

    # R, Θ      = rmin:dr:rmax, dθ:dθ:pi


    #Preparation of visualisation
    if VISUAL
        ENV["GKSwstype"]="nul";
        path = string("viz2D_out")
        mkpath(path);
        old_files = readdir(path; join=true)
        for i in 1:length(old_files) rm(old_files[i]) end
        loadpath = string("./",path,"/")
        anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        #if isopen(file) close(file) end
        old_files = readdir(path; join=true)
        for i in 1:length(old_files) 
            rm(old_files[i]) 
        end

        # path = string("2dData")
        # old_files = readdir(path; join=true)
        # for i in 1:length(old_files) 
        #     rm(old_files[i]) 
        # end
        # datafile = h5open(path*"/data.h5","cw")
        # φdata = create_dataset(datafile, "phi", datatype(Data.Number), dataspace(nsave,nr,nθ), chunk=(1,nr,nθ))
        # φdata[1,:,:] = Array(φ)

        # coordsfile = h5open(path*"/coords.h5","cw")
        # coordsfile["r"] = Array(R)
        # coordsfile["theta"]  = Array(Θ)
        
    end
    #return
    try
    # Time loop
    iter=2
    global wtime0 = Base.time()
    res=1

    for it = 1:nt
        #if (it==11)  global wtime0 = Base.time()  end
        # @parallel compute_V!(P,Vr,Vθ,r,θ,ql,qr,dt,dr,dθ)
        # @parallel compute_P!(P,Vr,Vθ,r,θ,ql,qr,dt,dr,dθ)

        # @parallel (1:nr,1:nθ)  rhs!(k,U,metric,t,r,θ,ql,qr,dt,dr,dθ,nr,nθ) #Calculate right hand side and store it in k
        # @parallel (1:nr)       BCθ!(k,U,metric,dθ)
        # @parallel (1:nθ)       BCr!(k,U,metric,t,rmax,θ[1,:],dr)
        # @parallel (1:nr,1:nθ)  lincomb!(k,U,k,1.,dt/2)
        # @parallel (1:nr,1:nθ)  rhs!(∂ₜU,k,metric,t+dt/2,r,θ,ql,qr,dt,dr,dθ,nr,nθ) #Calculate right hand side and store it in ∂ₜU
        # @parallel (1:nr)       BCθ!(∂ₜU,k,metric,dθ)
        # @parallel (1:nθ)       BCr!(∂ₜU,k,metric,t+dt/2,rmax,θ[1,:],dr)
        # @parallel (1:nr,1:nθ)  lincomb!(U,U,∂ₜU,1.,dt)

        #rhs!(∂ₜU,U,gauge,rm,θm,nr,nθ,_dr,_dθ)
        # res=1;
        # plot(Array(θ[2,1:res:end]), Array(U.x[11][2,1:res:end])); 
        # ylims!(1.5, 1.8)
        # frame(anim)
        # 6 fine, 7 fine, 9 fine, 10 fine
        # Kθθ  negative spikes
        # fθθθ positive spikes, and a mid spike
        # frθθ positive spikes
        #γrr,γθθ,γϕϕ,Krr,Kθθ,Kϕϕ,frrr,frθθ,frϕϕ,fθrr,fθθθ,fθϕϕ = U.x
        var = 11

        # Fourth Order Explicit Runge-Kutta


        # plot(Array(θ[2,1:res:end]), Array(∂ₜU.x[var][1:res:end,1])); 
        # ylims!(-100, 100)
        # frame(anim)
        # println(mean(∂ₜU.x[var][1:res:end,1]))

        # plot(Array(θ[2,1:res:end]), Array(∂ₜU.x[var][1:res:end,1])); 
        # ylims!(-100, 100)
        # frame(anim)
        # println(mean(∂ₜU.x[var][1:res:end,1]))

        # res=1;
        # plot(Array(θ[2,1:res:end]), Array(U.x[11][2,1:res:end])); 
        # ylims!(1.5, 1.8)
        # frame(anim)
        # if rand(1:Int(round(nt/10)))==1
        #     x,y = rand(rmin+2:rmax-2), rand(θmin+2:θmax-2)
        #     φ  .+= Data.Array([f(r,θ,x,y)  for r=rmin:dr:rmax, θ=θmin:dθ:θmax])
        #     ψr .+= Data.Array([fr(r,θ,x,y) for r=rmin:dr:rmax, θ=θmin:dθ:θmax])
        #     ψθ .+= Data.Array([fθ(r,θ,x,y) for r=rmin:dr:rmax, θ=θmin:dθ:θmax])
        # end

        t = t + dt

        res=1;
        # heatmap(Array(r[1:res:end,1]), Array(θ[1,1:res:end]), Array(∂ₜU.x[3][1:res:end,1:res:end])', 
        # aspect_ratio=1, xlims=(rmin,rmax), ylims=(θmin,θmax),clim=(-1,1), c=:viridis); 

        # plot(Array(θ[2,1:res:end]), Array(U.x[2][2,1:res:end])); 
        # #println(Array(U.x[2][2,1:res:end]))
        # ylims!(24., 26.)

        # frame(anim)

        #Visualisation

        if mod(it,nout)==0 || it == 1
            res=1;

            plot(Array(θ[2,1:res:end]), Array(∂ₜU.x[var][2,1:res:end])); 
            ylims!(-0.1, 0.1)
            frame(anim)
            println(mean(∂ₜU.x[var][2,1:res:end]))

            # c = @zeroes(nr,nθ)
            # @. c = sqrt(ψr^2 + ψθ^2)
            #φ[1:res:end,1:res:end]
            # nr = ψr/c
            # heatmap(Array(r[1:res:end,1]), Array(θ[1,1:res:end]), Array(∂ₜU.x[3][1:res:end,1:res:end])', 
            # aspect_ratio=1, xlims=(rmin,rmax), ylims=(θmin,θmax),clim=(-1,1), c=:viridis); 

            # heatmap(Array(r[1:res:end,1]), Array(θ[1,1:res:end]), Array(-ψr[1:res:end,1:res:end]./Π[1:res:end,1:res:end])', 
            # aspect_ratio=1, xlims=(rmin,rmax), ylims=(θmin,θmax),clim=(-1,1), c=:viridis); 

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection="polar")
            # ρ = LinRange(0., 7, 200)
            # θ = LinRange(0., 2π, 360)
            # funp(ρ,θ) =  sin(2ρ) * cos(θ)
            # pc = pcolormesh(θ, ρ, funp.(ρ,θ'))
            # cbar = plt.colorbar(pc)
            # cbar.set_label("Intensity")
            # ax[:grid](true)

            # plot(Array(r[1:res:end,1]), Array(-ψr[1:res:end,1]./Π[1:res:end,1])); 
            # ylims!(-1.5, 1.5)

            #frame(anim)

            # φdata[iter,:,:] = Array(φ)
            # iter += 1

        end
    end

    catch error
        # close(datafile)
        # close(coordsfile)
        throw(error)
    end
    
    # close(datafile)
    # close(coordsfile)

    # Performance
    wtime    = Base.time()-wtime0
    A_eff    = (3*2)/1e9*nr*nθ*sizeof(Data.Number)  # Effective main memory access per iteration [GB] 
    wtime_it = wtime/(nt-10)                        # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                       # Effective memory throughput [GB/s]
    @printf("Total steps=%d, time=%1.2e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=3))
    if VISUAL gif(anim, "acoustic2D.gif", fps = 15) end
    return
end

end