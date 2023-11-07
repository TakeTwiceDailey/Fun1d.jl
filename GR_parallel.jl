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
const StateTensor{T} = SymmetricSecondOrderTensor{3,T,6}

# Alias for SymmetricSecondOrderTensor 3x3
const TwoTensor{T} = SymmetricSecondOrderTensor{2,T,3}

# Alias for non-symmetric 4 tensor
const ThreeTensor{T} = SecondOrderTensor{3,T,9}

# Alias for non-symmetric 4 tensor
const StateScalar{T} = Vec{4,T}

# Alias for tensor to hold metric derivatives and Christoffel Symbols
# Defined to be symmetric in the last two indices
const Symmetric3rdOrderTensor{T} = Tensor{Tuple{3,@Symmetry{3,3}},T,3,18}

# This struct type is used to package the state vector into 
# spherical components. This is used to store the state vector
# into memory, and does not store ϕ derivatives as those vanish in axisymmetry.
# Another 30% memory savings is possible if we exclude ϕ coponents of tensors,
# but that is not implemented.
struct StateVector{T}
    ψ::StateScalar{T}
    g::StateTensor{T}
    dr::StateTensor{T}
    dθ::StateTensor{T}
    P::StateTensor{T}
end

# Define math operators for StateVector
math_operators = [:+, :-, :*, :/, :^]
for op in math_operators
    @eval import Base.$op
    @eval @inline function $op(A::StateVector{T},B::StateVector{T}) where T
        ψ  = @. $op(A.ψ,B.ψ)
        g  = @. $op(A.g,B.g)
        dr = @. $op(A.dr,B.dr)
        dθ = @. $op(A.dθ,B.dθ)
        P  = @. $op(A.P,B.P)
        return StateVector{T}(ψ,g,dr,dθ,P)
    end

    @eval @inline function $op(a::Number,B::StateVector{T}) where T
        ψ  = $op(a,B.ψ)
        g  = $op(a,B.g)
        dr = $op(a,B.dr)
        dθ = $op(a,B.dθ)
        P  = $op(a,B.P)
        return StateVector{T}(ψ,g,dr,dθ,P)
    end
end

@inline function Base.zero(::Type{StateVector{T}}) where T
    ψ  = zero(StateScalar{T})
    g  = zero(StateTensor{T})
    dr = zero(StateTensor{T})
    dθ = zero(StateTensor{T})
    P  = zero(StateTensor{T})
    return StateVector{T}(ψ,g,dr,dθ,P)
end

# Define functions to return Stuct components for finite 
# differencing in constraint calculations
@inline fg(U::StateVector)  = U.g
@inline fdr(U::StateVector) = U.dr
@inline fdθ(U::StateVector) = U.dθ
@inline fP(U::StateVector)  = U.P


const parity  = StateTensor{Data.Number}((1,-1,1,1,-1,1))
const ψparity = Vec{4}((1,-1,1,1))
const StateVectorParity = StateVector{Data.Number}(ψparity,parity,-parity,parity,parity)
const parityC = Vec{3}((1,-1,1))

# @inline function DrC(f::Function,U,ns,x,y) 
#     n = ns[1]
#     if x in 2:n-1
#         -0.5*f(U[x-1,y])+0.5*f(U[x+1,y])
#     elseif x==1
#         -f(U[1,y]) + f(U[2,y])
#     else#if x==n 
#         -f(U[n-1,y]) + f(U[n,y])
#     end
#     #else @assert false end
# end

# @inline function DθC(f::Function,U,ns,x,y)
#     n = ns[2]
#     if y in 2:n-1
#         -0.5*f(U[x,y-1]) + 0.5*f(U[x,y+1])
#     elseif y==1
#         #-(3/2)*f(x,1,U) + 2*f(x,2,U) - (1/2)*f(x,3,U)
#         -0.5*parityC.*f(U[x,1]) + 0.5*f(U[x,2])
#         #-f(x,1,U) + f(x,2,U)
#     elseif y==n
#         -0.5*f(U[x,n-1]) + 0.5*parityC.*f(U[x,n]) 
#         #-f(x,n-1,U) + f(x,n,U)
#     end
# end

@inline function Dissipation(U,x,y,ns)

    # ψv = U.ψ
    # g  = U.g 
    # dr = U.dr 
    # dθ = U.dθ
    # P  = U.P 

    # ψ,ψr,ψθ,Ψ = ψv.data

    # ∂ψ = @Vec [∂ψ,∂ψr,∂ψθ,∂Ψ]

    # A = StateVector{T}((∂ψ,∂g,∂dr,∂dθ,∂P))
    nx,ny = ns
    ϵ = 2.
    if x in 2:nx-1 && y in 2:ny-1 
        (ϵ/4)*(U[x-1,y] + U[x+1,y] - 4*U[x,y] + U[x,y-1] + U[x,y+1])
    # elseif x==1 && y ≠ 1 && y ≠ ny
    #     (ϵ/4)*(2*U[2,y] - 4*U[1,y] + U[1,y-1] + U[1,y+1])
    # elseif x==nx && y ≠ 1 && y ≠ ny
    #     (ϵ/4)*(2*U[nx-1,y] - 4*U[nx,y] + U[nx,y-1] + U[nx,y+1])
    elseif y == 1 && x ≠ nx && x ≠ 1
        (ϵ/4)*(U[x-1,1] + U[x+1,1] - 4*U[x,1] + StateVectorParity*U[x,2] + U[x,2])
    elseif y == ny && x ≠ nx && x ≠ 1
        (ϵ/4)*(U[x-1,ny] + U[x+1,ny] - 4*U[x,ny] + StateVectorParity*U[x,ny-1] + U[x,ny-1])
    elseif x==1 && y==1
        (ϵ/4)*(2*U[2,1] - 4*U[1,1] + StateVectorParity*U[1,2] + U[1,2])
    elseif x==nx && y==1
        (ϵ/4)*(2*U[nx-1,1] - 4*U[nx,1] + StateVectorParity*U[nx,2] + U[nx,2])
    elseif x==1 && y==ny
        (ϵ/4)*(2*U[2,ny] - 4*U[1,ny] + StateVectorParity*U[1,ny-1] + U[1,ny-1])
    elseif x==nx && y==ny
        (ϵ/4)*(2*U[nx-1,ny] - 4*U[nx,ny] + StateVectorParity*U[nx,ny-1] + U[nx,ny-1])
    else
        zero(StateVector{Data.Number})
    end
end

@inline function DrC(f::Function,U,ns,x,y) 
    n = ns[1]
    if x in 2:n-1
        -0.5*f(U[x-1,y])+0.5*f(U[x+1,y])
    elseif x==1
        -f(U[1,y]) + f(U[2,y])
    elseif x==n 
        -f(U[n-1,y]) + f(U[n,y])
    end
end

@inline function DθC(f::Function,U,ns,x,y,p=1)
    n = ns[2]
    if y in 2:n-1
        -0.5*f(U[x,y-1]) + 0.5*f(U[x,y+1])
    elseif y==1
        -0.5*parityC.*f(U[x,2]) + 0.5*f(U[x,2]) #includes the axis
        #-0.5*parityC.*f(U[x,1]) + 0.5*f(U[x,2])
    elseif y==n
        -0.5*f(U[x,n-1]) + 0.5*parityC.*f(U[x,n-1]) #includes the axis
        #-0.5*f(U[x,n-1]) + 0.5*parityC.*f(U[x,n]) 
    end
end

@inline function DρC(f::Function,U,r,θ,ns,_ds,x,y,p=1) 
    DrC(f,U,ns,x,y)*_ds[1]*sin(θ) + DθC(f,U,ns,x,y,p)*_ds[2]*cos(θ)/r
end

@inline function DzC(f::Function,U,r,θ,ns,_ds,x,y,p=1)
    DrC(f,U,ns,x,y)*_ds[1]*cos(θ) - DθC(f,U,ns,x,y,p)*_ds[2]*sin(θ)/r
end

@inline function Dr2(f::Function,U,ns,x,y)
    n = ns[1]
    if x in 2:n-1
        0.5*(-f(U[x-1,y]) + f(U[x+1,y]))
    elseif x==1
        -f(U[1,y]) + f(U[2,y])
    elseif x==n 
        -f(U[n-1,y]) + f(U[n,y])
    end
end

@inline function Dθ2(f::Function,U,ns,x,y,p)
    n = ns[2]
    if y in 2:n-1
        0.5*(-f(U[x,y-1]) + f(U[x,y+1]))
    elseif y==1
        0.5*(-p*f(U[x,2]) + f(U[x,2])) # includes the axis
        #0.5*(-p*f(U[x,1]) + f(U[x,2]))
    elseif y==n
        0.5*(-f(U[x,n-1]) + p*f(U[x,n-1])) # includes the axis
        #0.5*(-f(U[x,n-1]) + p*f(U[x,n]))
    end
end


@inline function Dρ2(f::Function,U,r,θ,ns,_ds,x,y,p=1) 
    Dr2(f,U,ns,x,y)*_ds[1]*sin(θ) + Dθ2(f,U,ns,x,y,p)*_ds[2]*cos(θ)/r
end

@inline function Dz2(f::Function,U,r,θ,ns,_ds,x,y,p=1)
    Dr2(f,U,ns,x,y)*_ds[1]*cos(θ) - Dθ2(f,U,ns,x,y,p)*_ds[2]*sin(θ)/r
end

@inline function Div(vr::Function,vθ::Function,U,r,θ,ns,_ds,x,y)
    (Dρ2(vr,U,r,θ,ns,_ds,x,y,-1) + Dz2(vθ,U,r,θ,ns,_ds,x,y,-1))/rootγ(U[x,y])
end

@inline function Aθ2T(f::Function,U,r,θ,ns,x,y)
    n = ns[2]
    if y in 3:n-2
        0.5*(f(U,r,θ,x,y-1) + f(U,r,θ,x,y+1))
    elseif y==2
        0.5*(f(U,r,θ,x,2) + f(U,r,θ,x,y+1))
    elseif y==n-1
        0.5*(f(U,r,θ,x,n-2) + f(U,r,θ,x,n-1))
    elseif y==1
        f(U,r,θ,x,2)
    elseif y==n
        f(U,r,θ,x,n-1)
    end
end

@inline function Ar2T(f::Function,U,r,θ,ns,x,y)
    n = ns[1]
    if y==1 y+=1 end
    if y==n y-=1 end
    if x in 2:n-1
        0.5*(f(U,r,θ,x-1,y) + f(U,r,θ,x+1,y))
    elseif x==1
        f(U,r,θ,2,y)
    elseif x==n 
        f(U,r,θ,n-1,y)
    end
end

@inline function A2T(f::Function,U,r,θ,ns,x,y)
    0.5*(Ar2T(f,U,r,θ,ns,x,y) + Aθ2T(f,U,r,θ,ns,x,y))
    # nx,ny = ns
    # if x in 2:nx-1 && y in 2:ny-1
    #     0.25*(f(U,x,y-1) + f(U,x,y+1) + f(U,x-1,y) + f(U,x+1,y))
    # elseif x==1 && y in 2:ny-1
    #     0.25*(f(U,x,y-1) + f(U,x,y+1)) + 0.5*f(U,x+1,y)
    # elseif x==nx && y in 2:ny-1
    #     0.25*(f(U,x,y-1) + f(U,x,y+1)) + 0.5*f(U,x-1,y)
    # elseif x in 2:nx-1 && y == 1
    #     0.5*f(U,x,y+1) + 0.25*(f(U,x-1,y+1) + f(U,x+1,y+1))
    # elseif x in 2:nx-1 && y == ny
    #     0.5*f(U,x,y-1) + 0.25*(f(U,x-1,y-1) + f(U,x+1,y-1))
    # elseif x==1 && y==1
    #     0.5*(f(U,x,y+1) + f(U,x+1,y+1))
    # elseif x==nx && y==1
    #     0.5*(f(U,x,y+1) + f(U,x-1,y+1))
    # elseif x==1 && y==ny
    #     0.5*(f(U,x,y-1) + f(U,x+1,y-1))
    # elseif x==nx && y==ny
    #     0.5*(f(U,x,y-1) + f(U,x-1,y-1))
    # end
    #     @assert false
    # end
end

@inline function Dr2T(f::Function,U,ns,x,y)
    n = ns[1]
    if x in 2:n-1
        0.5*(-f(U[x-1,y]) + f(U[x+1,y]))
    elseif x==1
        -f(U[1,y]) + f(U[2,y])
    elseif x==n 
        -f(U[n-1,y]) + f(U[n,y])
    end
    #else @assert false end
end

@inline function Dθ2T(f::Function,U,ns,x,y,p)
    n = ns[2]
    if y in 2:n-1
        0.5*(-f(U[x,y-1]) + f(U[x,y+1]))
    elseif y==1
        -0.5*p*parity.*f(U[x,2]) + 0.5*f(U[x,2]) # includes the axis
        #0.5*(-p*parity.*f(U[x,1]) + f(U[x,2]))
    elseif y==n
        -0.5*f(U[x,n-1]) + 0.5*p*parity.*f(U[x,n-1]) # includes the axis
        #0.5*(-f(U[x,n-1]) + p*parity.*f(U[x,n]))
    end
end

# @inline function Divθ2(f::Function,U,ns,x,y)
#     n = ns[2]
#     if y in 2:n-1
#         -0.5*f(U[x,y-1]) + 0.5*f(U[x,y+1])
#     elseif y==1
#         #0.5*parity.*f(U[x,1]) + 0.5*f(U[x,2])
#     else#if y==n
#         -0.5*f(U[x,n-1]) - 0.5*parity.*f(U[x,n]) 
#     end
#     #else @assert false end
# end

@inline function Dρ2T(f::Function,U,r,θ,ns,_ds,x,y,p=1) 
    Dr2T(f,U,ns,x,y)*_ds[1]*sin(θ) + Dθ2T(f,U,ns,x,y,p)*_ds[2]*cos(θ)/r
end

@inline function Dz2T(f::Function,U,r,θ,ns,_ds,x,y,p=1)
    Dr2T(f,U,ns,x,y)*_ds[1]*cos(θ) - Dθ2T(f,U,ns,x,y,p)*_ds[2]*sin(θ)/r
end

# @inline function Div(vr::Function,vθ::Function,U,ns,_ds,x,y)
#     (Dr2(vr,U,ns,x,y)*_ds[1] + Dθ2(vθ,U,ns,x,y)*_ds[2])/rootγ(U[x,y])
# end

@inline function DivT(vr::Function,vθ::Function,U,r,θ,ns,_ds,x,y)
    (Dρ2T(vr,U,r,θ,ns,_ds,x,y,-1) + Dz2T(vθ,U,r,θ,ns,_ds,x,y,-1))/rootγ(U[x,y])
end

@inline function ψu(U::StateVector) # Scalar gradient-flux

    # Give names to stored arrays from the state vector
    ψv = U.ψ
    g  = U.g 

    ψ,ψr,ψθ,Ψ = ψv.data

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    return βr*ψr + βθ*ψθ - α*Ψ

end

@inline function ψvr(U::StateVector) # r component of the divergence-flux

    # Give names to stored arrays from the state vector
    ψv = U.ψ
    g  = U.g 

    ψ,ψr,ψθ,Ψ = ψv.data

    _,_,_,γs... = g.data

    γ = TwoTensor(γs)

    γi = inv(γ)

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]

    return rootγ(U)*(βr*Ψ - α*(γi[1,1]*ψr + γi[1,2]*ψθ))
    
end

@inline function ψvθ(U::StateVector) # θ component of the divergence-flux

    # Give names to stored arrays from the state vector
    ψv = U.ψ
    g  = U.g 

    ψ,ψr,ψθ,Ψ = ψv.data

    _,_,_,γs... = g.data

    γ = TwoTensor(γs)

    γi = inv(γ)

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βθ = -gi[1,3]/gi[1,1]

    return rootγ(U)*(βθ*Ψ - α*(γi[2,1]*ψr + γi[2,2]*ψθ))
    
end

@inline function u(U::StateVector) # Scalar gradient-flux

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

@inline function vr(U::StateVector) # r component of the divergence-flux

    # Give names to stored arrays from the state vector
    g  = U.g 
    dr = U.dr 
    dθ = U.dθ
    P  = U.P 

    _,_,_,γs... = g.data

    γ = TwoTensor(γs)

    γi = inv(γ)

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]

    return rootγ(U)*(βr*P - α*(γi[1,1]*dr + γi[1,2]*dθ))
    
end

@inline function vθ(U::StateVector) # θ component of the divergence-flux

    # Give names to stored arrays from the state vector
    g  = U.g 
    dr = U.dr
    dθ = U.dθ  
    P  = U.P 

    _,_,_,γs... = g.data

    γ = TwoTensor(γs)

    γi = inv(γ)

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βθ = -gi[1,3]/gi[1,1]

    return rootγ(U)*(βθ*P - α*(γi[2,1]*dr + γi[2,2]*dθ))
    
end


@inline function rootγ(U::StateVector)

    # Give names to stored arrays from the state vector
    g  = U.g 

    # Unpack the metric into indiviual components
    _,_,_,γs... = g.data

    γ = TwoTensor(γs)

    detγ = det(γ)

    if detγ < 0
        println(x," ",y," ",γ[1,1]," ",γ[2,2]," ",γ[3,3]," ")
    end

    return sqrt(detγ)
end

function divergent_terms(Um,rm,θm,x,y)

    Type = Data.Number

    U = Um[x,y]

    r = rm[x,y]
    θ = θm[x,y]

    # Give names to stored arrays from the state vector
    ψv = U.ψ
    g  = U.g 
    dr = U.dr   
    dθ = U.dθ  
    P  = U.P 

    ψ,ψr,ψθ,Ψ = ψv.data

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    ∂tψ  = βr*ψr + βθ*ψθ - α*Ψ

    ∂ψ   = @Vec [∂tψ,ψr,ψθ]

    ∂f = @Vec [0.,1/r/sin(θ),0.]

    #∂f = @Vec [0.,1.,0.]

    # Calculate time derivative of the metric
    ∂tg = βr*dr + βθ*dθ - α*P

    ∂g = Symmetric3rdOrderTensor{Type}(
        (σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dr[μ,ν] : σ==3 ? dθ[μ,ν] : @assert false)
        )

    Γ = Symmetric3rdOrderTensor{Type}(
        (σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν])
        )   
        
    S = symmetric(@einsum (μ,ν) -> 4*∂ψ[μ]*∂f[ν] - gi[ρ,σ]*Γ[ρ,μ,ν]*∂f[σ]) # - gi[ρ,σ]*H[ρ]*∂f[σ]*g[μ,ν])

    #S = Dρ2T(u,U,r,θ,ns,_ds,x,y)

    return StateTensor{Type}((S[1,1],0.,S[1,3],S[2,2],0.,S[3,3]))

end

function constraints(U::StateVector{Type}) where Type

    # Give names to stored arrays from the state vector
    g  = U.g 
    dr = U.dr   
    dθ = U.dθ  
    P  = U.P 

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    # Calculate time derivative of the metric
    ∂tg = βr*dr + βθ*dθ - α*P

    ∂g = Symmetric3rdOrderTensor{Type}(
        (σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dr[μ,ν] : σ==3 ? dθ[μ,ν] : @assert false)
        )

    Γ  = Symmetric3rdOrderTensor{Type}(
        (σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν])
        )    

    C_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ]

    return C_

end

function integrate_constraints(U,H,ns,_ds)

    int = 0
    #Type = Data.Number

    for x in ns[1], y in ns[2]

        Uxy = U[x,y]
        Hxy = H[x,y]

        g  = Uxy.g 

        # Unpack the metric into individual components
        _,_,_,γs... = g.data

        γ = TwoTensor(γs)

        rootγ = sqrt(det(γ))

        γi2 = inv(γ)

        γi = StateTensor((0.,0.,0.,γi2.data...))

        gi = inv(g)

        Cxy = constraints(Uxy) - Hxy

        # Calculate lapse and shift
        α  = 1/sqrt(-gi[1,1])
        βr = -gi[1,2]/gi[1,1]
        βθ = -gi[1,3]/gi[1,1]

        nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 

        n = @Vec [nt,nr,nθ]

        C = @einsum n[μ]*Cxy[μ]

        Cij = @einsum γi[μ,ν]*Cxy[μ]*Cxy[ν]

        int += sqrt(abs((C^2+Cij)*rootγ/_ds[1]/_ds[2]))

    end

    return int

end

@parallel_indices (x,y) function rhs!(Type,U1,U2,U3,H,∂H,rm,θm,t,ns,dt,_ds,iter)

    #Explicit slices from main memory
    if iter == 1
        U = U1
        Uxy = U1[x,y]
    elseif iter == 2
        U = U2
        Uxy = U2[x,y]
    else
        U = U3
        Uxy = U3[x,y]
    end

    Hxy = H[x,y]; ∂Hxy = ∂H[x,y];

    r = rm[x,y]; θ = θm[x,y];

    # Give names to stored arrays from the state vector
    ψv = Uxy.ψ
    g  = Uxy.g 
    dr = Uxy.dr   
    dθ = Uxy.dθ  
    P  = Uxy.P 

    ψ,ψr,ψθ,Ψ = ψv.data

    # Calculate inverse metric components
    gi = inv(g)

    # Calculate lapse and shift
    α  = 1/sqrt(-gi[1,1])
    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    # Time derivatives of the metric
    ∂tg = βr*dr + βθ*dθ - α*P

    # Unpack the metric into individual components
    _,_,_,γs... = g.data

    γ = TwoTensor{Type}(γs)

    detγ = det(γ)

    if detγ < 0
        println(x," ",y," ",g[1,1]," ",g[2,2]," ",g[3,3]," ")
    end

    γi2 = inv(γ)

    γi = StateTensor{Type}((0.,0.,0.,γi2.data...))

    nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 

    n = @Vec [nt,nr,nθ]

    n_ = @Vec [-α,0.0,0.0]

    #Derivatives of the lapse and the shift 

    ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])
    ∂tβ = α*(@einsum γi[α,μ]*n[ν]*∂tg[μ,ν]) # result is a 3-vector

    # Metric derivatives
    ∂g = Symmetric3rdOrderTensor{Type}(
        (σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dr[μ,ν] : σ==3 ? dθ[μ,ν] : @assert false)
        )

    # Chistoffel Symbols (of the first kind, i.e. all covariant indices)
    Γ  = Symmetric3rdOrderTensor{Type}((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))

    C_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ] - Hxy[μ]

    Cr = (Dρ2T(fg,U,r,θ,ns,_ds,x,y) - dr)
    Cθ = (Dz2T(fg,U,r,θ,ns,_ds,x,y) - dθ)

    # C2 = Tensor{Tuple{4,@Symmetry{4,4}},S}((σ,μ,ν) -> (σ==1 ? 0.0 : σ==2 ? Cr[μ,ν] : σ==3 ? Cθ[μ,ν] : σ==4 ? 0. : @assert false))

    # if (x == 100 && y == 3 && iter==4) 
    #     display(C) 
    # end

    #∂∂f = StateTensor{Type}((0.,0.,0.,1/r^2,0.,1/sin(θ)^2))

    #∂f = @Vec [0.,1/r/sin(θ),0.]

    ∂f = @Vec [0.,1/r/sin(θ),0.]

    c1 = (x==1); c2 = (x==ns[1]);

    #δ = one(StateTensor{Type})
    γ0 = 1.
    #γ1 = -1.
    γ2 = 1.

    # Scalar Evolution
    ######################################################################

    ∂tψ  = βr*ψr + βθ*ψθ - α*Ψ

    ∂ψ   = @Vec [∂tψ,ψr,ψθ]

    # if  y == 1

    #     #S = divergent_terms(U[x,y+1],H[x,y+1],rm[x,y+1],θm[x,y+1])

    #     #S = (  8*divergent_terms(U[x,y+1]) - divergent_terms(U[x,y+2]) )*_ds[2]/6

    #     #2*f(U[x,y+1]) - f(U[x,y+2])

    #     S = Dρ2T(divergent_terms,U,r,θ,ns,_ds,x,y)
    #     St = @einsum gi[μ,ν]*S[μ,ν]
        
    # elseif y == ns[2]

    #     #S = divergent_terms(U[x,y-1],H[x,y-1],rm[x,y-1],θm[x,y-1])

    #     #S = -(  8*divergent_terms(U[x,y-1]) - divergent_terms(U[x,y-2]) )*_ds[2]/6

    #     S = Dρ2T(divergent_terms,U,r,θ,ns,_ds,x,y)
    #     St = @einsum gi[μ,ν]*S[μ,ν]

    # else
    #     S = symmetric(@einsum (μ,ν) -> 4*∂ψ[μ]*∂f[ν] - gi[ρ,σ]*Γ[ρ,μ,ν]*∂f[σ] - gi[ρ,σ]*Hxy[ρ]*∂f[σ]*g[μ,ν])

    #     St = @einsum 4*gi[μ,ν]*∂ψ[μ]*∂f[ν]# - 4*gi[μ,ν]*Hxy[μ]*∂f[ν])
    # end

    # if y in 2:ns[2]
    #     S = A2T(divergent_terms,U,rm,θm,ns,x,y)#/r/sin(θ)
    # else
    #     S = A2T(divergent_terms,U,ns,x,y)
    # end

    S = A2T(divergent_terms,U,rm,θm,ns,x,y)
    St = @einsum gi[μ,ν]*S[μ,ν]

    # S = zero(StateTensor{Type})
    # St = 0.

    ∂tψr = Dρ2(ψu,U,r,θ,ns,_ds,x,y)

    ∂tψθ = Dz2(ψu,U,r,θ,ns,_ds,x,y)

    ∂tΨ  = Div(ψvr,ψvθ,U,r,θ,ns,_ds,x,y) + (α/4)*St  #- 16*π*T

    #######################################################################
    # Define Stress energy tensor and trace 
    # T = zero(StateTensor{Type})
    # Tt = 0.

    ∂tP = -2*α*symmetric(@einsum (μ,ν) -> 2*∂ψ[μ]*∂ψ[ν] + S[μ,ν])  # + 8*pi*Tt*g - 16*pi*T 

    ∂tP += 2*α*symmetric(∂Hxy)

    ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*Hxy[ϵ]*∂g[μ,ν,σ])

    ∂tP += 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*∂g[λ,ϵ,μ]*∂g[ρ,σ,ν])

    ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*Γ[μ,ϵ,λ]*Γ[ν,σ,ρ])

    # Constraint damping term for C_
    ∂tP += γ0*α*symmetric(@einsum (μ,ν) -> C_[μ]*n_[ν] + C_[ν]*n_[μ] - g[μ,ν]*n[ϵ]*C_[ϵ])

    ∂tP -= (@einsum 0.5*γi[i,j]*∂tg[i,j])*P

    ∂tP -= γ2*(βr*Cr + βθ*Cθ)

    ###########################################
    # All finite differencing occurs here

    ∂tP += DivT(vr,vθ,U,r,θ,ns,_ds,x,y) # Div(vr::Function,vθ::Function,U,r,θ,ns,_ds,x,y)

    ∂tdr = symmetric(Dρ2T(u,U,r,θ,ns,_ds,x,y)) + α*γ2*Cr 

    ∂tdθ = symmetric(Dz2T(u,U,r,θ,ns,_ds,x,y)) + α*γ2*Cθ

    #########################################
    ∂tP = symmetric(∂tP)

    #Boundary conditions

    if (c1 || c2) #&& y ≠ 1 && y ≠ ns[2] && false

        if c1 p=-1 else p=1 end

        s_ = @Vec [0.0,p*sin(θ),p*cos(θ)]

        snorm = @einsum gi[μ,ν]*s_[μ]*s_[ν]

        if snorm < 0
            println(x," ",y)
        end
    
        s_ = s_/sqrt(snorm) 

        s = @einsum gi[μ,ν]*s_[ν]
    
        rhat_ = @Vec [p*sin(θ),p*cos(θ)]

        rnorm = @einsum γi2[i,j]*rhat_[i]*rhat_[j]

        rhat_ = rhat_/sqrt(rnorm)
    
        rhat = @einsum γi2[i,j]*rhat_[j]
    
        θhat = @Vec [cos(θ),-sin(θ)]

        θnorm = @einsum γ[i,j]*θhat[i]*θhat[j]

        θhat = θhat/sqrt(θnorm)
    
        θhat_ = @einsum γ[i,j]*θhat[j]
    
        cp =  α - βr*rhat_[1] - βθ*rhat_[2]
        cm = -α - βr*rhat_[1] - βθ*rhat_[2]
        c0 =    - βr*rhat_[1] - βθ*rhat_[2]
    
        βdotθ = βr*θhat_[1] + βθ*θhat_[2]
    
        Up = P + rhat[1]*dr + rhat[2]*dθ
        U0 = θhat[1]*dr + θhat[2]*dθ
    
        # Boundary Condition:
        # You get to choose the incoming 
        # characteristic modes (Um)
        # Pick a function Um = f(Up,U0)
    
        # l = @einsum (n[α] + s[α])/sqrt(2)
        # k = @einsum (n[α] - s[α])/sqrt(2)
    
        # l_  = @einsum g[μ,α]*l[α]
        # #Θ_  = @einsum g[μ,α]*Θ[α]
        # #k_ = @einsum g[μ,α]*k[α]
    
        # #σ = StateTensor((μ,ν) -> gi[μ,ν] + k[μ]*l[ν] + l[μ]*k[ν])
    
        # σ_ = StateTensor((μ,ν) -> g[μ,ν] + n_[μ]*n_[ν] - s_[μ]*s_[ν])
    
        # σ = @einsum gi[μ,α]*gi[ν,β]*σ_[α,β]
    
        # σm = @einsum gi[μ,α]*σ_[ν,α] # mixed indices (raised second index)
    
        # #δ4 = one(SymmetricFourthOrderTensor{4})
        # δ = one(SymmetricSecondOrderTensor{3})

        # γp = @einsum δ[μ,ν] + n_[μ]*n[ν] 

        # Q4 = SymmetricFourthOrderTensor{3,Type}(
        #     (μ,ν,α,β) -> σ_[μ,ν]*σ[α,β]/2 - 2*l_[μ]*σm[ν,α]*k[β] + l_[μ]*l_[ν]*k[α]*k[β]
        # ) # Four index constraint projector (indices down down up up)
    
        # Q3 = Symmetric3rdOrderTensor{Type}(
        #     (μ,ν,α) -> l_[μ]*σm[ν,α] - σ_[μ,ν]*l[α]/2 - l_[μ]*l_[ν]*k[α]/2
        # ) # Three index constraint projector (indices down down up)

        #Pij = @einsum δ3[i,j] - rhat[i]*r_hat[j]

        # O = SymmetricFourthOrderTensor{4}(
        #     (μ,ν,α,β) -> σm[μ,α]*σm[ν,β] - σ_[μ,ν]*σ[α,β]/2
        # ) # Gravitational wave projector
    
        # Pl = Tensor{Tuple{@Symmetry{4,4},4}}((μ,ν,α) -> l[μ]*δ[ν,α] - l_[α]*gi[μ,ν]/2)
    
        # Pθ = Tensor{Tuple{@Symmetry{4,4},4}}((μ,ν,α) -> Θ[μ]*δ[ν,α] - Θ_[α]*gi[μ,ν]/2)
    
        #Um1 = @einsum (sqrt(2)/2)*Pl[μ,ν,α]*Up[μ,ν] + Pθ[μ,ν,α]*U0[μ,ν] - Hxy[α]
    
        # Condition ∂tgμν = 0 on the boundary
        Umb = (cp/cm)*Up - 2*(βdotθ/cm)*U0

        #Umb = -Up
    
        #Um2 = P - rhat[1]*dx - rhat[3]*dz
        #-sqrt(2)*Q3[μ,ν,α]*Um1[α]
    
        #Um = @einsum -sqrt(2)*Q3[μ,ν,α]*Um1[α]# + δ4[μ,ν,α,β]*Um2[α,β] - Q4[μ,ν,α,β]*Um2[α,β]
        #Um = Um2

        #SAT type boundary conditions

        ε = 2*abs(cm)*_ds[1]
    
        Pb  = 0.5*(Up + Umb)
        drb = 0.5*(Up - Umb)*rhat_[1] + U0*θhat_[1] 
        dθb = 0.5*(Up - Umb)*rhat_[2] + U0*θhat_[2] 
    
        ∂tP  += ε*(Pb - P)
        ∂tdr += ε*(drb - dr)
        ∂tdθ += ε*(dθb - dθ)

        # Boundary conditions for Killing Scalar

        Uψp = Ψ + rhat[1]*ψr + rhat[2]*ψθ
        Uψ0 =     θhat[1]*ψr + θhat[2]*ψθ


        if c1 
            # condition ∂tψ = 0
            Uψmb = (cp/cm)*Uψp - 2*(βdotθ/cm)*Uψ0
        else
            Amp = 0.01
            σ = 0.5
            μ0 = 12.0

            f(t,ρ,z) = (μ0-t-σ)<z<(μ0-t+σ) ? (Amp/σ^8)*(z-((μ0-t)-σ))^4*(z-((μ0-t)+σ))^4 : 0.
            #f(t,ρ,z) = (μ0-t-σ)<ρ<(μ0-t+σ) ? (Amp/σ^8)*(ρ-((μ0-t)-σ))^4*(ρ-((μ0-t)+σ))^4 : 0.

            ∂tf(t,ρ,z) = ForwardDiff.derivative(t -> f(t,ρ,z),t)
            ∂ρf(t,ρ,z) = ForwardDiff.derivative(ρ -> f(t,ρ,z),ρ)
            ∂zf(t,ρ,z) = ForwardDiff.derivative(z -> f(t,ρ,z),z)

            Ψf(t,ρ,z) = -(∂tf(t,ρ,z) - βr*∂ρf(t,ρ,z) - βθ*∂zf(t,ρ,z))/α

            Uψmbf(t,ρ,z) = Ψf(t,ρ,z) - rhat[1]*∂ρf(t,ρ,z) - rhat[2]*∂zf(t,ρ,z)

            Uψmb = Uψmbf(t,r*sin(θ),r*cos(θ))
        end

        Ψb  = 0.5*(Uψp + Uψmb)
        ψrb = 0.5*(Uψp - Uψmb)*rhat_[1] + Uψ0*θhat_[1] 
        ψθb = 0.5*(Uψp - Uψmb)*rhat_[2] + Uψ0*θhat_[2] 
    
        ∂tΨ  += ε*(Ψb - Ψ)
        ∂tψr += ε*(ψrb - ψr)
        ∂tψθ += ε*(ψθb - ψθ)

        # ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])
    
        # ∂tβ = α*(@einsum γi[α,μ]*n[ν]*∂tg[μ,ν]) # result is a 3-vector
    
        # ∂t∂tg = (βr*∂tdr + βθ*∂tdθ - α*∂tP) + (∂tβ[2]*dr + ∂tβ[3]*dθ - ∂tα*P)
    
        # ∂t∂g = Symmetric3rdOrderTensor{Type}((σ,μ,ν) -> (σ==1 ? ∂t∂tg[μ,ν] : σ==2 ? ∂tdr[μ,ν] : σ==3 ? ∂tdθ[μ,ν] : @assert false))
    
        # ∂tΓ  = Symmetric3rdOrderTensor{Type}((σ,μ,ν) -> 0.5*(∂t∂g[ν,μ,σ] + ∂t∂g[μ,ν,σ] - ∂t∂g[σ,μ,ν]))   

        # ∂tH = Vec{3}((∂Hxy[1,:]...))
        # ∂rH = Vec{3}((∂Hxy[2,:]...))
        # ∂θH = Vec{3}((∂Hxy[3,:]...))
    
        # ∂tC = (@einsum gi[ϵ,σ]*∂tΓ[λ,ϵ,σ]) - (@einsum gi[μ,ϵ]*gi[ν,σ]*Γ[λ,μ,ν]*∂tg[ϵ,σ]) - ∂tH
    
        # # set up finite differencing for the constraints, by defining a function
        # # that calculates the constraints for any x and y index. This
        # # might not be the best idea, but should work.

        # ∂rC = DρC(constraints,U,r,θ,ns,_ds,x,y)*_ds[1] - ∂rH # + 0.5*γ2*(@einsum (n_[σ]*gi[μ,ν]*Cr[μ,ν] - n[ν]*Cr[σ,ν]))
        # ∂θC = DzC(constraints,U,r,θ,ns,_ds,x,y)*_ds[2] - ∂θH # + 0.5*γ2*(@einsum (n_[σ]*gi[μ,ν]*Cθ[μ,ν] - n[ν]*Cθ[σ,ν]))
    
        # F = (∂tC - βr*∂rC - βθ*∂θC)/α # + γ2*(@einsum γi[μ,ν]*C2[μ,ν,λ] - 0.5*γp[λ,σ]*gi[μ,ν]*C2[σ,μ,ν])

        # ∂Cm = F + rhat[1]*∂rC + rhat[2]*∂θC

        # #c4rθ = Dr2(fdr,U,r,θ,ns,_ds,x,y) - Dθ2(fdθ,U,r,θ,ns,_ds,x,y)
        # #c4θr = -c4rθ

        # ∂tUp = ∂tP + rhat[1]*∂tdr + rhat[2]*∂tdθ# - γ2*∂tg   
        # ∂tUm = ∂tP - rhat[1]*∂tdr - rhat[2]*∂tdθ# - γ2*∂tg
        # ∂tU0 = θhat[1]*∂tdr + θhat[2]*∂tdθ

        # #∂tU0 = ()∂tdx + ∂tdz

        # ∂tUmb = @einsum Q4[μ,ν,α,β]*∂tUm[α,β]
        # ∂tUmb -= sqrt(2)*cm*(@einsum Q3[α,μ,ν]*∂Cm[α]) # Constraint preserving BCs

        # ∂tU0b = ∂tU0# + c0*(rhat[1]*θhat[2]*c4θr + rhat[2]*θhat[1]*c4rθ)

        # #∂tUmb = @einsum O[μ,ν,α,β]*∂th[α,β] # Incoming Gravitational waveform

        # # Time derivatives are OVERWRITTEN here, but still depends on evolution values
        # ∂tP  = 0.5*(∂tUp + ∂tUmb)
        # ∂tdr = 0.5*(∂tUp - ∂tUmb)*rhat_[1] + ∂tU0b*θhat_[1] 
        # ∂tdθ = 0.5*(∂tUp - ∂tUmb)*rhat_[2] + ∂tU0b*θhat_[2] 

    end

    #Axis regularity, acts as a generic boundary
    if (y==1 || y==ns[2]) && false

        if y==1 p=-1 else p=1 end
    
        rhat_ = @Vec [0.,-p]

        rnorm = @einsum γi2[i,j]*rhat_[i]*rhat_[j]

        rhat_ = rhat_/sqrt(rnorm)
    
        rhat = @einsum γi2[i,j]*rhat_[j]
    
        # It is now θhat that points OUT of the domain
        θhat = @Vec [-1.,0.]

        θnorm = @einsum γ[i,j]*θhat[i]*θhat[j]

        θhat = θhat/sqrt(θnorm)
    
        θhat_ = @einsum γ[i,j]*θhat[j]
    
        cp =  α - βr*θhat_[1] - βθ*θhat_[2]
        cm = -α - βr*θhat_[1] - βθ*θhat_[2]
        c0 =    - βr*θhat_[1] - βθ*θhat_[2]
    
        βdotr = βr*rhat_[1] + βθ*rhat_[2]
    
        Up = P + θhat[1]*dr + θhat[2]*dθ
        U0 = rhat[1]*dr + rhat[2]*dθ

        Dt = StateTensor{Type}((0.,1.,0.,1.,1.,0.))
        Dρ = StateTensor{Type}((1.,0.,1.,0.,0.,1.))

        # Condition ∂tgμν = 0 on the boundary
        Umbt = (cp/cm)*Up - 2*(βdotr/cm)*U0

        # Condition ∂ρgμν = 0 on the boundary
        Umbρ = Up

        Umb = Dt.*Umbt + Dρ.*Umbρ

        #SAT type boundary conditions

        ε = 2*abs(cm)*_ds[2]
    
        Pb  = 0.5*(Up + Umb)
        drb = 0.5*(Up - Umb)*θhat_[1] + U0*rhat_[1] 
        dθb = 0.5*(Up - Umb)*θhat_[2] + U0*rhat_[2] 
    
        ∂tP  += ε*(Pb - P)
        ∂tdr += ε*(drb - dr)
        ∂tdθ += ε*(dθb - dθ)
        
    end

    ∂tψv = @Vec [∂tψ,∂tψr,∂tψθ,∂tΨ] 

    ∂tU = StateVector{Type}(∂tψv,∂tg,∂tdr,∂tdθ,∂tP)

    #∂tU += Dissipation(U,x,y,ns)

    if iter == 1
        U2[x,y] = Uxy + dt*∂tU
    elseif iter == 2
        U3[x,y] = (3/4)*U1[x,y] + (1/4)*Uxy + (1/4)*dt*∂tU
    elseif iter == 3
        U1[x,y] = (1/3)*U1[x,y] + (2/3)*Uxy + (2/3)*dt*∂tU
    end

    return
    
end

function RK4!(S,A,B,C,H,∂H,r,θ,t,ns,dt,_ds)

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

    @parallel bulk rhs!(S,A,B,C,H,∂H,r,θ,t,ns,dt,_ds,1) 

    # Second stage (iter=2)

    @parallel bulk rhs!(S,A,B,C,H,∂H,r,θ,t,ns,dt,_ds,2) 

    # Third stage (iter=3)

    @parallel bulk rhs!(S,A,B,C,H,∂H,r,θ,t,ns,dt,_ds,3) 

    # Main bottleneck is update!(...)
    # perhaps because it has more main memory accesses
    # Is there any way we can improve its performance?

    return

end

@inline function P_init(g_init::Function,∂g_init::Function,r,θ,x,y)

    g   = StateTensor((μ,ν)->  g_init(r[x,y],θ[x,y]  ,μ,ν))
    ∂tg = StateTensor((μ,ν)-> ∂g_init(r[x,y],θ[x,y],1,μ,ν))
    ∂rg = StateTensor((μ,ν)-> ∂g_init(r[x,y],θ[x,y],2,μ,ν))
    ∂θg = StateTensor((μ,ν)-> ∂g_init(r[x,y],θ[x,y],3,μ,ν))

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    return -(∂tg - βr*∂rg - βθ*∂θg)/α
    
end

@inline function ψ_init(fψ::Function,g_init::Function,r,θ)

    ψ    =   fψ(r,θ)
    ∂tψ  = 0.#f∂tψ(r[x,y],θ[x,y])

    g   = StateTensor((μ,ν)->  g_init(r,θ,μ,ν))
    
    ∂rψ = ForwardDiff.derivative(rv -> fψ(rv,θ), r)
    ∂θψ = ForwardDiff.derivative(θv -> fψ(r,θv), θ)

    ∂ρψ = ∂rψ*sin(θ) + ∂θψ*cos(θ)/r
    ∂zψ = ∂rψ*cos(θ) - ∂θψ*sin(θ)/r

    gi = inv(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    Ψ = -(∂tψ - βr*∂ρψ - βθ*∂zψ)/α

    return StateScalar((ψ,∂ρψ,∂zψ,Ψ))
    
end

function sample!(f, ψ, g, ∂g, ns, r, θ, T)

    for x in 1:ns[1], y in 1:ns[2]
        f[x,y] = StateVector{T}(
        ψ_init(ψ,g,r[x,y],θ[x,y]),
        StateTensor{T}((μ,ν) ->  g(r[x,y],θ[x,y]  ,μ,ν)),
        StateTensor{T}((μ,ν) -> ∂g(r[x,y],θ[x,y],2,μ,ν)),
        StateTensor{T}((μ,ν) -> ∂g(r[x,y],θ[x,y],3,μ,ν)),
        P_init(g,∂g,r,θ,x,y)
        )
    end

end

##################################################
function main()
    # Physics


    T = Data.Number

    #numvar=4*7

    # domains
    rmin, rmax = 5.0, 10.0
    θmin, θmax = 0.0, pi
    tmin, tmax = 0.0, 50.

    t         = tmin         # physical time
    # Numerics
    #scale = 20 # normal amount to test with
    scale = 1

    nr, nθ    = scale*100, scale*100
    #nr, nθ    = 32*scale-1, 32*scale  # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf

    ns = (nr,nθ)

    # Derived numerics
    dr = (rmax-rmin)/(nr-1)
    #dθ = pi/(nθ) # cell size for straddled axis, no cells at θ = 0,π
    dθ = (θmax-θmin)/(nθ-1) # cell size for included axis
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
    θ .= Data.Array([θmin + dθ*(j-1) for i in 1:nr, j in 1:nθ]) # include cells on the axis

    #θ .= Data.Array([dθ/2 + dθ*(j-1) for i in 1:nr, j in 1:nθ])

    A  = StructArray{StateVector{T}}(undef,nr,nθ)
    B  = StructArray{StateVector{T}}(undef,nr,nθ)
    C  = StructArray{StateVector{T}}(undef,nr,nθ)

    U_init  = StructArray{StateVector{T}}(undef,nr,nθ)

    H  = StructArray{Tensor{Tuple{3}, T, 1, 3}}(undef,nr,nθ)
    ∂H = StructArray{ThreeTensor{T}}(undef,nr,nθ)

    # Define initial conditions

    M = 1.
    sign = 1.
     
    # @inline g_init(r,θ,μ,ν) =  (r^2*sin(θ)^2)*(( -(1 - 2*M/r) , 2*M/r  , 0.  ),
    #                                            (  sign*2*M/r  , (1 + 2*M/r) , 0.  ),
    #                                            (      0.      ,      0.     , r^2 ))[μ][ν]

    @inline g_init(r,θ,μ,ν) =  (( -(1 - 2*M/r)  ,     2*M*sin(θ)/r    ,    2*M*cos(θ)/r     ),
                                (  2*M*sin(θ)/r , 1 + M*(1-cos(2θ))/r ,     M*sin(2θ)/r     ),
                                (  2*M*cos(θ)/r ,      M*sin(2θ)/r    , 1 + M*(1+cos(2θ))/r ))[μ][ν]

    @inline ∂tg_init(r,θ,μ,ν) =  ((  0. ,  0.  ,  0.  ),
                                  (  0. ,  0.  ,  0.  ),
                                  (  0. ,  0.  ,  0.  ))[μ][ν]

    @inline ∂rg(r,θ,μ,ν) = ForwardDiff.derivative(r -> g_init(r,θ,μ,ν), r)
    @inline ∂θg(r,θ,μ,ν) = ForwardDiff.derivative(θ -> g_init(r,θ,μ,ν), θ)

    @inline ∂ρg(r,θ,μ,ν) = ∂rg(r,θ,μ,ν)*sin(θ) + ∂θg(r,θ,μ,ν)*cos(θ)/r
    @inline ∂zg(r,θ,μ,ν) = ∂rg(r,θ,μ,ν)*cos(θ) - ∂θg(r,θ,μ,ν)*sin(θ)/r
    
    #@inline ∂g_init(r,θ,σ,μ,ν) = (∂tg_init(r,θ,μ,ν),∂rg(r,θ,μ,ν),∂θg(r,θ,μ,ν))[σ]

    @inline ∂g_init(r,θ,σ,μ,ν) = (∂tg_init(r,θ,μ,ν),∂ρg(r,θ,μ,ν),∂zg(r,θ,μ,ν))[σ]

    #@inline fH_(r,θ,μ) = (-2*M/r^2,-2*(M+r)/r^2,-cos(θ)/sin(θ))[μ] # lower index

    @inline fH_(r,θ,μ) = (0.,0.,0.)[μ] # lower index

    # @inline f∂H_(r,θ,μ,ν) = ((0.,0.,0.)[ν],
    #                         ForwardDiff.derivative(r -> fH_(r,θ,ν), r),
    #                         ForwardDiff.derivative(θ -> fH_(r,θ,ν), θ))[μ]
    
    @inline f∂H_(r,θ,μ,ν) = ((0.,0.,0.)[ν],
                             (0.,0.,0.)[ν],
                             (0.,0.,0.)[ν])[μ]

    @inline ψ_init(r,θ) =  0.

    sample!(A, ψ_init, g_init, ∂g_init, ns, r, θ, T)

    B .= A
    C .= A
    U_init .= A

    for i in 1:ns[1], j in 1:ns[2]
        H[i,j]  = @Vec [fH_(r[i,j],θ[i,j],μ) for μ in 1:3]
        ∂H[i,j] = ThreeTensor{T}((μ,ν) -> f∂H_(r[i,j],θ[i,j],μ,ν))
    end

    # x=75;y=75;

    # display(A[x,y].ψ)
    # display(A[x,y].g)
    # display(A[x,y].dr)
    # display(A[x,y].dθ)
    # display(A[x,y].P)

    #return 

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

    nt = 2500
    nout = 10 #round(nt/100)          # plotting frequency

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

    path = string("2dData")
    old_files = readdir(path; join=true)
    for i in 1:length(old_files) 
        rm(old_files[i]) 
    end
    datafile = h5open(path*"/data.h5","cw")
    nsave = Int64(nt/nout) + 1
    ψdata = create_dataset(datafile, "psi", datatype(Data.Number), dataspace(nsave,nr,nθ), chunk=(1,nr,nθ))

    coordsfile = h5open(path*"/coords.h5","cw")
    coordsfile["r"] = Array(r)
    coordsfile["theta"]  = Array(θ)
    close(coordsfile)

    ints = zeros(0)

    iter = 1

    try

    for i in 1:nt


        if mod(i,nout)==0 || i == 1
            res=1;

            # println("")
            # #display(A[xout,yout].P -temp)
            # display(Dθ2(u,A,ns,xout,yout))
            # #println(A[xout,yout].P -temp)
            # println("")
            
            # 2D slice
            #data = [constraints(A[x,y])[1] - H[x,y][1] for x in 1:nr, y in 1:nθ]
            data = [A[x,y].ψ[1] for x in 1:nr, y in 1:nθ]

            # r slice
            #data = [constraints(A[x,100])- H[x,100] for x in 1:nr]
            #data = [A[x,50].g[1,1] - U_init[x,50].g[1,1] for x in 1:nr]
            #data = [A[x,50].ψ[1] - U_init[x,50].ψ[1] for x in 1:nr]

            # θ slice
            #data = [A[50,y].g[1,1] - U_init[50,y].g[1,1] for y in 1:nθ]
            #data = [constraints(A[50,y]) - H[50,y] for y in 1:nθ]
            #data = [divergent_terms(A[1,y])[1,1]/(r[1,y]*sin(θ[1,y])) for y in 1:nθ]

            #return typeof(Array(data))
            labels = ["Ct" "Cr" "Cθ"]

            #data = getindex.(A[:,50].g,μi,νi) - temp_array[:,50]

            #reduce(hcat,data)'

            # plot(Array(θ[10,:]), reduce(hcat,data)', label=labels, title = "Time = "*string(round(t; digits=2)) )
            # ylims!(-10^-2, 10^-2)
            # frame(anim)

            # plot(Array(θ[10,:]), data, label=labels, title = "Time = "*string(round(t; digits=2)) )
            # ylims!(-10^-2, 10^-2)
            # frame(anim)

            #println(t)

            display(A[1,1].g - U_init[1,1].g)
            #display(A[100,1].P)

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
            heatmap(Array(r[1:res:end,1]), Array(θ[1,1:res:end]), Array(data)', title = "Time = "*string(round(t; digits=2)),
            aspect_ratio=1, xlims=(rmin,rmax), ylims=(0,pi),clim=(-2*10^(-2),2*10^(-2)), c=:viridis); 
            frame(anim)

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

            ψdata[iter,:,:] = [A[x,y].ψ[1] for x in 1:nr, y in 1:nθ]
            
            iter += 1

        end

        append!(ints,integrate_constraints(A,H,ns,_ds))

        RK4!(T,A,B,C,H,∂H,r,θ,t,ns,dt,_ds)

        t += dt

    end

    catch error
        close(datafile)
        throw(error)
    end

    close(datafile)

    plot(ints, yaxis=:log)
    ylims!(10^-10, 10^-1)
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
        
    end
    #return
    try
    # Time loop
    iter=2
    global wtime0 = Base.time()
    res=1

    for it = 1:nt


        if mod(it,nout)==0 || it == 1
            res=1;

            plot(Array(θ[2,1:res:end]), Array(∂ₜU.x[var][2,1:res:end])); 
            ylims!(-0.1, 0.1)
            frame(anim)
            println(mean(∂ₜU.x[var][2,1:res:end]))


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