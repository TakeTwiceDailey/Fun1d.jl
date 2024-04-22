module GR_Axial_parallel

const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
const VISUAL = true
using ParallelStencil
using BenchmarkTools
using Plots
#using PyPlot
#using GR
using RecursiveArrayTools
using TensorOperations
using StaticArrays
using InteractiveUtils
using Traceur
using FFTW

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

# Scaled real numbers
# Allows for the multiplication of factors of ρ
# during the evolution, without actually multiplying
# so that divisions can happen later-on even at ρ = 0

struct SReal{N,T<:Real} <: Real
    ρ::T
    x::T
    function SReal{N,T}(ρ::T, x::T) where {N,T}
        @assert N >= 0
        @assert ρ >= 0.
        new{N,T}(ρ,x)
    end
end

# Base.zero
# Base.one 

@inline SReal{N}(ρ::T, x::T) where {N,T} = SReal{N,T}(ρ,x)

@inline Base.convert(::Type{T}, A::SReal{N,T}) where {N,T} = A.ρ^N*A.x

#@inline Base.convert(::Type{SReal{N,T}}, A::T, ρ::T) where {N,T<:Real} = SReal{N,T}(ρ,x)

@inline Base.zero(A::SReal{N,T}) where {N,T} = SReal{N,T}(0.,0.)

function Base.show(io::IO, ::MIME"text/plain", A::SReal{N,T}) where {N,T}
    print(io, [N,A.x])
end

@inline function Base.:+(A::SReal{M,T}, B::SReal{N,T}) where {M,N,T} 
    @assert A.ρ == B.ρ
    if N==M
        SReal{M,T}(A.ρ, A.x + B.x) 
    elseif N>M
        SReal{M,T}(A.ρ, A.x + B.x*(B.ρ)^(N-M)) 
    else
        SReal{N,T}(A.ρ, A.x*(A.ρ)^(M-N) + B.x)
    end
end

@inline Base.:-(A::SReal{N,T}) where {N,T} = SReal{N,T}(A.ρ,-A.x)

@inline Base.:-(A::SReal{M,T}, B::SReal{N,T}) where {M,N,T} = A + (-B)

@inline function Base.:*(A::SReal{N,T}, B::SReal{M,T}) where {N,M,T} 
    @assert A.ρ == B.ρ 
    SReal{N+M,T}(A.ρ, A.x * B.x)
end

@inline function Base.:/(A::SReal{N,T}, B::SReal{M,T}) where {N,M,T} 
    @assert A.ρ == B.ρ 
    SReal{N-M,T}(A.ρ, A.x / B.x)
end

@inline Base.:*(a::T, A::SReal{N,T}) where {N,T} = SReal{N,T}(A.ρ, a*A.x )

@inline Base.:*(A::SReal{N,T}, a::T) where {N,T} = a*A

@inline Base.:/(A::SReal{N,T}, y::T) where {N,T} = SReal{N,T}(A.x / y)

@inline Base.:*(a::Int, A::SReal{N,T}) where {N,T} = SReal{N,T}(A.ρ, a*A.x )

@inline Base.:*(A::SReal{N,T}, a::Int) where {N,T} = a*A

@inline Base.:/(A::SReal{N,T}, y::Int) where {N,T} = SReal{N,T}(A.x / y)

@inline Base.sqrt(A::SReal{0,T}) where T = SReal{0,T}(A.ρ,sqrt(A.x))

struct EvenTensor{T}
    tt::SReal{0,T}
    tρ::SReal{1,T}
    tz::SReal{0,T}
    ρρ::SReal{0,T}
    ρz::SReal{1,T}
    zz::SReal{0,T}
end

struct OddTensor{T}
    tt::SReal{1,T}
    tρ::SReal{0,T}
    tz::SReal{1,T}
    ρρ::SReal{1,T}
    ρz::SReal{0,T}
    zz::SReal{1,T}
end

const StateTensor{T} = Union{EvenTensor{T},OddTensor{T}}

@inline function Base.getindex(A::StateTensor{T},i,j) where T
    field = ((:tt,:tρ,:tz),(:tρ,:ρρ,:ρz),(:tz,:ρz,:zz))[i,j]
    return getproperty(A,field)
end


# Alias for SymmetricSecondOrderTensor 2x2
const TwoTensor{T} = SymmetricSecondOrderTensor{2,T,3}

# Alias for non-symmetric 3 tensor
const ThreeTensor{T} = SecondOrderTensor{3,T,9}

# Alias for StateVector of a scalar field
const StateScalar{T} = Vec{4,T}

# Alias for tensor to hold metric derivatives and Christoffel Symbols
# Defined to be symmetric in the last two indices
const Symmetric3rdOrderTensor{T} = Tensor{Tuple{3,@Symmetry{3,3}},T,3,18}

# # Alias for SymmetricSecondOrderTensor 3x3
# const StateTensor{T} = SymmetricSecondOrderTensor{3,T,6}

# # Alias for SymmetricSecondOrderTensor 2x2
# const TwoTensor{T} = SymmetricSecondOrderTensor{2,T,3}

# # Alias for non-symmetric 3 tensor
# const ThreeTensor{T} = SecondOrderTensor{3,T,9}

# # Alias for StateVector of a scalar field
# const StateScalar{T} = Vec{4,T}

# # Alias for tensor to hold metric derivatives and Christoffel Symbols
# # Defined to be symmetric in the last two indices
# const Symmetric3rdOrderTensor{T} = Tensor{Tuple{3,@Symmetry{3,3}},T,3,18}

# @inline Base.convert(::StateTensor{T}, A::StateTensor{SReal{N,T} where N}) where {T<:Real} = StateTensor{T}(convert.(T,A.data)...)

@inline function det(A::StateTensor{SReal{N,T} where N}) where T
    a,b,c,d,e,f = A.data
    return (a*d*f - a*e*e - b*b*f - c*c*d + 2*b*c*e)
end

@inline function adjugate(A::StateTensor{SReal{N,T} where N}) where T
    a,b,c,d,e,f = A.data
    return StateTensor((d*f-e*e, c*e-b*f, b*e-c*d, a*f-c*c, b*c-a*e, a*d-b*b))
end

@inline function inverse(A::StateTensor{SReal{N,T} where N}) where T
    return adjugate(A)/det(A)
end

@inline function det(A::TwoTensor{SReal{N,T} where N}) where T
    a,b,c = A.data
    return (a*c - b*b)
end

@inline function adjugate(A::TwoTensor{SReal{N,T} where N}) where T
    a,b,c = A.data
    return TwoTensor((c,-b,a))
end

@inline function inverse(A::TwoTensor{SReal{N,T} where N}) where T
    return adjugate(A)/det(A)
end

# This struct type is used to package the state vector into 
# spherical components. This is used to store the state vector
# into memory, and does not store ϕ derivatives as those vanish in axisymmetry.
struct StateVector{T}
    ρ::T
    ψ::StateScalar{T}
    g::StateTensor{T}
    dr::StateTensor{T}
    dθ::StateTensor{T}
    P::StateTensor{T}
end

math_operators = [:+,:-,:*,:/]
# Define math operators for StateVector
for op in math_operators
    @eval import Base.$op
    @eval @inline function $op(A::StateVector{T},B::StateVector{T}) where T
        ψ  = @. $op(A.ψ,B.ψ)
        g  = @. $op(A.g,B.g)
        dr = @. $op(A.dr,B.dr)
        dθ = @. $op(A.dθ,B.dθ)
        P  = @. $op(A.P,B.P)
        return StateVector{T}(B.ρ,ψ,g,dr,dθ,P)
        # All operations inherit the ρ value of the second argument
    end

    @eval @inline function $op(a::Number,B::StateVector{T}) where T
        ψ  = $op(a,B.ψ)
        g  = $op(a,B.g)
        dr = $op(a,B.dr)
        dθ = $op(a,B.dθ)
        P  = $op(a,B.P)
        return StateVector{T}(B.ρ,ψ,g,dr,dθ,P)
    end
end

@inline function Base.:-(A::StateVector{T}) where T
    return StateVector{T}(A.ρ,-A.ψ,-A.g,-A.dr,-A.dθ,-A.P)
end

@inline function Base.zero(::Type{StateVector{T}},ρ) where T
    ψ  = zero(StateScalar{T})
    g  = zero(StateTensor{T})
    return StateVector{T}(ρ,ψ,g,g,g,g)
end

@inline function Base.getindex(A::StateVector{T},i) where T
    if i in 1:4
        A.ψ.data[i]
    elseif i in 5:10
        A.g.data[i-4]
    elseif i in 11:16
        A.dr.data[i-10]
    elseif i in 17:22
        A.dθ.data[i-16]
    elseif i in 23:28
        A.P.data[i-22]
    else
        @assert false "Attempt to index outside of 1:28"
    end
end

@inline function changevalue(A::StateVector{T},val,i) where T

    ψv = A.ψ
    g  = A.g
    dρ = A.dr
    dz = A.dθ
    P  = A.P

    if i in 1:4
        vec = [ψv.data...]
        vec[i] = val
        ψv = StateScalar{T}((vec...,))
    elseif i in 5:10
        vec = [g.data...]
        vec[i-4] = val
        g = StateTensor{T}((vec...,))
    elseif i in 11:16
        vec = [dρ.data...]
        vec[i-10] = val
        dρ = StateTensor{T}((vec...,))
    elseif i in 17:22
        vec = [dz.data...]
        vec[i-16] = val
        dz = StateTensor{T}((vec...,))
    elseif i in 23:28
        vec = [P.data...]
        vec[i-22] = val
        P = StateTensor{T}((vec...,))
    else
        @assert false "Attempt to index outside of 1:28"
    end
    
    return StateVector{T}(A.ρ,ψv,g,dρ,dz,P)
end

# Define functions to return Stuct components for finite 
# differencing in constraint calculations
@inline fψ(U::StateVector)  = unpack(U).ψ[1]
@inline fg(U::StateVector)  = unpack(U).g
@inline fdr(U::StateVector) = unpack(U).dr
@inline fdθ(U::StateVector) = unpack(U).dθ
@inline fP(U::StateVector)  = unpack(U).P

const parity  = StateTensor{Data.Number}((1,-1,1,1,-1,1))
const even_zero = StateTensor{Data.Number}((0,1,0,0,1,0))
const odd_zero  = StateTensor{Data.Number}((1,0,1,1,0,1))
const ψparity = Vec{4}((1,-1,1,1))
const StateVectorParity = StateVector{Data.Number}(0.,ψparity,parity,-parity,parity,parity)
const zero4 = @Vec [0.,0.,0.,0.]
const even_zeroSV = StateVector{Data.Number}(0.,(@Vec [0.,0.,1.,0.]),even_zero,odd_zero,even_zero,even_zero)
const  odd_zeroSV = StateVector{Data.Number}(0.,(@Vec [1.,1.,0.,1.]),odd_zero,even_zero,odd_zero,odd_zero)
const parityC = Vec{3}((1,-1,1))

@inline zeroT(p) = (p==1 ? even_zero : odd_zero)

@inline zeroS(p) = (p==1 ? 0 : 1)

@inline function Dr2ST(U,ns,x,y) 
    n = ns[1]
    f(x) = unpack(x)
    if x in 2:n-1
        f(U[x+1,y])-2*f(U[x,y])+f(U[x-1,y])
    elseif x==1
        #zeroST
        #-2*f(U[1,y]) + 2*f(U[2,y])
        U[1,y] - 2*U[2,y] + U[3,y]
    elseif x==n 
        #zeroST
        #2*f(U[n-1,y]) - 2*f(U[n,y])
        U[n-2,y] - 2*U[n-1,y] + U[n,y]
    end
end

# @inline function DrST(U,ns,x,y) 
#     n = ns[1]
#     f(x) = unpack(x)
#     if x in 2:n-1
#         0.5*(f(U[x+1,y])-f(U[x-1,y]))
#     elseif x==1
#          f(U[2,y]) - f(U[1,y])
#     elseif x==n 
#         f(U[n-1,y]) - f(U[n,y])
#     end
# end

@inline function Dθ2ST(U,ns,x,y)
    n = ns[2]
    f(x) = unpack(x)
    if y in 2:n-1
        f(U[x,y+1]) - 2*f(U[x,y]) + f(U[x,y-1])
    elseif y==1
        f(U[x,y+1]) - 2*f(U[x,y]) + StateVectorParity*f(U[x,y+1])
    elseif y==n
        f(U[x,n-1]) - 2*f(U[x,n]) + StateVectorParity*f(U[x,n-1])
    end
end

const a11 = -2.8235294117647058823529411764705882352941176470588
const a21 = 5.6470588235294117647058823529411764705882352941176
const a31 = -2.8235294117647058823529411764705882352941176470588
const a41 = 0.0
const a51 = 0.0
const a61 = 0.0
const a12 = 1.6271186440677966101694915254237288135593220338983
const a22 = -4.0677966101694915254237288135593220338983050847458
const a32 = 3.2542372881355932203389830508474576271186440677966
const a42 = -0.81355932203389830508474576271186440677966101694915
const a52 = 0.0
const a62 = 0.0
const a13 = -1.1162790697674418604651162790697674418604651162791
const a23 = 4.4651162790697674418604651162790697674418604651163
const a33 = -6.6976744186046511627906976744186046511627906976744
const a43 = 4.4651162790697674418604651162790697674418604651163
const a53 = -1.1162790697674418604651162790697674418604651162791
const a63 = 0.0
const a14 = 0.0
const a24 = -0.97959183673469387755102040816326530612244897959184
const a34 = 3.9183673469387755102040816326530612244897959183673
const a44 = -5.8775510204081632653061224489795918367346938775510
const a54 = 3.9183673469387755102040816326530612244897959183673
const a64 = -0.97959183673469387755102040816326530612244897959184

@inline function Dr4ST(U,ns,x,y) 
    n = ns[1]
    f(x) = unpack(x)
    if x in 5:n-4
        -f(U[x-2,y]) + 4*f(U[x-1,y]) - 6*f(U[x,y]) + 4*f(U[x+1,y]) - f(U[x+2,y])
    elseif x==1
        (a11*f(U[1,y]) + a21*f(U[2,y]) + a31*f(U[3,y]))
    elseif x==2
        (a12*f(U[1,y]) + a22*f(U[2,y]) + a32*f(U[3,y]) + a42*f(U[4,y]))
    elseif x==3
        (a13*f(U[1,y]) + a23*f(U[2,y]) + a33*f(U[3,y]) + a43*f(U[4,y]) + a53*f(U[5,y]))
    elseif x==4
        (a24*f(U[2,y]) + a34*f(U[3,y]) + a44*f(U[4,y]) + a54*f(U[5,y]) + a64*f(U[6,y]))
    elseif x==n
        (a11*f(U[n,y]) + a21*f(U[n-1,y]) + a31*f(U[n-2,y]))
    elseif x==n-1
        (a12*f(U[n,y]) + a22*f(U[n-1,y]) + a32*f(U[n-2,y]) + a42*f(U[n-3,y]))
    elseif x==n-2
        (a13*f(U[n,y]) + a23*f(U[n-1,y]) + a33*f(U[n-2,y]) + a43*f(U[n-3,y]) + a53*f(U[n-4,y]))
    elseif x==n-3
        (a24*f(U[n-1,y]) + a34*f(U[n-2,y]) + a44*f(U[n-3,y]) + a54*f(U[n-4,y]) + a64*f(U[n-5,y]))
    end
end

@inline function Dθ4ST(U,ns,x,y)
    n = ns[2]
    f(x) = unpack(x)
    if y in 4:n-3
        -f(U[x,y-2]) + 4*f(U[x,y-1]) - 6*f(U[x,y]) + 4*f(U[x,y+1]) + -f(U[x,y+2])
    elseif y==3
        -odd_zeroSV*f(U[x,y-2]) + 4*f(U[x,y-1]) - 6*f(U[x,y]) + 4*f(U[x,y+1]) + -f(U[x,y+2])
    elseif y==2
        -StateVectorParity*f(U[x,2]) + 4*odd_zeroSV*f(U[x,1]) - 6*f(U[x,y]) + 4*f(U[x,y+1]) + -f(U[x,y+2])
    elseif y==1
        odd_zeroSV*(-6*f(U[x,1]) + 8*f(U[x,2]) - 2*f(U[x,3]))
    elseif y==n-2
        -f(U[x,n-4]) + 4*f(U[x,n-3]) - 6*f(U[x,n-2]) + 4*f(U[x,n-1]) - odd_zeroSV*f(U[x,n])
    elseif y==n-1
        -f(U[x,n-3]) + 4*f(U[x,n-2]) - 6*f(U[x,n-1]) + 4*odd_zeroSV*f(U[x,n]) - StateVectorParity*f(U[x,n-1])
    else y==n
        odd_zeroSV*(-6*f(U[x,n]) + 8*f(U[x,n-1]) - 2*f(U[x,n-2]))
    end
end


@inline function Dissipation(U,r,θ,x,y,ns)

    ϵ = 1.

    (ϵ/16)*(Dr4ST(U,ns,x,y)# + (2/r)*DrST(U,ns,x,y)
    + (1/r^2)*Dθ4ST(U,ns,x,y))
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
    elseif y==2
        0.5*(-zeroS(-p)*f(U[x,1]) + f(U[x,3])) # includes the axis
        #0.5*(-p*f(U[x,1]) + f(U[x,2])) # straddles the axis
    elseif y==n-1
        0.5*(-f(U[x,n-2]) + zeroS(-p)*f(U[x,n])) # includes the axis
        #0.5*(-f(U[x,n-1]) + p*f(U[x,n])) # straddles the axis
    elseif y==1
        zeroS(p)*(f(U[x,2])) # includes the axis
        #0.5*(-p*f(U[x,1]) + f(U[x,2])) # straddles the axis
    elseif y==n
        -zeroS(p)*(f(U[x,n-1])) # includes the axis
        #0.5*(-f(U[x,n-1]) + p*f(U[x,n])) # straddles the axis
    end
end

const q11 = -24/17.; const q21 = 59/34. ;
const q31 = -4/17. ; const q41 = -3/34. ;
const q51 = 0.     ; const q61 =  0.    ;

const q12 = -1/2.  ; const q22 = 0.     ;
const q32 = 1/2.   ; const q42 = 0.     ;
const q52 = 0.     ; const q62 = 0.     ;

const q13 =  4/43. ; const q23 = -59/86.;
const q33 =  0.    ; const q43 = 59/86. ;
const q53 =  -4/43.; const q63 = 0.     ;

const q14 = 3/98.  ; const q24 = 0.     ;
const q34 = -59/98.; const q44 = 0.     ;
const q54 = 32/49. ; const q64 = -4/49. ;

@inline function Dr4(f::Function,U,ns,x,y)
    n = ns[1]
    if x in 5:n-4
        (f(U[x-2,y]) - 8*f(U[x-1,y]) + 8*f(U[x+1,y]) - f(U[x+2,y]))/12
    elseif x==1
        (q11*f(U[1,y]) + q21*f(U[2,y]) + q31*f(U[3,y]) + q41*f(U[4,y]))
    elseif x==2
        (q12*f(U[1,y]) + q32*f(U[3,y]))
    elseif x==3
        (q13*f(U[1,y]) + q23*f(U[2,y]) + q43*f(U[4,y]) + q53*f(U[5,y]))
    elseif x==4
        (q14*f(U[1,y]) + q34*f(U[3,y]) + q54*f(U[5,y]) + q64*f(U[6,y]))
    elseif x==n
        -(q11*f(U[n,y]) + q21*f(U[n-1,y]) + q31*f(U[n-2,y]) + q41*f(U[n-3,y]))
    elseif x==n-1
        -(q12*f(U[n,y]) + q32*f(U[n-2,y]))
    elseif x==n-2
        -(q13*f(U[n,y]) + q23*f(U[n-1,y]) + q43*f(U[n-3,y]) + q53*f(U[n-4,y]))
    elseif x==n-3
        -(q14*f(U[n,y]) + q34*f(U[n-2,y]) + q54*f(U[n-4,y]) + q64*f(U[n-5,y]))
    end
    #else @assert false end
end

@inline function Dθ4(f::Function,U,ns,x,y,p)
    n = ns[2]
    if y in 4:n-3
        (f(U[x,y-2]) - 8*f(U[x,y-1]) + 8*f(U[x,y+1]) - f(U[x,y+2]))/12
    elseif y == 3
        (zeroS(-p).*f(U[x,y-2]) - 8*f(U[x,y-1]) + 8*f(U[x,y+1]) - f(U[x,y+2]))/12
    elseif y == n-2
        (f(U[x,y-2]) - 8*f(U[x,y-1]) + 8*f(U[x,y+1]) - zeroS(-p).*f(U[x,y+2]))/12
    elseif y == 2
        (p*f(U[x,2]) - 8*zeroS(-p).*f(U[x,1]) + 8*f(U[x,3]) - f(U[x,4]))/12
    elseif y == n-1
        (f(U[x,n-3]) - 8*f(U[x,n-2]) + 8*zeroS(-p).*f(U[x,n]) - p*f(U[x,n-1]))/12
    elseif y==1
        zeroS(p).*(16*f(U[x,2]) - 2*f(U[x,3]))/12
    elseif y==n
        zeroS(p).*(-16*f(U[x,n-1]) + 2*f(U[x,n-2]))/12
    end
end

@inline function Dρ4(f::Function,U,r,θ,ns,_ds,x,y,p=1) 
    Dr4(f,U,ns,x,y)*_ds[1]*sin(θ) + Dθ4(f,U,ns,x,y,p)*_ds[2]*cos(θ)/r
end

@inline function Dz4(f::Function,U,r,θ,ns,_ds,x,y,p=1)
    Dr4(f,U,ns,x,y)*_ds[1]*cos(θ) - Dθ4(f,U,ns,x,y,p)*_ds[2]*sin(θ)/r
end

@inline function Div4(vr::Function,vθ::Function,U,r,θ,ns,_ds,x,y)
    (Dρ4(vr,U,r,θ,ns,_ds,x,y,-1) + Dz4(vθ,U,r,θ,ns,_ds,x,y,1))/rootγ(U[x,y])
end


@inline function Dρ2(f::Function,U,r,θ,ns,_ds,x,y,p=1) 
    Dr2(f,U,ns,x,y)*_ds[1]*sin(θ) + Dθ2(f,U,ns,x,y,p)*_ds[2]*cos(θ)/r
end

@inline function Dz2(f::Function,U,r,θ,ns,_ds,x,y,p=1)
    Dr2(f,U,ns,x,y)*_ds[1]*cos(θ) - Dθ2(f,U,ns,x,y,p)*_ds[2]*sin(θ)/r
end

@inline function Div(vr::Function,vθ::Function,U,r,θ,ns,_ds,x,y)
    (Dρ2(vr,U,r,θ,ns,_ds,x,y,-1) + Dz2(vθ,U,r,θ,ns,_ds,x,y,1))/rootγ(U[x,y])
end

# @inline function Aθ2T(f::Function,U,r,θ,ns,x,y)
#     n = ns[2]
#     if y in 3:n-2
#         0.5*(f(U,r,θ,x,y-1) + f(U,r,θ,x,y+1))
#     elseif y==2
#         0.5*(f(U,r,θ,x,2) + f(U,r,θ,x,y+1))
#     elseif y==n-1
#         0.5*(f(U,r,θ,x,n-2) + f(U,r,θ,x,n-1))
#     elseif y==1
#         f(U,r,θ,x,2)
#     elseif y==n
#         f(U,r,θ,x,n-1)
#     end
# end

# @inline function Ar2T(f::Function,U,r,θ,ns,x,y)
#     n = ns[1]
#     if y==1 y+=1 end
#     if y==n y-=1 end
#     if x in 2:n-1
#         0.5*(f(U,r,θ,x-1,y) + f(U,r,θ,x+1,y))
#     elseif x==1
#         f(U,r,θ,2,y)
#     elseif x==n 
#         f(U,r,θ,n-1,y)
#     end
# end

@inline function Aθ2T(U,ns,x,y)
    n = ns[2]
    f(x) = unpack(x)
    if y in 2:n-1
        0.5*(f(U[x,y-1]) + f(U[x,y+1]))
    elseif y==1
        0.5*(StateVectorParity*f(U[x,2]) + f(U[x,2]))
    elseif y==n
        0.5*(StateVectorParity*f(U[x,n-1]) + f(U[x,n-1]))
    end
end

@inline function Ar2T(U,ns,x,y)
    n = ns[1]
    f(x) = unpack(x)
    if x in 2:n-1
        0.5*(f(U[x-1,y]) + f(U[x+1,y]))
    elseif x==1
        f(U[x,2])
    elseif x==n 
        f(U[x,n-1])
    end
end

@inline function Aρ(f::Function,U,r,θ,ns,_ds,x,y) 
    #Ar2(f,U,ns,x,y)*_ds[1]*sin(θ) + Aθ2(f,U,ns,x,y)*_ds[2]*cos(θ)/r
    Aθ2(f,U,ns,x,y)
end

@inline function Aθ2(f::Function,U,ns,x,y)
    n = ns[2]
    if y in 2:n-1
        0.5*(f(U[x,y-1]) + f(U[x,y+1]))
    elseif y==1
        0.5*(-ψparity.*f(U[x,2]) + f(U[x,2]))
    elseif y==n
        0.5*(-ψparity.*f(U[x,n-1]) + f(U[x,n-1]))
    end
end

@inline function Ar2(f::Function,U,ns,x,y)
    n = ns[1]
    if x in 2:n-1
        0.5*(f(U[x-1,y]) + f(U[x+1,y]))
    elseif x==1
        f(U[x,1])
    elseif x==n 
        f(U[x,n])
    end
end

@inline function A2T(U,ns,x,y)
    0.5*(Ar2T(U,ns,x,y) + Aθ2T(U,ns,x,y))
    #Aθ2T(U,ns,x,y)
end


@parallel_indices (x,y) function Average!(Type,U1,U2,ns)
    ρ = U1[x,y].ρ
    UA = A2T(U1,ns,x,y)
    UA = StateVector{Type}(ρ,UA.ψ,UA.g,UA.dr,UA.dθ,UA.P)
    U2[x,y] = pack(UA)

    return
end

@inline function Dr4T(f::Function,U,ns,x,y)
    n = ns[1]
    if x in 5:n-4
        (f(U[x-2,y]) - 8*f(U[x-1,y]) + 8*f(U[x+1,y]) - f(U[x+2,y]))/12
    elseif x==1
        (q11*f(U[1,y]) + q21*f(U[2,y]) + q31*f(U[3,y]) + q41*f(U[4,y]))
    elseif x==2
        (q12*f(U[1,y]) + q32*f(U[3,y]))
    elseif x==3
        (q13*f(U[1,y]) + q23*f(U[2,y]) + q43*f(U[4,y]) + q53*f(U[5,y]))
    elseif x==4
        (q14*f(U[1,y]) + q34*f(U[3,y]) + q54*f(U[5,y]) + q64*f(U[6,y]))
    elseif x==n
        -(q11*f(U[n,y]) + q21*f(U[n-1,y]) + q31*f(U[n-2,y]) + q41*f(U[n-3,y]))
    elseif x==n-1
        -(q12*f(U[n,y]) + q32*f(U[n-2,y]))
    elseif x==n-2
        -(q13*f(U[n,y]) + q23*f(U[n-1,y]) + q43*f(U[n-3,y]) + q53*f(U[n-4,y]))
    elseif x==n-3
        -(q14*f(U[n,y]) + q34*f(U[n-2,y]) + q54*f(U[n-4,y]) + q64*f(U[n-5,y]))
    end
    #else @assert false end
end

@inline function Dθ4T(f::Function,U,ns,x,y,p)
    n = ns[2]
    if y in 4:n-3
        (f(U[x,y-2]) - 8*f(U[x,y-1]) + 8*f(U[x,y+1]) - f(U[x,y+2]))/12
    elseif y == 3
        (zeroT(-p).*f(U[x,y-2]) - 8*f(U[x,y-1]) + 8*f(U[x,y+1]) - f(U[x,y+2]))/12
    elseif y == n-2
        (f(U[x,y-2]) - 8*f(U[x,y-1]) + 8*f(U[x,y+1]) - zeroT(-p).*f(U[x,y+2]))/12
    elseif y == 2
        (p*parity.*f(U[x,2]) - 8*zeroT(-p).*f(U[x,1]) + 8*f(U[x,3]) - f(U[x,4]))/12
    elseif y == n-1
        (f(U[x,n-3]) - 8*f(U[x,n-2]) + 8*zeroT(-p).*f(U[x,n]) - p*parity.*f(U[x,n-1]))/12
    elseif y==1
        zeroT(p).*(16*f(U[x,2]) - 2*f(U[x,3]))/12
    elseif y==n
        zeroT(p).*(-16*f(U[x,n-1]) + 2*f(U[x,n-2]))/12
    end
end

@inline function Dρ4T(f::Function,U,r,θ,ns,_ds,x,y,p=1) 
    Dr4T(f,U,ns,x,y)*_ds[1]*sin(θ) + Dθ4T(f,U,ns,x,y,p)*_ds[2]*cos(θ)/r
end

@inline function Dz4T(f::Function,U,r,θ,ns,_ds,x,y,p=1)
    Dr4T(f,U,ns,x,y)*_ds[1]*cos(θ) - Dθ4T(f,U,ns,x,y,p)*_ds[2]*sin(θ)/r
end

@inline function Div4T(vr::Function,vθ::Function,U,r,θ,ns,_ds,x,y)
    (Dρ4T(vr,U,r,θ,ns,_ds,x,y,-1) + Dz4T(vθ,U,r,θ,ns,_ds,x,y))/rootγ(U[x,y])
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
    if y in 3:n-2
        0.5*(-f(U[x,y-1]) + f(U[x,y+1]))
    elseif y == 2
        0.5*(-zeroT(-p).*f(U[x,1]) + f(U[x,3]))
    elseif y == n-1
        0.5*(-f(U[x,n-2]) + zeroT(-p).*f(U[x,n]))
    elseif y==1
        zeroT(p).*(f(U[x,2])) # includes the axis
        #-0.5*p*f(U[x,2]) + 0.5*f(U[x,2]) # includes the axis
        #0.5*(-p*parity.*f(U[x,1]) + f(U[x,2])) # straddles the axis
    elseif y==n
        -zeroT(p).*(f(U[x,n-1])) # includes the axis
        #-0.5*f(U[x,n-1]) + 0.5*p*f(U[x,n-1]) # includes the axis
        #0.5*(-f(U[x,n-1]) + p*parity.*f(U[x,n])) # straddles the axis
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

@inline function Div2T(vr::Function,vθ::Function,U,r,θ,ns,_ds,x,y)
    (Dρ2T(vr,U,r,θ,ns,_ds,x,y,-1) + Dz2T(vθ,U,r,θ,ns,_ds,x,y,1))/rootγ(U[x,y])
end

@inline function ψu(U::StateVector) # Scalar gradient-flux

    # Give names to stored arrays from the state vector
    _,ψr,ψθ,Ψ,g,_,_,_ = unpack(U,false)

    gi = inverse(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    return βr*ψr + βθ*ψθ - α*Ψ

end

@inline function ψvr(U::StateVector) # r component of the divergence-flux

    # Give names to stored arrays from the state vector
    _,ψr,ψθ,Ψ,g,_,_,_ = unpack(U,false)

    _,_,_,γs... = g.data

    γ = TwoTensor(γs)

    γi = inverse(γ)

    gi = inverse(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]

    return rootγ(U)*(βr*Ψ - α*(γi[1,1]*ψr + γi[1,2]*ψθ))
    
end

@inline function ψvθ(U::StateVector) # θ component of the divergence-flux

    # Give names to stored arrays from the state vector
    _,ψr,ψθ,Ψ,g,_,_,_ = unpack(U,false)

    _,_,_,γs... = g.data

    γ = TwoTensor(γs)

    γi = inverse(γ)

    gi = inverse(g)

    α = 1/sqrt(-gi[1,1])

    βθ = -gi[1,3]/gi[1,1]

    return rootγ(U)*(βθ*Ψ - α*(γi[2,1]*ψr + γi[2,2]*ψθ))
    
end

@inline function u_odd(U::StateVector{Type}) # Scalar gradient-flux

    # Give names to stored arrays from the state vector
    # _,_,_,_,g,dr,dθ,P = unpack(U,false)
    # ρ = U.ρ

    # Give names to stored arrays from the state vector
    g = U.g 
    dr = U.dr   
    dθ = U.dθ  
    P  = U.P 

    gi = inverse(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    # if ρ == 0.
    #     mask = StateTensor{Type}((1.,1/ρ,1.,1.,1/ρ,1.))
    # else
    #     mask = StateTensor{Type}((1.,0.,1.,1.,0.,1.))
    # end

    return (βr*dr + βθ*dθ - α*P)

end

@inline function u(U::StateVector) # Scalar gradient-flux

    # Give names to stored arrays from the state vector
    _,_,_,_,g,dr,dθ,P = unpack(U,false)

    gi = inverse(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    return βr*dr + βθ*dθ - α*P

end

@inline function vr(U::StateVector) # r component of the divergence-flux

    # Give names to stored arrays from the state vector
    _,_,_,_,g,dr,dθ,P = unpack(U,false)

    _,_,_,γs... = g.data

    γ = TwoTensor(γs)

    γi = inverse(γ)

    gi = inverse(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]

    return rootγ(U)*(βr*P - α*(γi[1,1]*dr + γi[1,2]*dθ))
    
end

@inline function vθ(U::StateVector) # θ component of the divergence-flux

    # Give names to stored arrays from the state vector
    _,_,_,_,g,dr,dθ,P = unpack(U,false)

    _,_,_,γs... = g.data

    γ = TwoTensor(γs)

    γi = inverse(γ)

    gi = inverse(g)

    α = 1/sqrt(-gi[1,1])

    βθ = -gi[1,3]/gi[1,1]

    return rootγ(U)*(βθ*P - α*(γi[2,1]*dr + γi[2,2]*dθ))
    
end


@inline function rootγ(U::StateVector)

    # Unpack the metric into indiviual components
    _,_,_,_,g,_,_,_ = unpack(U,false)

    _,_,_,γs... = g.data

    γ = TwoTensor(γs)

    detγ = det(γ)

    if detγ < 0
        display(U.g)
        display(g)
    end

    return sqrt(detγ)

    #return 1.
end

function divergent_terms(Um,U_init,H,x,y,axis=false)

    # if axis

    #     U1 = Um[x,y]

    #     ψ,ψr,ψθ,Ψ,g,dr,dθ,P = unpack(U,false)

    #     det = g[1,1]*g[3,3] - g[1,3]^2

    #     g12 = dr[1,2]
    #     g23 = dr[2,3]

    #     gi11 =  g[3,3]/det
    #     gi22 =  1/g[2,2]
    #     gi13 = -g[1,3]/det
    #     gi33 =  g[1,1]/det

    #     gi12 = gi22*(g[1,3]*g23-g12*g[3,3])/det
    #     gi23 = gi22*(g[1,3]*g12-g23*g[1,1])/det


    #     return (S,St) 
    # else
        Type = Data.Number

        U = Um[x,y]

        ρ = U.ρ

        #r = rm[x,y]; θ = θm[x,y];

        Hxy = H[x,y]

        # Give names to stored arrays from the state vector
        ψ,ψr,ψθ,Ψ,g,dr,dθ,P = unpack(U,false)

        gi = inverse(g)

        # git = symmetric(@einsum  )

        # if (x==10 && y==2)
        #     display(gi[2,2]-exp(-4ψ))
        # end

        # if axis 
        #     ec = StateTensor{Type}((1,0,1,1,0,1))
        #     gi = ec.*gi
        # end 

        α = 1/sqrt(-gi[1,1])

        βr = -gi[1,2]/gi[1,1]
        βθ = -gi[1,3]/gi[1,1]

        ∂tψ  = βr*ψr + βθ*ψθ - α*Ψ

        ∂ψ   = @Vec [∂tψ,ψr,ψθ]

        ∂f = @Vec [0.,1/ρ,0.]

        #∂f = @Vec [0.,1.,0.]

        # Calculate time derivative of the metric
        ∂tg = βr*dr + βθ*dθ - α*P

        ∂g = Symmetric3rdOrderTensor{Type}(
            (σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dr[μ,ν] : σ==3 ? dθ[μ,ν] : @assert false)
            )

        Γ = Symmetric3rdOrderTensor{Type}(
            (σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν])
            )

        #Γv = @einsum (σ) -> gi[μ,ν]*Γ[σ,μ,ν]

        C_ = @einsum gi[μ,ν]*Γ[σ,μ,ν] - Hxy[σ]
            
        #S = symmetric(@einsum (μ,ν) -> 4*∂ψ[μ]*∂f[ν] + 2*C[μ]*∂f[ν] - gi[ρ,σ]*Γ[ρ,μ,ν]*∂f[σ] - g[μ,ν]*gi[ρ,σ]*Hxy[ρ]*∂f[σ])

        S = symmetric(@einsum (μ,ν) -> 4*∂ψ[μ]*∂f[ν] - gi[ρ,σ]*Γ[ρ,μ,ν]*∂f[σ] - g[μ,ν]*gi[ρ,σ]*Hxy[ρ]*∂f[σ])

        #S += symmetric(@einsum (μ,ν) -> g[μ,ν]*gi[i,j]*C_[i]*∂f[j])

        St = 4*(@einsum gi[μ,ν]*∂ψ[μ]*∂f[ν])
        
        St -= 4*(@einsum gi[μ,ν]*Hxy[μ]*∂f[ν])

        #St += (@einsum gi[i,j]*C_[i]*∂f[j])

        #S = symmetric(@einsum (μ,ν) -> 4*∂ψ[μ]*∂f[ν] - gi[ρ,σ]*Γ[ρ,μ,ν]*∂f[σ] - g[μ,ν]*gi[ρ,σ]*Γv[ρ]*∂f[σ])

        # St = 4*(@einsum gi[μ,ν]*∂ψ[μ]*∂f[ν])
        
        # St -= 4*(@einsum gi[μ,ν]*Γv[μ]*∂f[ν])

        if axis
            ec = StateTensor{Type}((1,0,1,1,0,1))
            S = ec.*S
        end

        #zero = StateTensor{Type}((1,0,1,0,0,1))

        return (S,St) 

    #end

end

function constraints(U::StateVector{Type}) where Type

    _,_,_,_,g,dr,dθ,P = unpack(U,false)

    gi = inverse(g)

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

    for x in 1:ns[1], y in 1:ns[2]

        Uxy = U[x,y]
        Hxy = H[x,y]

        _,_,_,_,g,_,_,_ = unpack(Uxy,false)

        #g  = Uxy.g 

        # Unpack the metric into individual components
        _,_,_,γs... = g.data

        γ = TwoTensor(γs)

        rootγ = sqrt(det(γ))

        γi2 = inverse(γ)

        γi = StateTensor((0.,0.,0.,γi2.data...))

        gi = inverse(g)

        Cxy = constraints(Uxy) - Hxy

        # Calculate lapse and shift
        α  = 1/sqrt(-gi[1,1])
        βr = -gi[1,2]/gi[1,1]
        βθ = -gi[1,3]/gi[1,1]

        nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 

        n = @Vec [nt,nr,nθ]

        C = @einsum n[μ]*Cxy[μ]

        Cij = @einsum γi[μ,ν]*Cxy[μ]*Cxy[ν]

        if ((x==1 || x==ns[1]) || (y==1 || y==ns[2])) 
            int += 0.5*abs((C^2+Cij)*rootγ/_ds[1]/_ds[2])
        else
            int += abs((C^2+Cij)*rootγ/_ds[1]/_ds[2])
        end

    end

    return sqrt(int)

end

@inline function unpack(U::StateVector{T}, SV = true) where T

    # Give names to stored arrays from the state vector
    ρ  = U.ρ
    ψv = U.ψ
    g = U.g 
    dρ = U.dr   
    dz = U.dθ  
    P  = U.P 

    # mask1 = StateTensor{T}((1.,ρ,1.,1.,ρ,1.))
    # mask2 = StateTensor{T}((0.,1.,0.,0.,1.,0.))
    # mask3 = @Vec  [1.,ρ,1.,1.]

    # g  = mask1.*g
    # dz = mask1.*dz
    # P  = mask1.*P

    # dρ = ρ*dρ + mask2.*g

    #ψv = mask3.*ψv


    # dx = StateTensor{T}((ρ*dx[1,1],dx[1,2],ρ*dx[1,3],∂xgρρ,dx[2,3],ρ*dx[3,3]))
    # dy = StateTensor{T}((dy[1,1],ρ*dy[1,2],dy[1,3],∂ygρρ,ρ*dy[2,3],dy[3,3]))
    # P  = StateTensor{T}((P[1,1],ρ*P[1,2],P[1,3],Pρρ,ρ*P[2,3],P[3,3]))

    # gi = inv(g)
    # α = 1/sqrt(-gi[1,1])
    # βx = -gi[1,2]/gi[1,1]
    # nρ = -βx/α

    # χ,dρχ,dzχ,dnχ = ψv.data
    
    # if ρ == 0.

    #     ψ  = log(g[2,2])/4
    #     ψρ = 0.
    #     ψz = dz[2,2]/g[2,2]/4
    #     Ψ  =  P[2,2]/g[2,2]/4

    # else

    #     ψ  = log(g[2,2])/4 + ρ*χ 
    #     ψρ = dρ[2,2]/g[2,2]/4 + ρ*dρχ + χ/ρ 
    #     ψz = dz[2,2]/g[2,2]/4 + ρ*dzχ 
    #     Ψ  =  P[2,2]/g[2,2]/4 + ρ*dnχ - nρ*χ/ρ 

    # end
    
    #ψ,ψx,ψy,Ψ = ψv.data

    # G,dxG,dyG,PG = ψv.data

    # χ   = gs[2,2]
    # ∂xχ = dx[2,2]
    # ∂yχ = dy[2,2]
    # ∂nχ =  P[2,2]

    # F   =  gs[2,2]
    # dxF = dx[2,2]
    # dyF = dy[2,2]
    # PF  =  P[2,2]

    # #gρρ = exp(4ψ + ρ*χ)
    # gρρ = F^2 - ρ^4*G^2

    # g = StateTensor{T}((gs[1,1],ρ*gs[1,2],gs[1,3],gρρ,ρ*gs[2,3],gs[3,3]))

    # gi = inv(g)
    # α = 1/sqrt(-gi[1,1])
    # βx = -gi[1,2]/gi[1,1]
    # nρ = -βx/α

    ############################################################################

    # ∂xgρρ = gρρ*(4*ψx + ρ*∂xχ + χ)
    # ∂ygρρ = gρρ*(4*ψy + ρ*∂yχ)
    # Pρρ   = gρρ*(4*Ψ  + ρ*∂nχ - nρ*χ)
    
    # dx = StateTensor{T}((dx[1,1],dx[1,2],dx[1,3],∂xgρρ,dx[2,3],dx[3,3]))
    # dy = StateTensor{T}((dy[1,1],dy[1,2],dy[1,3],∂ygρρ,dy[2,3],dy[3,3]))
    # P  = StateTensor{T}((P[1,1],P[1,2],P[1,3],Pρρ,P[2,3],P[3,3]))

    # ∂xgρρ = 2F*dxF - 2ρ^4*G*dxG - 4ρ^3*G^2
    # ∂ygρρ = 2F*dyF - 2ρ^4*G*dyG
    # Pρρ   = 2F*PF  - 2ρ^4*G*PG + 4*nρ*ρ^3*G^2

    # ψ = 0.5*log(F-ρ^2*G)
    # ψx = 0.5*(dxF-ρ^2*dxG-2ρ*G)/exp(2ψ)
    # ψy = 0.5*(dyF-ρ^2*dyG)/exp(2ψ)
    # Ψ = 0.5*(PF-ρ^2*PG+2nρ*ρ*G)/exp(2ψ)

    # ψv = @Vec [ψ,ρ*ψx,ψy,Ψ]

    # dx = StateTensor{T}((ρ*dx[1,1],dx[1,2],ρ*dx[1,3],∂xgρρ,dx[2,3],ρ*dx[3,3]))
    # dy = StateTensor{T}((dy[1,1],ρ*dy[1,2],dy[1,3],∂ygρρ,ρ*dy[2,3],dy[3,3]))
    # P  = StateTensor{T}((P[1,1],ρ*P[1,2],P[1,3],Pρρ,ρ*P[2,3],P[3,3]))

    # if ρ == 0.
    #     dx = StateTensor{T}((0.,dx[1,2],0.,0.,dx[2,3],0.))
    #     dy = StateTensor{T}((dy[1,1],0.,dy[1,3],∂ygρρ,0.,dy[3,3]))
    #     P  = StateTensor{T}((P[1,1],0.,P[1,3],Pρρ,0.,P[3,3]))
    #     ψx = 0.
    # else

    # end

    # Rescale the ρρ component of the metric to make it the literal metric
    # dx  = replace_comp(dx,∂xgρρ)
    # dy  = replace_comp(dy,∂ygρρ)
    # P   = replace_comp(P,Pρρ)

    # dx = StateTensor{Type}((0.,ρ*dx[1,2]+g[1,2],0.,∂xgρρ,ρ*dx[2,3]+g[2,3],0.))

    # dy = StateTensor{Type}((dy[1,1],ρ*dy[1,2],dy[1,3],∂ygρρ,ρ*dy[2,3],dy[3,3]))

    # P  = StateTensor{Type}((P[1,1],ρ*P[1,2]-nρ*g[1,2],P[1,3],Pρρ,ρ*P[2,3]-nρ*g[2,3],P[3,3]))

    #ψv = @Vec [ψ,ψρ,ψz,Ψ]

    if SV
        return StateVector{T}(ρ,ψv,g,dρ,dz,P)
    else
        return (ψv.data...,g,dρ,dz,P)
    end

end

@inline function pack(U::StateVector{Type}) where Type

    # Give names to stored arrays from the state vector
    ρ  = U.ρ
    ψv = U.ψ
    g  = U.g 
    dρ = U.dr   
    dz = U.dθ  
    P  = U.P 

    #ψ,ψx,ψy,Ψ = ψv.data

    # gρρ   =  g[2,2]
    # ∂xgρρ = dx[2,2]
    # ∂ygρρ = dy[2,2]
    # Pρρ   =  P[2,2]

    # gi = inv(g)
    # α = 1/sqrt(-gi[1,1])
    # βx = -gi[1,2]/gi[1,1]
    # nρ = -βx/α

    # if ρ == 0.

    #     mask1 = StateTensor{Type}((1.,0.,1.,1.,0.,1.))
    #     mask2 = StateTensor{Type}((0.,1.,0.,0.,1.,0.))
    #     mask3 = @Vec [1.,0.,1.,1.]

    #     g  = mask1.*g
    #     dz = mask1.*dz
    #     P  = mask1.*P

    #     dρ = mask2.*dρ

    #     ψv = mask3.*ψv

    # else

    #     mask1 = StateTensor{Type}((1.,1/ρ,1.,1.,1/ρ,1.))
    #     mask2 = StateTensor{Type}((1/ρ,1.,1/ρ,1/ρ,1.,1/ρ))
    #     mask3 = @Vec [1.,1/ρ,1.,1.]

    #     g  = mask1.*g
    #     dz = mask1.*dz
    #     P  = mask1.*P

    #     dρ = mask2.*dρ

    #     ψv = mask3.*ψv
    # end

    # g  = StateTensor{Type}((g[1,1],g[1,2],g[1,3],χ,g[2,3],g[3,3]))
    # dx = StateTensor{Type}((dx[1,1],dx[1,2],dx[1,3],∂xχ,dx[2,3],dx[3,3]))
    # dy = StateTensor{Type}((dy[1,1],dy[1,2],dy[1,3],∂yχ,dy[2,3],dy[3,3]))
    # P  = StateTensor{Type}((P[1,1],P[1,2],P[1,3],∂nχ,P[2,3],P[3,3]))

    # if ρ == 0.

    #     F   = (exp(2ψ) + g[2,2]*exp(-2ψ))/2
    #     dxF = (2*exp(2ψ)*ψx + dx[2,2]*exp(-2ψ) - 2g[2,2]*exp(-2ψ)*ψx )/2
    #     dyF = (2*exp(2ψ)*ψy + dy[2,2]*exp(-2ψ) - 2g[2,2]*exp(-2ψ)*ψy )/2
    #     PF  = (2*exp(2ψ)*Ψ  +  P[2,2]*exp(-2ψ) - 2g[2,2]*exp(-2ψ)*Ψ  )/2

    #     G   = 0.
    #     dxG = 0.
    #     dyG = 0.
    #     PG  = 0.

    #     g  = StateTensor{Type}((g[1,1],0.,g[1,3],F,0.,g[3,3]))
    #     dx = StateTensor{Type}((0.,dx[1,2],0.,0.,dx[2,3],0.))
    #     dy = StateTensor{Type}((dy[1,1],0.,dy[1,3],dyF,0.,dy[3,3]))
    #     P  = StateTensor{Type}((P[1,1],0.,P[1,3],PF,0.,P[3,3]))
    
    #     ψv = @Vec [G,dxG,dyG,PG]

    # else

    #     F   = (exp(2ψ) + g[2,2]*exp(-2ψ))/2
    #     dxF = (2*exp(2ψ)*ψx + dx[2,2]*exp(-2ψ) - 2g[2,2]*exp(-2ψ)*ψx )/2
    #     dyF = (2*exp(2ψ)*ψy + dy[2,2]*exp(-2ψ) - 2g[2,2]*exp(-2ψ)*ψy )/2
    #     PF  = (2*exp(2ψ)*Ψ  +  P[2,2]*exp(-2ψ) - 2g[2,2]*exp(-2ψ)*Ψ  )/2

    #     G   = (-exp(2ψ) + g[2,2]*exp(-2ψ))/2/ρ^2
    #     dxG = (-2*exp(2ψ)*ψx + dx[2,2]*exp(-2ψ) - 2g[2,2]*exp(-2ψ)*ψx )/2/ρ^2 - 2*G/ρ
    #     dyG = (-2*exp(2ψ)*ψy + dy[2,2]*exp(-2ψ) - 2g[2,2]*exp(-2ψ)*ψy )/2/ρ^2
    #     PG  = (-2*exp(2ψ)*Ψ  +  P[2,2]*exp(-2ψ) - 2g[2,2]*exp(-2ψ)*Ψ  )/2/ρ^2 + 2*nρ*G/ρ

    #     g  = StateTensor{Type}((g[1,1],g[1,2]/ρ,g[1,3],F,g[2,3]/ρ,g[3,3]))
    #     dx = StateTensor{Type}((dx[1,1]/ρ,dx[1,2],dx[1,3]/ρ,dxF,dx[2,3],dx[3,3]/ρ))
    #     dy = StateTensor{Type}((dy[1,1],dy[1,2]/ρ,dy[1,3],dyF,dy[2,3]/ρ,dy[3,3]))
    #     P  = StateTensor{Type}((P[1,1],P[1,2]/ρ,P[1,3],PF,P[2,3]/ρ,P[3,3]))
    
    #     ψv = @Vec [G,dxG,dyG,PG]

    # end


    return StateVector{Type}(ρ,ψv,g,dρ,dz,P)

end

@parallel_indices (x,y) function rhs!(Type,U1,U2,U3,U_init,C1,C2,C3,H,∂H,rm,θm,t,ns,dt,_ds,iter)

    #Explicit slices from main memory
    # At each iteration in an Runge-Kutta algorithm,
    # a U-read (U) and U-write (Uw) are defined
    if iter == 1
        # U3 has past iteration.
        U = U1
        Uw = U2
        Uxy = U[x,y]
    elseif iter == 2
        # U1 has past iteration.
        U = U2
        Uw = U3
        Uxy = U[x,y]
    else
        # U2 has past iteration.
        U = U3
        Uw = U1
        Uxy = U[x,y]
    end

    Hxy = H[x,y]; ∂Hxy = ∂H[x,y];

    r = rm[x,y]; θ = θm[x,y];

    ρ = Uxy.ρ

    ψ,ψr,ψθ,Ψ,g,dr,dθ,P = unpack(Uxy,false)

    # Calculate inverse metric components
    gi = inverse(g)

    # Calculate lapse and shift
    α  = 1/sqrt(-gi[1,1])
    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    # Time derivatives of the metric
    ∂tg = βr*dr + βθ*dθ - α*P

    ################################################

    # χ, ∂ρχ, ∂zχ, ∂tχ = Aρ(chi,U,r,θ,ns,_ds,x,y).data

    # # Calculate lapse and shift
    # # α  = 1/sqrt(-gi[1,1])
    # # βρ = -gi[1,2]/gi[1,1]
    # # βz = -gi[1,3]/gi[1,1]

    # # Time derivatives of the metric
    # ∂tg = βr*dr + βθ*dθ - α*P
    # ∂tψ = βr*ψr + βθ*ψθ - α*Ψ

    # ψ   =  log(g[2,2])/4 + ρ*χ/2 
    # ψr  =  dr[2,2]/g[2,2]/4 + ρ*∂ρχ/2 + χ/2
    # ψθ  =  dθ[2,2]/g[2,2]/4 + ρ*∂zχ/2 
    # ∂tψ = ∂tg[2,2]/g[2,2]/4 + ρ*∂tχ/2 

    # Ψ = -(∂tψ - βr*ψr - βθ*ψθ)/α 

    ################################################

    # # Unpack the metric into individual components
    # _,_,_,γs... = g.data

    # γ = TwoTensor{Type}(γs)

    # detγ = det(γ)

    # if detγ < 0
    #     println(x," ",y," ",g[1,1]," ",g[2,2]," ",g[3,3]," ")
    # end

    # γi2 = inv(γ)

    # γi = StateTensor{Type}((0.,0.,0.,γi2.data...))

    nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 

    n = @Vec [nt,nr,nθ]

    n_ = @Vec [-α,0.0,0.0]

    γi = gi + symmetric(@einsum n[μ]*n[ν])

    δ = one(ThreeTensor)

    γm = δ + (@einsum n_[μ]*n[ν])

    γ = g + symmetric(@einsum n_[μ]*n_[ν])

    #Derivatives of the lapse and the shift 

    # ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])
    # ∂tβ = α*(@einsum γi[α,μ]*n[ν]*∂tg[μ,ν]) # result is a 3-vector

    # Metric derivatives
    ∂g = Symmetric3rdOrderTensor{Type}(
        (σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dr[μ,ν] : σ==3 ? dθ[μ,ν] : @assert false)
        )

    # Chistoffel Symbols (of the first kind, i.e. all covariant indices)
    Γ  = Symmetric3rdOrderTensor{Type}((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))

    C_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ] - Hxy[μ] #- ∂ψ[α]

    Cr = (Dρ2T(fg,U,r,θ,ns,_ds,x,y) - dr)
    Cθ = (Dz2T(fg,U,r,θ,ns,_ds,x,y) - dθ)

    Cψr = (Dρ2(fψ,U,r,θ,ns,_ds,x,y) - ψr)
    Cψθ = (Dz2(fψ,U,r,θ,ns,_ds,x,y) - ψθ)

    # C2 = Tensor{Tuple{4,@Symmetry{4,4}},S}((σ,μ,ν) -> (σ==1 ? 0.0 : σ==2 ? Cr[μ,ν] : σ==3 ? Cθ[μ,ν] : σ==4 ? 0. : @assert false))

    # if (x == 100 && y == 3 && iter==4) 
    #     display(C) 
    # end

    #∂∂f = StateTensor{Type}((0.,0.,0.,1/r^2,0.,1/sin(θ)^2))

    #∂f = @Vec [0.,1/r/sin(θ),0.]

    #∂f = @Vec [0.,1/r/sin(θ),0.]

    #δ = one(StateTensor{Type})

    # Scalar Evolution
    ######################################################################

    if  y == 1
        
        S1,St1 = divergent_terms(U,U_init,H,x,y+1,true)
        S2,St2 = divergent_terms(U,U_init,H,x,y+2,true)

        S = (16*S1 - 4*S2)/12
        St = (16*St1 - 4*St2)/12

    elseif y==ns[2]

        S1,St1 = divergent_terms(U,U_init,H,x,y-1,true)
        S2,St2 = divergent_terms(U,U_init,H,x,y-2,true)

        S = (16*S1 - 4*S2)/12
        St = (16*St1 - 4*St2)/12

    # elseif y==ns[2]

    #     S1,St1 = divergent_terms(U,U_init,H,x,y-1)
    #     S2,St2 = divergent_terms(U,U_init,H,x,y-2)

    #     S = (16*S1 - 4*S2)/12
    #     St = (16*St1 - 4*St2)/12

    else
        S,St = divergent_terms(U,U_init,H,x,y)
    end

    #S,St = divergent_terms(U,rm,θm,H,x,y)

    # if  y==1 || y==ns[2]
    #     S = StateTensor{Type}((S[1,1],0.,S[1,3],S[2,2],0.,S[3,3]))
    # end

    #S = divergent_terms(U,rm,θm,x,y)

    #St = @einsum gi[μ,ν]*S[μ,ν]  

    # if  y == 1

    #     #S = divergent_terms(U,rm,θm,x,y+1)

    #     ∂f = @Vec [0.,1/rm[x,y+1]/sin(θm[x,y+1]),0.] 

    # elseif y==ns[2]

    #     ∂f = @Vec [0.,1/rm[x,y-1]/sin(θm[x,y-1]),0.] 

    # else

    #     ∂f = @Vec [0.,1/rm[x,y]/sin(θm[x,y]),0.] 

    # end

    # St = @einsum 4*gi[μ,ν]*∂ψ[μ]*∂f[ν] 

    # S = zero(StateTensor{Type})
    # St = 0.

    γ1 = -1.;
    γ0 =  1.# + 9/(ρ+1)
    γ2 = 1.;

    ∂tψ  = βr*ψr + βθ*ψθ - α*Ψ

    ∂ψ   = @Vec [∂tψ,ψr,ψθ]
    
    ∂trootγ = @einsum 0.5*γi[i,j]*∂tg[i,j]

    #######################################################################
    # Define Stress energy tensor and trace 
    # T = zero(StateTensor{Type})
    # Tt = 0.

    ∂tP = -2*α*S  # + 8*pi*Tt*g - 16*pi*T 

    ∂tP += -2*α*symmetric(@einsum (μ,ν) -> 2*∂ψ[μ]*∂ψ[ν])  # + 8*pi*Tt*g - 16*pi*T 

    ∂tP += 2*α*symmetric(∂Hxy)

    #∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*Hxy[ϵ]*∂g[μ,ν,σ])

    ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*Hxy[ϵ]*Γ[σ,μ,ν])

    ∂tP -=  α*symmetric(@einsum (μ,ν) -> gi[λ,γ]*gi[ϵ,σ]*Γ[λ,ϵ,σ]*∂g[γ,μ,ν])

    ∂tP += 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*∂g[λ,ϵ,μ]*∂g[ρ,σ,ν])

    ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*Γ[μ,ϵ,λ]*Γ[ν,σ,ρ])

    # Constraint damping term for C_

    ∂tP += γ0*α*symmetric(@einsum (μ,ν) -> 2C_[μ]*n_[ν] - g[μ,ν]*n[ϵ]*C_[ϵ])

    # ρv = @Vec [0.,1.,0.]
    # ρv_ = @einsum g[μ,ν]*ρv[μ]
    
    # ∂tP += -2*γ0*α*exp(-ψ)*symmetric(@einsum (μ,ν) -> ρv_[μ]*n_[ν]*ρv[i]*C_[i])

    # if (y==1 || y==ns[2])
    
    # end

    #∂tP += -2*α*symmetric(@einsum (μ,ν) -> 2*C_[μ]*∂ψ[ν] - g[μ,ν]*gi[α,β]*∂ψ[α]*C_[β])

    #∂tP += γ0*α*exp(-ψ)*symmetric(@einsum (μ,ν) -> 2*n_[μ]*C_[ν] - g[μ,ν]*n[ϵ]*C_[ϵ])

    #vt = @Vec [0.,ρ,0.]

    #∂tP -= 2*α*g*(@einsum gi[μ,ν]*vt[μ]*C_[ν])

    #∂tP -= 2*α*g*(@einsum gi[μ,ν]*∂ψ[μ]*C_[ν])

    #∂tP -= γ0*α*g*exp(-ψ)*(@einsum n[ϵ]*C_[ϵ])

    ∂tP -= ∂trootγ*P

    ∂tP += γ1*γ2*(βr*Cr + βθ*Cθ)

    ###########################################
    # All finite differencing occurs here

    # mask1 = StateTensor{Type}((1.,ρ,1.,1.,ρ,1.))
    # mask2 = StateTensor{Type}((0.,1.,0.,0.,1.,0.))

    ∂tP += Div4T(vr,vθ,U,r,θ,ns,_ds,x,y) 

    ∂tdr = symmetric(Dρ4T(u,U,r,θ,ns,_ds,x,y)) + α*γ2*Cr #+ mask2.*u_odd(U) 

    ∂tdθ = symmetric(Dz4T(u,U,r,θ,ns,_ds,x,y)) + α*γ2*Cθ

    ##################################################

    ∂tP = symmetric(∂tP)

    ##################################################

    ∂tψr = Dρ4(ψu,U,r,θ,ns,_ds,x,y) + α*γ2*Cψr

    ∂tψθ = Dz4(ψu,U,r,θ,ns,_ds,x,y) + α*γ2*Cψθ

    ∂tΨ  = Div4(ψvr,ψvθ,U,r,θ,ns,_ds,x,y) #- (α/4)*St 

    #∂tΨ  -= α*(@einsum gi[i,j]*∂ψ[i]*C_[j]) 

    #∂tΨ  += α*γ0*(@einsum n[i]*C_[i]) 

    ∂tΨ  -= ∂trootγ*Ψ

    # ∂tψ  = 0.
    # ∂tψr = 0.
    # ∂tψθ = 0.
    # ∂tΨ  = 0.

    ######################################################

    ∂ρg = Dρ2T(fg,U,r,θ,ns,_ds,x,y)
    ∂zg = Dz2T(fg,U,r,θ,ns,_ds,x,y)
    ∂ρψ = Dρ2(fψ,U,r,θ,ns,_ds,x,y)
    ∂zψ = Dz2(fψ,U,r,θ,ns,_ds,x,y)

    # ∂tg = βr*∂ρg + βθ*∂zg - α*P

    ###################################################
    #Boundary conditions

    c1 = (x==1); c2 = (x==ns[1]);

    if (c1 || c2) #&& false

        if c1 
            b=1; p=-1
        else 
            b=2; p=1
        end

        Cy = C1[b,y]

        # if iter == 1
        #     C = C1
        #     Cw = C2
        #     Cy = C[b,y]
        # elseif iter == 2
        #     C = C2
        #     Cw = C3
        #     Cy = C[b,y]
        # else
        #     C = C3
        #     Cw = C1
        #     Cy = C[b,y]
        # end

        # Non-conformally transformed metric and inverse
        # h = exp(-2*ψ)*g
        # hi = exp(2*ψ)*gi

        # ∂h = exp(-2*ψ)*(@einsum (α,μ,ν) -> ∂g[α,μ,ν] - 2*g[μ,ν]*∂ψ[α])

        # ∂ρψ = Dρ2(fψ,U,r,θ,ns,_ds,x,y)
        # ∂zψ = Dz2(fψ,U,r,θ,ns,_ds,x,y)

        # ∂g = Symmetric3rdOrderTensor{Type}(
        #     (σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? ∂ρg[μ,ν] : σ==3 ? ∂zg[μ,ν] : @assert false)
        #     )

        h = g
        hi = gi
        ∂h = ∂g
    
        α  = 1/sqrt(-hi[1,1])
        βr = -hi[1,2]/hi[1,1]
        βθ = -hi[1,3]/hi[1,1]
    
        nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 
    
        n = @Vec [nt,nr,nθ]
    
        n_ = @Vec [-α,0.0,0.0]

        # Form the unit normal vector to the boundary

        s = @Vec [0.0,p*sin(θ),p*cos(θ)]

        snorm = @einsum h[μ,ν]*s[μ]*s[ν]
    
        s = s/sqrt(snorm) 

        s_ = @einsum h[μ,ν]*s[ν]
    
        # Form the unit tangent to the boundary

        # if y == 1 || y== ns[2]
        #     Θ_ = @Vec [0.,0.,0.]
        # else

        # end
        #Θ = @Vec [0.0,r*cos(θ),-r*sin(θ)]

        Θ_ = @Vec [βr*cos(θ)-βθ*sin(θ),cos(θ),-sin(θ)]
        Θnorm = @einsum hi[μ,ν]*Θ_[μ]*Θ_[ν]
        Θ_ = Θ_/sqrt(Θnorm)

        Θ = @einsum hi[μ,ν]*Θ_[ν]

        # Form ingoing and outgoing null vectors

        ℓ = @einsum (n[α] + s[α])/sqrt(2)
        k = @einsum (n[α] - s[α])/sqrt(2)

        ℓ_ = @einsum h[μ,α]*ℓ[α]
        k_ = @einsum h[μ,α]*k[α]

        # σ = StateTensor{Type}((μ,ν) -> gi[μ,ν] + k[μ]*ℓ[ν] + ℓ[μ]*k[ν])
        
        # σm = @einsum g[μ,α]*σ[α,ν] # mixed indices (raised second index)

        # σ_ = @einsum g[μ,α]*σm[ν,α]

        σ = @einsum Θ[μ]*Θ[ν]
        
        σm = @einsum Θ_[μ]*Θ[ν] # mixed indices (raised second index)

        σ_ = @einsum Θ_[μ]*Θ_[ν]

        # if (y==1 && iter==1)
        #     gv = @einsum (s_[μ]*s_[ν] - n_[μ]*n_[ν] + Θ_[μ]*Θ_[ν])
        #     display(g-gv)
        # end
    
        # if (c1 && y==50 && iter==1)
        #     # gv = @einsum (s_[μ]*s_[ν] - n_[μ]*n_[ν] + Θ_[μ]*Θ_[ν])
        #     # giv = @einsum (s[μ]*s[ν] - n[μ]*n[ν] + Θ[μ]*Θ[ν])
        #     # display(g-gv)
        #     display(g)
        # end

        cp =  α - βr*s_[2] - βθ*s_[3]
        cm = -α - βr*s_[2] - βθ*s_[3]
        c0 =    - βr*s_[2] - βθ*s_[3]

        # if y==1 && x==ns[1]
        #     println(cp," ",cm," ",c0)
        # end

        βdotθ = βr*Θ_[2] + βθ*Θ_[3]
    
        Up = @einsum StateTensor{Type} k[α]*∂h[α,μ,ν]
        #Um = @einsum StateTensor{Type} ℓ[α]*∂h[α,μ,ν]
        U0 = @einsum StateTensor{Type} Θ[α]*∂h[α,μ,ν]

        # Up = -(P + s[2]*dr + s[3]*dθ)/sqrt(2)
        # U0 = Θ[2]*dr + Θ[3]*dθ

        Uψp = @einsum k[α]*∂ψ[α]
        Uψ0 = @einsum Θ[α]*∂ψ[α]

        # U0b = Θ[2]*∂ρg + Θ[3]*∂zg
        # Uψ0b = Θ[2]*∂ρψ + Θ[3]*∂zψ

        #if (c0 < 0) U0 = U0b; Uψ0 = Uψ0b; end

        # Characteristics for Killing Scalar
    
        # Boundary Condition:
        # You get to choose the incoming 
        # characteristic modes (Um)
        # Pick a function Um = f(Up,U0)
    
        #δ4 = one(SymmetricFourthOrderTensor{4})
        #δ = one(SymmetricFourthOrderTensor{3})

        # γp = @einsum δ[μ,ν] + n_[μ]*n[ν] 

        # Q4 = SymmetricFourthOrderTensor{3,Type}(
        #     (μ,ν,α,β) -> σ_[μ,ν]*σ[α,β] - 2*ℓ_[μ]*σm[ν,α]*k[β] + ℓ_[μ]*ℓ_[ν]*k[α]*k[β]
        # ) # Four index constraint projector (indices down down up up)
    
        Q3 = Symmetric3rdOrderTensor{Type}(
            (α,μ,ν) -> ℓ_[μ]*σm[ν,α]/2 + ℓ_[ν]*σm[μ,α]/2 - σ_[μ,ν]*ℓ[α] - ℓ_[μ]*ℓ_[ν]*k[α]/2
        ) # Three index constraint projector (indices up down down)
        # Note order of indices here

        # # O = SymmetricFourthOrderTensor{4}(
        # #     (μ,ν,α,β) -> σm[μ,α]*σm[ν,β] - σ_[μ,ν]*σ[α,β]/2
        # # ) # Gravitational wave projector

        G = FourthOrderTensor{3,Type}(
            (μ,ν,α,β) -> (2k_[μ]*ℓ_[ν]*k[α]*ℓ[β] - 2k_[μ]*σm[ν,α]*ℓ[β] + k_[μ]*k_[ν]*ℓ[α]*ℓ[β])
        ) # Four index gauge projector (indices down down up up)

        # G = minorsymmetric(G)

        Amp = 0.00001
        #Amp = 0.0
        σ0 = 0.5
        μ0 = 10.5

        #f(t,z) = (μ0-t-σ0)<z<(μ0-t+σ0) ? (Amp/σ0^8)*(z-((μ0-t)-σ0))^4*(z-((μ0-t)+σ0))^4 : 0.

        #f(t,z) = (μ0-t-σ0)<z<(μ0-t+σ0) ? Amp : 0.

        f(t,ρ,z) = (μ0-t-σ0)<ρ<(μ0-t+σ0) ? Amp : 0.

        if c2
            Cf = @Vec [f(t,r*sin(θ),r*cos(θ)),r*sin(θ)*f(t,r*sin(θ),r*cos(θ)),f(t,r*sin(θ),r*cos(θ))]
            #Cf = @Vec [0.,0.,0.] 
            #Cf = C_
        else
            #Cf = Cy

            # CBC = constraints(U[2,y]) 
            Cf = @Vec [0.,0.,0.] 
            #Cf = C_

            #Cf = @Vec [f(-t,r*cos(θ)),0.,f(-t,r*cos(θ))]
        end
    
        A_ = @einsum (2*ℓ[μ]*Up[μ,α] - hi[μ,ν]*Up[μ,ν]*ℓ_[α] + hi[μ,ν]*U0[μ,ν]*Θ_[α] - 2*Θ[μ]*U0[μ,α] + 2*Hxy[α] + 2*Cf[α])
        # # index down

        # # Condition ∂tgμν = 0 on the boundary
        Umb2 = (cp/cm)*Up + sqrt(2)*(βdotθ/cm)*U0
        #Umb2 = ℓ[1]*∂tg + ℓ[2]*∂ρg + ℓ[3]*∂zg
        #Umb2 = zero(StateTensor)

        #Umb2 = Cy.g

        # function f(A) 
        #     At_ = @Vec [A[1],A[2],A[3]]
        #     Umt = @einsum StateTensor{Type} (Q3[α,μ,ν]*At_[α] + G[μ,ν,α,β]*Umb2[α,β])
        #     Bt_ = @einsum (2*k[μ]*Umt[μ,α] - hi[μ,ν]*Umt[μ,ν]*k_[α] 
        #                   - hi[μ,ν]*Up[μ,ν]*ℓ_[α] + 2*ℓ[μ]*Up[μ,α] 
        #                   + hi[μ,ν]*U0[μ,ν]*Θ_[α] - 2*Θ[μ]*U0[μ,α] 
        #                   + 2*Hxy[α] + 2*Cf[α])
        #     return [Bt_.data...]
        # end

        # sol = nlsolve(f,[A_.data...])

        # Asol = sol.zero

        # Af_ = @Vec [Asol[1],Asol[2],Asol[3]]

        Umbh = @einsum StateTensor{Type} (Q3[α,μ,ν]*A_[α] + G[μ,ν,α,β]*Umb2[α,β])

        #Umbh = Umb2

        #Uψmb = 0.#-(@einsum ℓ[μ]*A_[μ])/4# - term

        # #Umb = exp(2*ψ)*Umbh + 2*g*Uψmb
        #Umb = Umbh

        #Umb = Umb2

        # if y==1 || y==ns[2]
        #     Umb = StateTensor{Type}((Umb[1,1],Umb[1,2],Umb[1,3],4*exp(4*ψ)*Uψmb,Umb[2,3],Umb[3,3]))
        # end

        # if y==1 || y==ns[2]

        #     # Axis regularity for boundary conditions
        #     #f2   = exp(4*ψ)
        #    # Umf2 = 4*exp(4*ψ)*Uψmb

        #     #2*f*∂zf
        #     f   = (exp(4*ψ)+g[2,2])/exp(2*ψ)/2

        #     Umf = (2*(exp(4*ψ)-g[2,2])*Uψmb + Umb[2,2])/exp(2*ψ)/2

        #     Umbρρ = 2*f*Umf
        #     Uψmb  = Umf/f/2

        #     Umb = StateTensor{Type}((Umb[1,1],Umb[1,2],Umb[1,3],Umbρρ,Umb[2,3],Umb[3,3]))

        # end

        #SAT type boundary conditions

        #ε = 2*_ds[1]
        #ε = 10*_ds[1]

        # δ = one(StateTensor)

        # γi = gi + symmetric(@einsum n[μ]*n[ν])

        # γm = δ + (@einsum n_[μ]*n[ν])

        # γ = g + symmetric(@einsum n_[μ]*n_[ν])

        # # dρ = Dρ2T(fg,U1,r[x,y],θ[x,y],ns,_ds,x,y)
        # # dz = Dz2T(fg,U1,r[x,y],θ[x,y],ns,_ds,x,y)

        # d = Symmetric3rdOrderTensor{Type}(
        #     (σ,μ,ν) -> (σ==1 ? 0. : σ==2 ? ∂ρg[μ,ν] : σ==3 ? ∂zg[μ,ν] : 0.)
        #     )

        # # C = FourthOrderTensor{3,T}(
        # #     (μ,ν,α,β)-> n_[μ]*n_[ν]*n[α]*n[β] - 2*n_[μ]*γm[ν,α]*n[β] + n_[μ]*n_[ν]*γi[α,β]
        # #     )

        # G = FourthOrderTensor{3,Type}(
        #     (μ,ν,α,β)-> γm[μ,α]*γm[ν,β] - n_[μ]*n_[ν]*γi[α,β]
        #     )

        # G = minorsymmetric(G)

        # C = Symmetric3rdOrderTensor{Type}(
        #     (α,μ,ν) -> n_[μ]*γm[ν,α]/2 + n_[ν]*γm[μ,α]/2 - n_[μ]*n_[ν]*n[α]
        # )

        # #A_ = @einsum ( 2*γi[i,μ]*d[i,μ,α] - gi[μ,ν]*γm[α,i]*d[i,μ,ν] - 2*Hxy[α] - 2*Cf[α])
        # # index down

        # P2 = (βr*∂ρg + βθ*∂zg)/α

        # Pb = symmetric(@einsum C[α,μ,ν]*A_[α] + G[μ,ν,α,β]*P2[α,β])
        # dxb = ∂ρg
        # dyb = ∂zg

        #if (cm < 0) Um = Umb end

        # Q4 = FourthOrderTensor{3,Type}(
        #     (μ,ν,α,β) -> σ_[μ,ν]*σ[α,β]/2 - 2*ℓ_[μ]*σm[ν,α]*k[β] + ℓ_[μ]*ℓ_[ν]*k[α]*k[β]
        # ) # Four index constraint projector (indices down down up up)
        # Q4 = minorsymmetric(Q4)

        # O = FourthOrderTensor{3,Type}(
        #     (μ,ν,α,β) -> σm[μ,α]*σm[ν,β] - σ_[μ,ν]*σ[α,β]/2 #- σ_[μ,ν]*ξ[α]*ξ[β]/2
        # ) # Gravitational wave projector
        # O = minorsymmetric(O)

        # #2ρ (-βρ/α + sρ)

        # Um2 = Cy.g
        # Um2h = exp(-2*ψ)*(Um2)
        # #∂g[α,μ,ν] - 2*g[μ,ν]*∂ψ[α]

        # Umb2 = (cp/cm)*Up + sqrt(2)*(βdotθ/cm)*U0

        # Umh = @einsum StateTensor{Type} (O[μ,ν,α,β]*Um2h[α,β] + G[μ,ν,α,β]*Um2h[α,β])

        # function f(A) 
        #     At_ = @Vec [A[1],A[2],A[3]]
        #     Umt = @einsum StateTensor{Type} (Q3[α,μ,ν]*At_[α] + O[μ,ν,α,β]*Um2h[α,β] + G[μ,ν,α,β]*Um2h[α,β])
        #     Bt_ = @einsum (-hi[μ,ν]*Umt[μ,ν]*k_[α] + 2*k[μ]*Umt[μ,α]
        #                   - hi[μ,ν]*Up[μ,ν]*ℓ_[α] + 2*ℓ[μ]*Up[μ,α] 
        #                   + hi[μ,ν]*U0[μ,ν]*Θ_[α] - 2*Θ[μ]*U0[μ,α] 
        #                   + 2*∂ψ[α] + 2*Hxy[α] + 2*Cf[α])
        #     return [Bt_.data...]
        # end

        # sol = nlsolve(f,[A_.data...])

        # Asol = sol.zero

        # Af_ = @Vec [Asol[1],Asol[2],Asol[3]]

        #if (y==50 && iter==3) display(Af_-A_) end

        #Ch_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ] - Hxy[μ] - ∂ψ[μ]

        # A_ = @einsum (-hi[μ,ν]*Umh[μ,ν]*k_[α] + 2*k[μ]*Umh[μ,α]
        #               - hi[μ,ν]*Up[μ,ν]*ℓ_[α] + 2*ℓ[μ]*Up[μ,α] 
        #               + hi[μ,ν]*U0[μ,ν]*Θ_[α] - 2*Θ[μ]*U0[μ,α] 
        #               + 2*∂ψ[α] + 2*Hxy[α] + 2*Cf[α])
        # # index down

        #Umh += @einsum StateTensor{Type} (Q3[α,μ,ν]*Af_[α])

        # Cf(c1,c2,c3) = 

        # find_zero

        # Umh += c1*symmetric(@einsum ℓ_[μ]*ℓ_[ν]) 

        # Umh += c2*symmetric(@einsum ℓ_[μ]*Θ_[ν]) 

        # Umh += c3*symmetric(@einsum Θ_[μ]*Θ_[ν]) 

        #Umh = @einsum StateTensor{Type} (Q4[μ,ν,α,β]*Umb2[α,β] + O[μ,ν,α,β]*Um2h[α,β] + G[μ,ν,α,β]*Um2h[α,β])

        #Uψmb = 0.#-(@einsum ℓ[μ]*A_[μ])/4 #- (@einsum Θ[μ]*Θ[ν]*Um2h[μ,ν])/4
        #Uψmb = (cp/cm)*Uψp + 2*(βdotθ/cm)*Uψ0

        #Uψmb = Umh[2,2]/(2*exp(2*ψ))

        # Um = exp(2*ψ)*Umh + 2*g*Uψmb
        # Up = exp(2*ψ)*Up  + 2*g*Uψp
        # U0 = exp(2*ψ)*U0  + 2*g*Uψ0

        Umb = symmetric(Umbh)

        Pb  = -(Up + Umb)/sqrt(2)
        # dxb = ∂ρg
        # dyb = ∂zg
        dxb = Θ_[2]*U0 - k_[2]*Umb - ℓ_[2]*Up
        dyb = Θ_[3]*U0 - k_[3]*Umb - ℓ_[3]*Up 
        

        #∂tg = (Umb - ℓ[2]*dxb - ℓ[3]*dyb)/ℓ[1]
        #∂tg = (Umb - ℓ[2]*∂ρg - ℓ[3]*∂zg)/ℓ[1]

        ∂ρχ = 2*ψr   - dr[2,2]/g[2,2]/2
        ∂zχ = 2*ψθ   - dθ[2,2]/g[2,2]/2
        ∂tχ = 2*∂tψ  - ∂tg[2,2]/g[2,2]/2

        ∂χ = @Vec [∂tχ,∂ρχ,∂zχ]

        Uχp = @einsum k[α]*∂χ[α]
        Uχ0 = @einsum Θ[α]*∂χ[α]

        #Uχmb = (cp/cm)*Uχp + 2*(βdotθ/cm)*Uχ0
        Uχmb = (cp/cm)*Uχp + 2*(βdotθ/cm)*Uχ0
        #Uχmb = 0. #(Θ_[2]*Uχ0 - ℓ_[2]*Uχp)/k_[2]

        Uψmb = Umb[2,2]/g[2,2]/4 + Uχmb/2

        #Uψmb = (cp/cm)*Uψp + 2*(βdotθ/cm)*Uψ0

        Ψb  = -(Uψp + Uψmb)/sqrt(2)
        ψxb = Θ_[2]*Uψ0 - k_[2]*Uψmb - ℓ_[2]*Uψp
        ψyb = Θ_[3]*Uψ0 - k_[3]*Uψmb - ℓ_[3]*Uψp 

        #∂tψ = (Uψmb - ℓ[2]*ψxb - ℓ[3]*ψyb)/ℓ[1]
        #∂tψ = (Uψmb - ℓ[2]*∂ρψ - ℓ[3]*∂zψ)/ℓ[1]

        ##########################################################################

        # ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])

        # ∂tβ = α*(@einsum γi[α,μ]*n[ν]*∂tg[μ,ν]) # result is a 3-vector

        # ∂t∂tg = (βr*∂tdr + βθ*∂tdθ - α*∂tP) + (∂tβ[2]*dr + ∂tβ[3]*dθ - ∂tα*P)

        # ∂t∂g = Symmetric3rdOrderTensor{Type}(
        #     (σ,μ,ν) -> (σ==1 ? ∂t∂tg[μ,ν] : σ==2 ? ∂tdr[μ,ν] : σ==3 ? ∂tdθ[μ,ν] : @assert false))

        # ∂tΓ  = Symmetric3rdOrderTensor{Type}(
        #     (σ,μ,ν) -> 0.5*(∂t∂g[ν,μ,σ] + ∂t∂g[μ,ν,σ] - ∂t∂g[σ,μ,ν])
        #     )   

        # ∂tH = Vec{3}((∂Hxy[1,:]...))
        # ∂xH = Vec{3}((∂Hxy[2,:]...))
        # ∂zH = Vec{3}((∂Hxy[3,:]...))

        # ∂tC = (@einsum gi[ϵ,σ]*∂tΓ[λ,ϵ,σ] - gi[μ,ϵ]*gi[ν,σ]*Γ[λ,μ,ν]*∂tg[ϵ,σ]) - ∂tH

        # # set up finite differencing for the constraints, by defining a function
        # # that calculates the constraints for any x and y index. This
        # # might not be the best idea, but should work.

        # dxC = DρC(constraints,U,r,θ,ns,_ds,x,y) - ∂xH 
        # dyC = DzC(constraints,U,r,θ,ns,_ds,x,y) - ∂zH 

        # ∂C = ThreeTensor{Type}(
        #     (σ,ν) ->  (σ==1 ? ∂tC[ν] : σ==2 ? dxC[ν] : σ==3 ? dyC[ν] : @assert false)
        #     )

        # UpC = @einsum k[α]*∂C[α,μ]
        # U0C = @einsum Θ[α]*∂C[α,μ]

        # UmbC = @Vec [0.,0.,0.] #(U0C).^(2)./UpC

        # ∂tCb = zeroST(ρ)# Θ_[1]*U0C - k_[1]*UmbC - ℓ_[1]*UpC

        #∂tCb = -γ0*C_

        ε = 2*abs(cm)*_ds[1]

        ∂tP  += ε*(Pb - P)
        ∂tdr += ε*(dxb - dr)
        ∂tdθ += ε*(dyb - dθ)
    
        ∂tΨ  += ε*(Ψb - Ψ)
        ∂tψr += ε*(ψxb - ψr)
        ∂tψθ += ε*(ψyb - ψθ)

        # ∂tg = ∂tg - (s[2]*Cr + s[3]*Cθ)

        # if iter == 1
        #     C1t = Cy
        #     Cwy = C1t + dt*∂tCb
        # elseif iter == 2
        #     C1t = C1[b,y]
        #     C2t = Cy
        #     Cwy = (3/4)*C1t + (1/4)*C2t + (1/4)*dt*∂tCb
        # elseif iter == 3
        #     C1t = C1[b,y]
        #     C2t = Cy
        #     Cwy = (1/3)*C1t + (2/3)*C2t + (2/3)*dt*∂tCb
        # end

        #Cw[b,y] = Cwy 

        # βv = @Vec [0.,βr,βθ] 

        # if (iter == 1 && (y == 10) && c2) println(-(@einsum s_[α]*βv[α])) end 

    end

    ∂tψv = @Vec [∂tψ,∂tψr,∂tψθ,∂tΨ]

    ∂tU = StateVector{Type}(ρ,∂tψv,∂tg,∂tdr,∂tdθ,∂tP)

    Dis = Dissipation(U,r,θ,x,y,ns)
    Dis = StateVector{Type}(ρ,Dis.ψ,Dis.g,Dis.dr,Dis.dθ,Dis.P)

    ∂tU += Dis

    ##########################################################

    # if ρ == 0.
    #     mask1 = StateTensor{Type}((1.,0.,1.,1.,0.,1.))
    #     mask2 = StateTensor{Type}((0.,1.,0.,0.,1.,0.))
    #     mask3 = @Vec [1.,0.,1.,1.]
    # else
    #     mask1 = StateTensor{Type}((1.,1/ρ,1.,1.,1/ρ,1.))
    #     mask2 = StateTensor{Type}((1/ρ,1.,1/ρ,1/ρ,1.,1/ρ))
    #     mask3 = @Vec [1.,1/ρ,1.,1.]
    # end

    # ∂tU = StateVector{Type}(ρ,∂tψv,mask1.*∂tg,mask2.*∂tdr,mask1.*∂tdθ,mask1.*∂tP) #mask2.*mask3.*

    #∂tU = StateVector{Type}(ρ,∂tψv,∂tg,∂tdr,∂tdθ,∂tP) #mask2.*mask3.*

    #########################################################

    # if iter == 1
    #     U1t = unpack(Uxy)
    #     Uwxy = U1t + dt*∂tU
    # elseif iter == 2
    #     U1t = unpack(U1[x,y])
    #     U2t = unpack(Uxy)
    #     Uwxy = (3/4)*U1t + (1/4)*U2t + (1/4)*dt*∂tU
    # elseif iter == 3
    #     U1t = unpack(U1[x,y])
    #     U2t = unpack(Uxy)
    #     Uwxy = (1/3)*U1t + (2/3)*U2t + (2/3)*dt*∂tU
    # end

    if iter == 1
        U1t = Uxy
        Uwxy = U1t + dt*∂tU
    elseif iter == 2
        U1t = U1[x,y]
        U2t = Uxy
        Uwxy = (3/4)*U1t + (1/4)*U2t + (1/4)*dt*∂tU
    elseif iter == 3
        U1t = U1[x,y]
        U2t = Uxy
        Uwxy = (1/3)*U1t + (2/3)*U2t + (2/3)*dt*∂tU
    end

    #Uw[x,y] = pack(Uwxy)

    Uw[x,y] = Uwxy

    return
    
end

@inline function chi(U,spherical=false,r=0.,θ=0.)

    ψ,ψρ,ψz,Ψ,g,dρ,dz,P = unpack(U,false)

    # gi = inv(g)

    # # Calculate lapse and shift

    # α  = 1/sqrt(-gi[1,1])
    # βρ = -gi[1,2]/gi[1,1]
    # βz = -gi[1,3]/gi[1,1]

    # Time derivatives of the metric
    #∂tg = βρ*dρ + βz*dz - α*P
    #∂tψ = βρ*ψρ + βz*ψz - α*Ψ

    χ   = 2*ψ   - log(g[2,2])/2
    ∂ρχ = 2*ψρ  - dρ[2,2]/g[2,2]/2
    ∂zχ = 2*ψz  - dz[2,2]/g[2,2]/2
    ∂nχ = 2*Ψ   -  P[2,2]/g[2,2]/2

    if spherical
        ∂rχ = ∂ρχ*sin(θ) + ∂zχ*cos(θ)
        ∂θχ = r*(∂ρχ*cos(θ) - ∂zχ*sin(θ))

        χv = @Vec [χ, ∂rχ, ∂θχ, ∂nχ]
    else
        χv = @Vec [χ, ∂ρχ, ∂zχ, ∂nχ]
    end

    #∂nχ = -(∂tχ - βρ*∂ρχ - βz*∂zχ)/α

    
    return χv

end

@parallel_indices (x) function regularization!(U,ns)
    
    Ux = U[x,:]

    #ρx = Ux.ρ

    N = 2*(ns[2]-1)

    # f(z) = imag(z)im # take imaginary part
    # g(z) = real(z)   # take real part
    if false
        for i in 5:28
            if i in [5,7,8,10,12,15,17,19,20,22,23,25,26,28]
                a = getindex.(Ux,i)
                append!(a,reverse(a)[2:end-1])
                af = real(rfft(a))
                # if (i == 8)   global gρρ0  = (af[1] + 2sum(af[2:end]))/N end
                # if (i == 20)  global dzρρ0 = (af[1] + 2sum(af[2:end]))/N end
                # if (i == 26)  global Pρρ0  = (af[1] + 2sum(af[2:end]))/N end
                a .= irfft(af,N)
            else
                a = getindex.(Ux,i)
                append!(a,-reverse(a)[2:end-1])
                af = imag(rfft(a))im
                #if (i == 14) dρρρ0 = af[1] - sum[af] end
                a .= irfft(af,N)
            end

            for y in 1:ns[2]
                Ux[y] = changevalue(Ux[y],a[y],i)
            end

        end
    end

    # instead form chi variable that is O(ρ)
    # ψ   = getindex.(Ux,1)
    gρρ   = getindex.(Ux,8)
    dρρρ  = getindex.(Ux,14)
    dzρρ  = getindex.(Ux,20)
    Pρρ   = getindex.(Ux,26)
    #∂tgρρ = similar(Pρρ)

    # for y in 1:ns[2]
    #     Uxy = Ux[y]
    #     g = Uxy.g
    #     gi = inv(g)
    #     α  = 1/sqrt(-gi[1,1])
    #     βρ = -gi[1,2]/gi[1,1]
    #     βz = -gi[1,3]/gi[1,1]
    #     ∂tgρρ[y] = βρ*dρρρ[y] + βz*dzρρ[y] - α*Pρρ[y]
    # end

    χv = [chi(Ux[y]) for y in 1:ns[2]]
    # χv[1] = @Vec [0.,χv[2][2],0.,0.]
    # χv[ns[2]] = @Vec [0.,χv[ns[2]-1][2],0.,0.]

    χ = getindex.(χv,1)
    append!(χ,reverse(χ)[2:end-1])
    χf = real(fft(χ))
    χf[1] = -(sum(χf[3:2:end]))
    χf[2] = -(sum(χf[4:2:end])) 
    χ .= real(ifft(χf))
    ψ = @. log(gρρ)/4 + χ[1:ns[2]]/2

    for y in 1:ns[2]
        Ux[y] = changevalue(Ux[y],ψ[y],1)
    end

    # χ = getindex.(χv,2)
    # append!(χ,-reverse(χ)[2:end-1])
    # χ .= real(ifft(imag(fft(χ))im))
    # ψ = @. dρρρ/gρρ/4 + χ[1:ns[2]]/2

    # for y in 1:ns[2]
    #     Ux[y] = changevalue(Ux[y],ψ[y],2)
    # end

    χ = getindex.(χv,3)
    append!(χ,reverse(χ)[2:end-1])
    χf = real(fft(χ))
    χf[1] = -(sum(χf[3:2:end]))
    χf[2] = -(sum(χf[4:2:end])) 
    χ .= real(ifft(χf))
    ψ = @. dzρρ/gρρ/4 + χ[1:ns[2]]/2

    for y in 1:ns[2]
        Ux[y] = changevalue(Ux[y],ψ[y],3)
    end


    χ = getindex.(χv,4)
    append!(χ,reverse(χ)[2:end-1])
    χf = real(fft(χ))
    χf[1] = -(sum(χf[3:2:end]))
    χf[2] = -(sum(χf[4:2:end])) 
    χ .= real(ifft(χf))
    ψ = @. Pρρ/gρρ/4 + χ[1:ns[2]]/2

    for y in 1:ns[2]
        Ux[y] = changevalue(Ux[y],ψ[y],4)
    end

    # χ = getindex.(Ux.P,1,2)
    # append!(χ,reverse(χ)[2:end-1])
    # χf = real(fft(χ))
    # χf[1] = -(sum(χf[3:2:end]))
    # χf[2] = -(sum(χf[4:2:end])) 
    # χ .= real(ifft(χf))
    # ψ = @. log(gρρ)/4 + χ[1:ns[2]]/2

    # for y in 1:ns[2]
    #     Ux[y] = changevalue(Ux[y],ψ[y],1)
    # end

    U[x,:] .= Ux

    return 

end

function RK4!(S,U1,U2,U3,U_init,C1,C2,C3,H,∂H,r,θ,t,ns,dt,_ds)

    nr,nθ = ns

    #bulk = (1:nr,1:nθ)
    bulk = (1:nr,1:nθ)
    rslice = (1:nr)

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

    @parallel bulk rhs!(S,U1,U2,U3,U_init,C1,C2,C3,H,∂H,r,θ,t,ns,dt,_ds,1) 

    @parallel (1:nr) regularization!(U2, ns)
 
    # Second stage (iter=2)

    @parallel bulk rhs!(S,U1,U2,U3,U_init,C1,C2,C3,H,∂H,r,θ,t,ns,dt,_ds,2) 

    @parallel (1:nr) regularization!(U3, ns)

    # Third stage (iter=3)

    @parallel bulk rhs!(S,U1,U2,U3,U_init,C1,C2,C3,H,∂H,r,θ,t,ns,dt,_ds,3) 

    @parallel (1:nr) regularization!(U1, ns)

    return

end

@inline function P_init(g_init::Function,∂g_init::Function,r,θ,x,y)

    g   = StateTensor((μ,ν)->  g_init(r[x,y],θ[x,y]  ,μ,ν))
    ∂tg = StateTensor((μ,ν)-> ∂g_init(r[x,y],θ[x,y],1,μ,ν))
    ∂rg = StateTensor((μ,ν)-> ∂g_init(r[x,y],θ[x,y],2,μ,ν))
    ∂θg = StateTensor((μ,ν)-> ∂g_init(r[x,y],θ[x,y],3,μ,ν))

    gi = inverse(g)

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

    gi = inverse(g)

    α = 1/sqrt(-gi[1,1])

    βr = -gi[1,2]/gi[1,1]
    βθ = -gi[1,3]/gi[1,1]

    Ψ = -(∂tψ - βr*∂ρψ - βθ*∂zψ)/α

    return StateScalar((ψ,∂ρψ,∂zψ,Ψ))
    
end

function sample!(f, ψ, g, ∂g, ns, r, θ, T)

    for x in 1:ns[1], y in 1:ns[2]
        sv = StateVector{T}( (y == 1 || y == ns[2]) ? 0. : r[x,y]*sin(θ[x,y]),
        ψ_init(ψ,g,r[x,y],θ[x,y]),
        StateTensor{T}((μ,ν) ->  g(r[x,y],θ[x,y]  ,μ,ν)),
        StateTensor{T}((μ,ν) -> ∂g(r[x,y],θ[x,y],2,μ,ν)),
        StateTensor{T}((μ,ν) -> ∂g(r[x,y],θ[x,y],3,μ,ν)),
        P_init(g,∂g,r,θ,x,y)
        )
        f[x,y] = pack(sv)
    end

end

##################################################
function main()
    # Physics


    a = SReal{0}(2.,1.)

    b = SReal{1}(2.,0.)

    c = SReal{0}(2.,2.)

    A = EvenTensor(a,b,c,a,b,a)

    return A[1,1]

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

    nr, nθ    = scale*100, scale*100 +1
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

    rM  = @zeros(nr,nθ)
    θM  = @zeros(nr,nθ)

    rM .= Data.Array([rmin + dr*(i-1) for i in 1:nr, j in 1:nθ])
    θM .= Data.Array([θmin + dθ*(j-1) for i in 1:nr, j in 1:nθ]) # include cells on the axis

    #θ .= Data.Array([dθ/2 + dθ*(j-1) for i in 1:nr, j in 1:nθ])

    U1  = StructArray{StateVector{T}}(undef,nr,nθ)
    U2  = StructArray{StateVector{T}}(undef,nr,nθ)
    U3  = StructArray{StateVector{T}}(undef,nr,nθ)

    C1  = StructArray{StateVector{T}}(undef,2,nθ)
    C2  = StructArray{StateVector{T}}(undef,2,nθ)
    C3  = StructArray{StateVector{T}}(undef,2,nθ)

    U_init  = StructArray{StateVector{T}}(undef,nr,nθ)

    H  = StructArray{Tensor{Tuple{3}, T, 1, 3}}(undef,nr,nθ)
    ∂H = StructArray{ThreeTensor{T}}(undef,nr,nθ)

    # Define initial conditions

    M = 0.1
    sign = 1.
     
    # @inline g_init(r,θ,μ,ν) =  (r^2*sin(θ)^2)*(( -(1 - 2*M/r) , 2*M/r  , 0.  ),
    #                                            (  sign*2*M/r  , (1 + 2*M/r) , 0.  ),
    #                                            (      0.      ,      0.     , r^2 ))[μ][ν]

    @inline g_init(r,θ,μ,ν) =  (( -(1 - 2*M/r)  ,     2*M*sin(θ)/r    ,    2*M*cos(θ)/r     ),
                                (  2*M*sin(θ)/r , 1 + M*(1-cos(2θ))/r ,     M*sin(2θ)/r     ),
                                (  2*M*cos(θ)/r ,      M*sin(2θ)/r    , 1 + M*(1+cos(2θ))/r ))[μ][ν]

    # @inline g_init(r,θ,μ,ν) =  (( -(1 - 2*M/r)  ,     2*M/r^2       ,    2*M*cos(θ)/r       ),
    #                             (  2*M/r^2 , 1 + M*(1-cos(2θ))/r    ,    2*M*cos(θ)/r^2     ),
    #                             (  2*M*cos(θ)/r ,   2*M*cos(θ)/r^2  , 1 + M*(1+cos(2θ))/r   ))[μ][ν]


    @inline ∂tg_init(r,θ,μ,ν) =  ((  0. ,  0.  ,  0.  ),
                                  (  0. ,  0.  ,  0.  ),
                                  (  0. ,  0.  ,  0.  ))[μ][ν]

    @inline ∂rg(r,θ,μ,ν) = ForwardDiff.derivative(r -> g_init(r,θ,μ,ν), r)
    @inline ∂θg(r,θ,μ,ν) = ForwardDiff.derivative(θ -> g_init(r,θ,μ,ν), θ)

    @inline ∂ρg(r,θ,μ,ν) = ∂rg(r,θ,μ,ν)*sin(θ) + ∂θg(r,θ,μ,ν)*cos(θ)/r
    @inline ∂zg(r,θ,μ,ν) = ∂rg(r,θ,μ,ν)*cos(θ) - ∂θg(r,θ,μ,ν)*sin(θ)/r
    
    # Dρ2T(f::Function,U,r,θ,ns,_ds,x,y,p=1) 
    
    # @inline function Dz2T(f::Function,U,r,θ,ns,_ds,x,y,p=1)
    #     Dr2T(f,U,ns,x,y)*_ds[1]*cos(θ) - Dθ2T(f,U,ns,x,y,p)*_ds[2]*sin(θ)/r
    # end
    
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

    #sample!(U1, ψ_init, g_init, ∂g_init, ns, r, θ, T)

    for i in 1:ns[1], j in 1:ns[2]
        H[i,j]  = @Vec [fH_(rM[i,j],θM[i,j],μ) for μ in 1:3]
        ∂H[i,j] = ThreeTensor{T}((μ,ν) -> f∂H_(rM[i,j],θM[i,j],μ,ν))
    end

    # for j in 1:ns[2]
    #     C1[1,j]  = @Vec [0.,0.,0.]
    #     C1[2,j]  = @Vec [0.,0.,0.]
    # end

    for x in 1:ns[1], y in 1:ns[2]
        ρ = rM[x,y]*sin(θM[x,y])
        sv = StateVector{T}( (y == 1 || y == ns[2]) ? 0. : ρ,
        (@Vec [0.,0.,0.,0.]),
        StateTensor{T}((μ,ν) -> g_init(rM[x,y],θM[x,y],μ,ν)),
        StateTensor{T}((μ,ν) -> 0.),
        StateTensor{T}((μ,ν) -> 0.),
        StateTensor{T}((μ,ν) -> 0.)
        )
        U1[x,y] = pack(sv)
    end

    constraint_init = false

    for x in 1:ns[1], y in 1:ns[2]

        r = rM[x,y]; θ = θM[x,y]

        g = unpack(U1[x,y]).g

        Hxy = H[x,y]
    
        gi = inverse(g)
    
        α = 1/sqrt(-gi[1,1])
    
        βρ = -gi[1,2]/gi[1,1]
        βz = -gi[1,3]/gi[1,1]

        nt = 1.0/α; nρ = -βρ/α; nz = -βz/α; 
    
        n = @Vec [nt,nρ,nz]
    
        n_ = @Vec [-α,0.0,0.0]

        δ = one(ThreeTensor)

        γi = gi + symmetric(@einsum n[μ]*n[ν])

        γm = δ + (@einsum n_[μ]*n[ν])

        γ = g + symmetric(@einsum n_[μ]*n_[ν])

        if constraint_init
            dρ = Dρ2T(fg,U1,r,θ,ns,_ds,x,y)
            dz = Dz2T(fg,U1,r,θ,ns,_ds,x,y)
        else
            dρ = StateTensor{T}((μ,ν) -> ∂ρg(r,θ,μ,ν))
            dz = StateTensor{T}((μ,ν) -> ∂zg(r,θ,μ,ν))
        end

        ∂g = Symmetric3rdOrderTensor{T}(
            (σ,μ,ν) -> (σ==1 ? 0. : σ==2 ? dρ[μ,ν] : σ==3 ? dz[μ,ν] : 0.)
            )

        # C = FourthOrderTensor{3,T}(
        #     (μ,ν,α,β)-> n_[μ]*n_[ν]*n[α]*n[β] - 2*n_[μ]*γm[ν,α]*n[β] + n_[μ]*n_[ν]*γi[α,β]
        #     )

        G = FourthOrderTensor{3,T}(
            (μ,ν,α,β)-> γm[μ,α]*γm[ν,β] - n_[μ]*n_[ν]*γi[α,β]
            )

        G = minorsymmetric(G)

        C = Symmetric3rdOrderTensor{T}(
            (α,μ,ν) -> n_[μ]*γm[ν,α]/2 + n_[ν]*γm[μ,α]/2 - n_[μ]*n_[ν]*n[α]
        )

        Cf = @Vec [0.,0.,0.]

        A_ = @einsum ( 2*γi[i,μ]*∂g[i,μ,α] - gi[μ,ν]*γm[α,i]*∂g[i,μ,ν] - 2*Hxy[α] - 2*Cf[α])
        # index down

        P2 = (βρ*dρ + βz*dz)/α

        P = symmetric(@einsum C[α,μ,ν]*A_[α] + G[μ,ν,α,β]*P2[α,β])

        if !(constraint_init)
            P = P2
        end

        ∂tg = βρ*dρ + βz*dz - α*P

        # χ   = (log(gρρ)-4*ψ)/ρ
        # ∂xχ =  ∂xgρρ/gρρ/ρ - log(gρρ)/ρ^2 - 4*ψx/ρ + 4*ψ/ρ^2
        # ∂yχ =  ∂ygρρ/gρρ/ρ  - 4*ψy/ρ
        # ∂nχ =  Pρρ/gρρ/ρ + nρ*log(gρρ)/ρ^2 - 4*Ψ/ρ - 4*nρ*ψ/ρ^2

        # ψx = (∂xgρρ/gρρ - ρ*∂xχ - χ)/4
        # ψy = (∂ygρρ/gρρ - ρ*∂yχ)/4 
        # ∂ygρρ = gρρ*(4*ψy + ρ*∂yχ)
        # Pρρ   = gρρ*(4*Ψ  + ρ*∂nχ - nρ*χ)

        #∂ygρρ = gρρ*(4*ψy + ∂ygρρ/gρρ  - 4*ψy)

        # ψz  = gi[2,2]*dz[2,2]/4
        # ∂tψ = gi[2,2]*∂tg[2,2]/4

        # ψ = 0.
        # ψρ = 0.
        # ψz = 0.
        # Ψ  = 0.
        
        # if (y == 1 || y==ns[2])

        #     χ = 0.
        #     dzχ = 0.
        #     dnχ = 0.

        # else
        #     ρ = r*sin(θ)

        #     χ   = (ψ - log(g[2,2])/4)/ρ
        #     dzχ = (ψz - dz[2,2]/g[2,2]/4)/ρ
        #     dnχ = (Ψ  -  P[2,2]/g[2,2]/4)/ρ + nρ*χ/ρ^2

        # end

        # dρχ = Dρ2(fψ,U1,r,θ,ns,_ds,x,y)

        # ψρ = Dρ2(fψ,U1,r,θ,ns,_ds,x,y)

        # Ψ = -(∂tψ - βρ*ψρ - βz*ψz)/α

        #[χ,dρχ,dzχ,dnχ]

        Uxy = StateVector{T}( (y == 1 || y == ns[2]) ? 0. : r*sin(θ),
        (@Vec [0.,0.,0.,0.]),
        StateTensor{T}((μ,ν) -> g_init(r,θ,μ,ν)),
        StateTensor{T}((μ,ν) -> dρ[μ,ν]),
        StateTensor{T}((μ,ν) -> dz[μ,ν]),
        StateTensor{T}((μ,ν) -> P[μ,ν])
        )

        #U2[x,y] = pack(Uxy)
        U2[x,y] = Uxy

        c1 = (x==1); c2 = (x==ns[1]);

        if (c1 || c2) #&& false
    
            if c1 
                b=1; p=-1
            else 
                b=2; p=1
            end

            dρ = Uxy.dr   
            dz = Uxy.dθ  
            P  = Uxy.P 

            # Form the unit normal vector to the boundary

            s = @Vec [0.0,p*sin(θ),p*cos(θ)]

            snorm = @einsum g[μ,ν]*s[μ]*s[ν]
        
            s = s/sqrt(snorm) 

            s_ = @einsum g[μ,ν]*s[ν]
        
            # Form the unit tangent to the boundary

            Θ_ = @Vec [βρ*cos(θ)-βz*sin(θ),cos(θ),-sin(θ)]
            Θnorm = @einsum gi[μ,ν]*Θ_[μ]*Θ_[ν]
            Θ_ = Θ_/sqrt(Θnorm)

            Θ = @einsum gi[μ,ν]*Θ_[ν]

            # Form ingoing and outgoing null vectors

            ℓ = @einsum (n[α] + s[α])/sqrt(2)

            # Time derivatives of the metric
            ∂tg = βρ*dρ + βz*dz - α*P

            ∂g = Symmetric3rdOrderTensor{T}(
                (σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dρ[μ,ν] : σ==3 ? dz[μ,ν] : @assert false)
                )
        
            # Chistoffel Symbols (of the first kind, i.e. all covariant indices)
            Γ  = Symmetric3rdOrderTensor{T}((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))

            Um = @einsum ℓ[α]*∂g[α,μ,ν]

            C1[b,y] = StateVector{T}( (y == 1 || y == ns[2]) ? 0. : r*sin(θ),
            (@Vec [0.,0.,0.,0.]),
            StateTensor{T}((μ,ν) -> Um[μ,ν]),
            StateTensor{T}((μ,ν) -> 0.),
            StateTensor{T}((μ,ν) -> 0.),
            StateTensor{T}((μ,ν) -> 0.)
            )
        end
    end

    U1 .= U2
    U3 .= U2
    U_init .= U2

    C2 .= C1
    C3 .= C1

    x=50; y=50;

    ρ = 2.

    a = SReal{0}(2.,1.)

    b = SReal{1}(2.,0.)

    c = SReal{0}(2.,2.)

    return Div(vr::Function,vθ::Function,U,r,θ,ns,_ds,x,y)

    # A = StateTensor(a,b,c,a,b,a)

    # B = -A

    # display(A)
    # display(B)

    #Tensorial.adj(A)/det(A)

    # N = 2(nθ-1)

    # #a1 = [U1[x,y].g[2,2] for y in 1:nθ]

    # a1 = [cos(2*pi*(i-1)/N) + sin(2*pi*(i-1)/N) for i in 1:nθ]

    # #a1 = [exp(-(i-25)^2/10^2) for i in 1:nθ]


    # #a2 = [cos(2*pi*(i-1)/N) for i in 1:nθ]

    # # p=-1

    # #a2 = [a1; reverse(a1[2:end-1])]
    # #a3 = [a1; reverse(a1[2:end-1])]

    # # #println(length(a1))

    # b1 = fft(a1)

    # # b2 = imag(fft(a2))im
    # # b3 = real(fft(a3))

    # # b4 = b2+b3



    # #b1 = real(b1)
    # #b1 = imag(b1)im

    # #b1[2] = 0.
    # b1[2] = 10. + b1[2]im
    # b1[1] = -(sum(real(b1[2:end])))
    


    # display(b1)

    # c1 = real(ifft(b1))

    # # println(c1[1]," ", c1[101])

    # return plot([a1,c1])

    # #display(A[x,y])

    # display(pack(unpack(U1[x,y]))-U1[x,y])

    # display(A[x,y].ψ)
    # display(A[x,y].g)
    # display(A[x,y].dr)
    # display(A[x,y].dθ)
    # display(A[x,y].P)
    # println(rootγ(A[x,y]))

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

    xout, yout = 90,1

    # temp = A[xout,yout].P

    μi,νi = 1,1

    nt  = 3000
    nout = 10 #round(nt/100)          # plotting frequency

    
    #return @benchmark RK4!($T,$U1,$U2,$U3,$C1,$C2,$C3,$H,$∂H,$rM,$θM,$t,$ns,$dt,$_ds)


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
    gdata = create_dataset(datafile, "g", datatype(Data.Number), dataspace(nsave,6,nr,nθ), chunk=(1,6,nr,nθ))

    coordsfile = h5open(path*"/coords.h5","cw")
    coordsfile["r"] = Array(rM)
    coordsfile["theta"]  = Array(θM)
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
            data = [constraints(U1[x,y])[2] - H[x,y][2] for x in 1:nr, y in 1:nθ]
            #data = [(chi(U1[x,y])-chi(U_init[x,y]))[3] for x in 1:nr, y in 1:nθ]
            #data = [(U1[x,y]-U_init[x,y]).g[2,2] for x in 1:nr, y in 1:nθ]

            #data = [(pack(unpackSV(A[x,y]))-A[x,y]).dr[2,2] for x in 1:nr, y in 1:nθ]
            #data = [(Dρ2T(fg,A,r[x,y],θ[x,y],ns,_ds,x,y) - unpack(A[x,y])[6])[2,2] for x in 1:nr, y in 1:nθ]
            #data = [unpack(A[x,y])[6][2,2] for x in 1:nr, y in 1:nθ]
            #data = [(Dz2T(fg,A,r[x,y],θ[x,y],ns,_ds,x,y) - A[x,y].dθ)[1,1] for x in 1:nr, y in 1:nθ]
            #data = [A[x,y].ψ[1] for x in 1:nr, y in 1:nθ]

            # r slice

            #data = [[(divergent_terms(U1,U_init,H,90,y)[1]).data...] for y in 1:nθ]#-divergent_terms(U_init,U_init,H,x,2)[1]

            # y = 30
            # data = [(Ct = (constraints(U1[x,y])[1]- H[x,y][1]);
            #         Cρ = (constraints(U1[x,y])[2]- H[x,y][2]);
            #         Cz = (constraints(U1[x,y])[3]- H[x,y][3]);
            #         @Vec [Ct, Cρ, Cz])
            #         for x in 1:nr]

            # x = 1
            # data = [(Ct = (constraints(U1[x,y])[1]- H[x,y][1]);
            #         Cρ = (constraints(U1[x,y])[2]- H[x,y][2]);
            #         Cz = (constraints(U1[x,y])[3]- H[x,y][3]);
            #         @Vec [Ct, Cρ, Cz])
            #         for y in 1:ns[2]]

            #data = [[(chi(U1[x,30],true,rM[x,30],θM[x,30])-chi(U_init[x,30],true,rM[x,30],θM[x,30])).data...] for x in 1:nr]#
            #data = [[(U1[x,30]).ψ.data...] for x in 1:nr]
            #data = [[(U1[x,30]-U_init[x,30]).g[2,2],(U1[x,30]-U_init[x,30]).dr[2,2],(U1[x,30]-U_init[x,30]).dθ[2,2],(U1[x,30]-U_init[x,30]).P[2,2]] for x in 1:nr]

            #data = [[(Dz2T(fg,U1,r[x,1],θ[x,1],ns,_ds,x,1) - unpack(U1[x,1])[7]).data...] for x in 1:nr]
            
            #data = [A[x,50].g[1,1] - U_init[x,50].g[1,1] for x in 1:nr]
            #data = [A[x,1].g[1,1] - U_init[x,1].g[1,1] for x in 1:nr]

            # θ slice
            #data = [[(Dz2T(fg,U1,rM[90,y],θM[90,y],ns,_ds,90,y) - unpack(U1[90,y]).dθ).data...] for y in 1:nθ]
            #data = [[(divergent_terms(U1,U_init,H,50,y,(y==1 || y==nθ))[1]).data...] for y in 1:nθ]
            #data = [[(U1[50,y].dθ - U_init[50,y].dθ).data...] for y in 1:nθ] # - U_init[x,10]
            #data = [[U1[50,y].dθ[1,2],U1[50,y].dθ[2,3]]/(sin(θM[50,y])) for y in 1:nθ]
            #data = [[U1[50,y].dr[1,1],U1[50,y].dr[1,3],U1[50,y].dr[2,2],U1[50,y].dr[3,3]] for y in 1:nθ]
            #data = [[(divergent_terms(U1,U_init,H,50,y,(y==1 || y==nθ))[1])[1,2],(divergent_terms(U1,U_init,H,50,y,(y==1 || y==nθ))[1])[2,3]] for y in 1:nθ]
            # data = [[(U1[100,y]).ψ.data...] for y in 1:nθ] # - U_init[x,10]
            #data = [[(U1[100,y]-U_init[100,y]).g[2,2],(U1[100,y]-U_init[100,y]).dr[2,2],(U1[100,y]-U_init[100,y]).dθ[2,2],(U1[100,y]-U_init[100,y]).P[2,2]] for y in 1:nθ] # - U_init[x,10]
            #data = [[(chi(U1[50,y])-chi(U_init[50,y])).data...] for y in 1:nθ]#

            #data = [constraints(U1[1,y]) - H[1,y] for y in 1:nθ]
            #(ψ,ψx,ψy,Ψ,g,dx,dy,P) f::Function,U,ns,x,y,p
            # data = [[
            #     (Dρ2T(fdθ,U1,rM[x,30],θM[x,30],ns,_ds,x,30)-Dz2T(fdr,U1,rM[x,30],θM[x,30],ns,_ds,x,30)).data...,
            #    ] for x in 1:nr]
            # data = [[
            #            ((Dρ2T(fg,U1,rM[x,30],θM[x,30],ns,_ds,x,30) - unpack(U1[x,30],false)[6])*sin(θM[x,30])
            #             + (Dz2T(fg,U1,rM[x,30],θM[x,30],ns,_ds,x,30) - unpack(U1[x,30],false)[7])*cos(θM[x,30])).data...,
            #            ((Dρ2(fψ,U1,rM[x,30],θM[x,30],ns,_ds,x,30) - unpack(U1[x,30],false)[2])*sin(θM[x,30])
            #            + (Dz2(fψ,U1,rM[x,30],θM[x,30],ns,_ds,x,30) - unpack(U1[x,30],false)[3])*cos(θM[x,30]))
            #           ] for x in 1:nr]
            #data = [divergent_terms(A[1,y])[1,1]/(r[1,y]*sin(θ[1,y])) for y in 1:nθ]
            #data = [A[50,y].g[2,2] - U_init[x,50].g[2,2] for x in 1:nr]

            #return typeof(Array(data))
            labels = ["Ct" "Cρ" "Cz"]
            #labels = ["Cρtt" "Cρtρ" "Cρtz" "Cρρρ" "Cρρz" "Cρzz" ]

            #data = getindex.(A[:,50].g,μi,νi) - temp_array[:,50]

            #reduce(hcat,data)'
            #p = 5
            # plot(Array(θM[1,:]), reduce(hcat,data)', label=labels, title = "Time = "*string(round(t; digits=2)) )
            # #ylims!(-5*10^-4, 5*10^-4)
            # ylims!(-0.0001, 0.0001)
            # frame(anim)

            # plot(Array(rM[:,1]), reduce(hcat,data)', label=labels, title = "Time = "*string(round(t; digits=2)) )
            # ylims!(-5*10^-5, 5*10^-5)
            # frame(anim)

            #println(t)
            # (ST,St) = divergent_terms(U1,U_init,H,xout,yout+1)
            # (STi,Sti) = divergent_terms(U_init,U_init,H,xout,yout+1)

            # display(ST-STi)

            #println(unpack(U1[90,1]).ρ)

            #display(inv(unpack(U1[90,1]).g)[2,2] - exp(-4*unpack(U1[90,1]).ψ[1]))

            #display(U1[10,10] - U_init[10,10])
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

            #2D Constraints
            heatmap(Array(rM[1:res:end,1]), Array(θM[1,1:res:end]), Array(data)', title = "Time = "*string(round(t; digits=2)),
            aspect_ratio=1, xlims=(rmin,rmax), ylims=(0,pi),clim=(-10^(-4),10^(-4)), c=:viridis); 
            frame(anim) #clim=(-10^(-4),10^(-4))

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

            ψdata[iter,:,:] = [unpack(U1[x,y]).ψ[1] for y in 1:nθ, x in 1:nr]

            gdata[iter,:,:,:] = [unpack(U1[x,y]).g.data[i] for i in 1:6, y in 1:nθ, x in 1:nr]
            
            iter += 1

        end

        append!(ints,integrate_constraints(U1,H,ns,_ds))

        RK4!(T,U1,U2,U3,U_init,C1,C2,C3,H,∂H,rM,θM,t,ns,dt,_ds)

        t += dt

    end

    catch error
        close(datafile)
        throw(error)
    end

    close(datafile)

    gif(anim, "tests.gif", fps = 30)

    return plot(ints, yaxis=:log, ylim = (10^-10, 10^1))

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