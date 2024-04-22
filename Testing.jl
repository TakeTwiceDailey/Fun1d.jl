module Testing

using Tensorial
using BenchmarkTools
using Profile

# Scaled real numbers
# Allows for the multiplication of factors of ρ
# during the evolution, without actually multiplying
# so that divisions can happen later-on even at ρ = 0

# struct SReal{N,T<:Real} <: Real
#     ρ::T
#     x::T
#     function SReal{N,T}(ρ::T, x::T) where {N,T}
#         @assert N >= 0
#         @assert (ρ >= zero(T) || ρ==NaN)
#         new{N,T}(ρ,x)
#     end
# end

# # # Base.zero
# # # Base.one 

# @inline SReal{N}(ρ::T, x::T) where {N,T} = SReal{N,T}(ρ,x)

# @inline Base.convert(::Type{T}, A::SReal{N,T}) where {N,T} = A.ρ^N*A.x

# #@inline Base.convert(::Type{SReal{N,T}}, A::T, ρ::T) where {N,T<:Real} = SReal{N,T}(ρ,x)

# #@inline Base.zero(::SReal{N,T}) where {N,T} = SReal{Missing,T}(NaN,0.)

# @inline function Base.show(io::IO, ::MIME"text/plain", A::SReal{N,T}) where {N,T}
#     print(io, [N,A.x])
# end

# @inline function Base.:+(A::SReal{M,T}, B::SReal{N,T}) where {M,N,T} 
#     @assert A.ρ == B.ρ
#     if N==M
#         SReal{M,T}(A.ρ, A.x + B.x) 
#     elseif N>M
#         SReal{M,T}(A.ρ, A.x + B.x*(B.ρ)^(N-M)) 
#     else
#         SReal{N,T}(A.ρ, A.x*(A.ρ)^(M-N) + B.x)
#     end
# end

# @inline Base.:-(A::SReal{N,T}) where {N,T} = SReal{N,T}(A.ρ,-A.x)

# @inline Base.:-(A::SReal{M,T}, B::SReal{N,T}) where {M,N,T} = A + (-B)

# @inline function Base.:*(A::SReal{N,T}, B::SReal{M,T}) where {N,M,T} 
#     @assert A.ρ == B.ρ 
#     SReal{N+M,T}(A.ρ, A.x * B.x)
# end

# @inline function Base.:/(A::SReal{N,T}, B::SReal{M,T}) where {N,M,T} 
#     @assert A.ρ == B.ρ 
#     SReal{N-M,T}(A.ρ, A.x / B.x)
# end

# @inline Base.:*(a::T, A::SReal{N,T}) where {N,T} = SReal{N,T}(A.ρ, a*A.x )

# @inline Base.:*(A::SReal{N,T}, a::T) where {N,T} = a*A

# @inline Base.:/(A::SReal{N,T}, y::T) where {N,T} = SReal{N,T}(A.x / y)

# @inline Base.:*(a::Int, A::SReal{N,T}) where {N,T} = SReal{N,T}(A.ρ, a*A.x )

# @inline Base.:*(A::SReal{N,T}, a::Int) where {N,T} = a*A

# @inline Base.:/(A::SReal{N,T}, y::Int) where {N,T} = SReal{N,T}(A.x / y)

# @inline Base.sqrt(A::SReal{0,T}) where T = SReal{0,T}(A.ρ,sqrt(A.x))


struct SReal{T<:Real} <: Real
    odd::Bool
    ρ::T
    x::T
    # function SReal{T}(ρ::T, x::T) where {T}
    #     new{N,T}(ρ,x)
    # end
end

# Base.zero
# Base.one 

#@inline SReal(odd::Bool,ρ::T, x::T) where T = SReal{T}(odd,ρ,x)

@inline Base.convert(::Type{T}, A::SReal{T}) where T = odd ? A.ρ*A.x : A.x

#@inline Base.convert(::Type{SReal{N,T}}, A::T, ρ::T) where {N,T<:Real} = SReal{N,T}(ρ,x)

#@inline Base.zero(::SReal{N,T}) where {N,T} = SReal{Missing,T}(NaN,0.)

@inline function Base.show(io::IO, ::MIME"text/plain", A::SReal{T}) where {T}
    print(io, A.x)
end

@inline function Base.:+(A::SReal{T}, B::SReal{T}) where {T} 
    # @assert A.ρ == B.ρ
    # @assert A.odd == B.odd
    SReal{T}(A.odd, A.ρ, A.x + B.x) 
    # if A.odd && B.odd
    #     SReal{T}(A.n, A.ρ, A.x + B.x) 
    # elseif A.n > B.n
    #     SReal{T}(B.n, A.ρ, A.x + B.x*(B.ρ)^(A.n-B.n)) 
    # else
    #     SReal{T}(A.n, A.ρ, A.x*(A.ρ)^(B.n-A.n) + B.x)
    # end
end

@inline Base.:-(A::SReal{T}) where T = SReal{T}(A.odd,A.ρ,-A.x)

@inline Base.:-(A::SReal{T}, B::SReal{T}) where {T} = A + (-B)

@inline function Base.:*(A::SReal{T}, B::SReal{T}) where {T} 
    #@assert A.ρ == B.ρ 
    ρ = A.ρ::T
    a = A.x*B.x
    if A.odd*B.odd; a*=2. end
    SReal{T}(A.odd ⊻ B.odd, ρ, a)

    # if A.odd
    #     if B.odd
    #         SReal{T}(false, A.ρ, (A.ρ)^2*(A.x * B.x))
    #     else
    #         SReal{T}(true , A.ρ,   A.x * B.x)
    #     end
    # else
    #     if B.odd
    #         SReal{T}(true , A.ρ,   A.x * B.x)
    #     else
    #         SReal{T}(false, A.ρ,   A.x*B.x)
    #     end
    # end
end

# @inline function Base.:/(A::SReal{T}, B::SReal{T}) where {T} 
#     @assert A.ρ == B.ρ 
#     if A.odd
#         if B.odd
#             SReal{T}(false, A.ρ, (A.x/B.x))
#         else
#             SReal{T}(true , A.ρ,   A.x/B.x)
#         end
#     else
#         if B.odd
#             @assert false
#         else
#             SReal{T}(false, A.ρ,   A.x/B.x)
#         end
#     end
# end

@inline Base.:*(a::T, A::SReal{T}) where {T} = SReal{T}(A.odd, A.ρ, a*A.x )

@inline Base.:*(A::SReal{T}, a::T) where {T} = a*A

@inline Base.:/(A::SReal{T}, y::T) where {T} = SReal{T}(A.odd, A.ρ, A.x / y)

#@inline Base.sqrt(A::SReal{T}) where T = SReal{T}(A.ρ,sqrt(A.x))

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

# @inline function Base.getindex(A::StateTensor{T},i,j) where T
#     field = ((:tt,:tρ,:tz),(:tρ,:ρρ,:ρz),(:tz,:ρz,:zz))[i][j]
#     return getproperty(A,field)
# end

# @inline function Base.getindex(A::StateTensor{T},i) where T
#     field = (:tt,:tρ,:tz,:ρρ,:ρz,:zz)[i]
#     return getproperty(A,field)
# end

# struct EvenTensor{T}
#     tt::SReal{0,T}
#     tρ::SReal{1,T}
#     tz::SReal{0,T}
#     ρρ::SReal{0,T}
#     ρz::SReal{1,T}
#     zz::SReal{0,T}
# end

# struct OddTensor{T}
#     tt::SReal{1,T}
#     tρ::SReal{0,T}
#     tz::SReal{1,T}
#     ρρ::SReal{1,T}
#     ρz::SReal{0,T}
#     zz::SReal{1,T}
# end

# const StateTensor{T} = Union{EvenTensor{T},OddTensor{T}}

# @inline function Base.getindex(A::StateTensor{T},i,j) where T
#     field = ((:tt,:tρ,:tz),(:tρ,:ρρ,:ρz),(:tz,:ρz,:zz))[i][j]
#     return getproperty(A,field)
# end

# @inline function Base.getindex(A::StateTensor{T},i) where T
#     field = (:tt,:tρ,:tz,:ρρ,:ρz,:zz)[i]
#     return getproperty(A,field)
# end

function main()

    a = SReal(false,1.,1.)

    b = SReal(true,1.,0.)

    c = SReal(false,1.,2.)

    A = SymmetricSecondOrderTensor{3,SReal{Float64},6}((a,b,c,a,b,a))

    B = SymmetricSecondOrderTensor{3,Float64,6}((1.,2.,3.,4.,5.,6.))

    # @profile (@einsum A[i,j]*A[i,j])

    # @profile (@einsum A[i,j]*A[i,j])

    return @benchmark @einsum $A[i,j]*$A[j,k]*$A[k,l]*$A[l,m]

    return @benchmark @einsum $B[i,j]*$B[j,k]*$B[k,l]*$B[l,m]  # 7.1 ns

    # sum = SReal{0}(1.,0.)
    # return @benchmark ( 
    #     for i in 1:3, j in 1:3
    #         $sum += $A[i,j]*$A[i,j]
    #     end 
    #     )

end

end