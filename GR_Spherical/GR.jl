module GR_Spherical

using DifferentialEquations
using Fun1d
using DataFrames
using CSV
using Plots
using Roots

# using Random

struct GBSSN_Variables{S,T} <: AbstractArray{T,2}
    α::GridFun{S,T}
    βr::GridFun{S,T}
    Br::GridFun{S,T}
    χ::GridFun{S,T}
    grr::GridFun{S,T}
    gθθ::GridFun{S,T}
    Arr::GridFun{S,T}
    K::GridFun{S,T}
    Γr::GridFun{S,T}
end

# Iteration
Base.IteratorSize(::Type{<:GBSSN_Variables}) = Iterators.HasShape{2}()
Base.eltype(::Type{GBSSN_Variables{S,T}}) where {S,T} = T
Base.isempty(x::GBSSN_Variables) = isempty(x.α)
function Base.iterate(x::GBSSN_Variables, state...)
    return iterate(Iterators.flatten((x.α, x.βr, x.Br, x.χ, x.grr, x.gθθ, x.Arr, x.K, x.Γr)), state...)
end
Base.size(x::GBSSN_Variables) = (length(x.α), 9)
Base.size(x::GBSSN_Variables, d) = size(x)[d]

# Indexing
function lin2cart(x::GBSSN_Variables, i::Number)
    n = length(x.α)
    return (i - 1) % n + 1, (i - 1) ÷ n + 1
end
Base.firstindex(x::GBSSN_Variables) = error("not implemented")
Base.getindex(x::GBSSN_Variables, i) = getindex(x, i.I...)
Base.getindex(x::GBSSN_Variables, i::Number) = getindex(x, lin2cart(x, i)...)
Base.getindex(x::GBSSN_Variables, i, j) = getindex((x.α, x.βr, x.Br, x.χ, x.grr, x.gθθ, x.Arr, x.K, x.Γr)[j], i)
Base.lastindex(x::GBSSN_Variables) = error("not implemented")
Base.setindex!(x::GBSSN_Variables, v, i) = setindex!(x, v, i.I...)
Base.setindex!(x::GBSSN_Variables, v, i::Number) = setindex!(x, v, lin2cart(x, i))
Base.setindex!(x::GBSSN_Variables, v, i, j) = setindex!((x.α, x.βr, x.Br, x.χ, x.grr, x.gθθ, x.Arr, x.K, x.Γr)[j], v, i)

# Abstract Array
Base.IndexStyle(::GBSSN_Variables) = IndexCartesian()
Base.similar(x::GBSSN_Variables) = GBSSN_Variables(map(similar, (x.α, x.βr, x.Br, x.χ, x.grr, x.gθθ, x.Arr, x.K, x.Γr))...)
function Base.similar(x::GBSSN_Variables, ::Type{T}) where {T}
    return GBSSN_Variables(map(y -> similar(y,T), (x.α, x.βr, x.Br, x.χ, x.grr, x.gθθ, x.Arr, x.K, x.Γr))...)
end
Base.similar(x::GBSSN_Variables, ::Dims) = similar(x)
Base.similar(x::GBSSN_Variables, ::Dims, ::Type{T}) where {T} = similar(x, T)

# Broadcasting
Base.BroadcastStyle(::Type{<:GBSSN_Variables}) = Broadcast.ArrayStyle{GBSSN_Variables}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GBSSN_Variables}},
                      ::Type{T}) where {T}
    x = find_GBSSN_Variables(bc)
    return similar(x, T)
end
find_GBSSN_Variables(bc::Base.Broadcast.Broadcasted) = find_GBSSN_Variables(bc.args)
find_GBSSN_Variables(args::Tuple) = find_GBSSN_Variables(find_GBSSN_Variables(args[1]), Base.tail(args))
find_GBSSN_Variables(x) = x
find_GBSSN_Variables(::Tuple{}) = nothing
find_GBSSN_Variables(a::GBSSN_Variables, rest) = a
find_GBSSN_Variables(::Any, rest) = find_GBSSN_Variables(rest)

# Others
function Base.map(fun, x::GBSSN_Variables, ys::GBSSN_Variables...)
    return GBSSN_Variables(
        map(fun, x.α, (y.α for y in ys)...),
        map(fun, x.βr, (y.βr for y in ys)...),
        map(fun, x.Br, (y.Br for y in ys)...),
        map(fun, x.χ, (y.χ for y in ys)...),
        map(fun, x.grr, (y.grr for y in ys)...),
        map(fun, x.gθθ, (y.gθθ for y in ys)...),
        map(fun, x.Arr, (y.Arr for y in ys)...),
        map(fun, x.K, (y.K for y in ys)...),
        map(fun, x.Γr, (y.Γr for y in ys)...)
        )
end

# function Base.rand(rng::AbstractRNG, ::Random.SamplerType{GBSSN_Variables{T}}) where {T}
#     return GBSSN_Variables{T}(rand(rng, T), rand(rng, T))
# end

Base.zero(::Type{<:GBSSN_Variables}) = error("not implemented")
Base.zero(x::GBSSN_Variables) = GBSSN_Variables(map(zero, (x.α, x.βr, x.Br, x.χ, x.grr, x.gθθ, x.Arr, x.K, x.Γr))...)

Base.:+(x::GBSSN_Variables) = map(+, x)
Base.:-(x::GBSSN_Variables) = map(-, x)

Base.:+(x::GBSSN_Variables, y::GBSSN_Variables) = map(+, x, y)
Base.:-(x::GBSSN_Variables, y::GBSSN_Variables) = map(-, x, y)

Base.:*(x::GBSSN_Variables, a::Number) = map(b -> b * a, x)
Base.:*(a::Number, x::GBSSN_Variables) = map(b -> a * b, x)
Base.:/(x::GBSSN_Variables, a::Number) = map(b -> b / a, x)
Base.:\(a::Number, x::GBSSN_Variables) = map(b -> a \ b, x)

################################################################################

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
"         ___\\/ / /\\\\\\\\\\\\\\\\\\  /____\\/ /\\\\\\\\_\\// //\\\\\\\\_____by Erik Schnetter_____\n",
"          ___\\/_/ / / / / /_/______\\/ /  /___\\// /  /_______and Conner Dailey____\n",
"           _____\\/_/_/_/_/__________\\/__/______\\/__/______________________________\n"
)

end

function setup(::Type{S}, rspan, points::Int) where {S}

    # Specify the radial domain and the
    # radial step amount
    rmin,rmax = rspan
    domain = Domain{S}(rmin, rmax)
    grid = Grid(domain, points)
    return grid

end

function dissipation(f::GridFun{S,T}) where {S,T}
    dx = spacing(f.grid)
    n = f.grid.ncells + 4
    dvalues = Array{T}(undef, n)

    # if f.values[2] == 0.
    #     dvalues[3:6] .= ((f.values[3] - 6*f.values[4] + 15*f.values[5]
    #     - 20*f.values[6] + 15*f.values[7] - 6*f.values[8] + f.values[9])/(dx^6))
    # else
    #     dvalues[3:6] .= ((f.values[2] - 6*f.values[3] + 15*f.values[4]
    #     - 20*f.values[5] + 15*f.values[6] - 6*f.values[7] + f.values[8])/(dx^6))
    # end
    #
    # for i in 7:(n - 5)
    #     dvalues[i] =  ((f.values[i-3] - 6*f.values[i-2] + 15*f.values[i-1]
    #     - 20*f.values[i] + 15*f.values[i+1] - 6*f.values[i+2] + f[i+3])/(dx^6))
    # end
    #
    # dvalues[(n-4):(n-2)] .= ((f.values[n-8] - 6*f.values[n-7] + 15*f.values[n-6]
    # - 20*f.values[n-5] + 15*f.values[n-4] - 6*f.values[n-3] + f.values[n-2])/(dx^6))

    # if f.values[2] == 0.
    #     for i in 3:4
    #         dvalues[i] = ((f.values[i] - 4*f.values[i+1] + 6*f.values[i+2]
    #         - 4*f.values[i+3] + f.values[i+4])/(dx^4))
    #     end
    # else
    #     dvalues[3:4] .= ((f.values[2] - 4*f.values[3] + 6*f.values[4]
    #     - 4*f.values[5] + f.values[6])/(dx^4))
    # end

    # dvalues[1:4] .= 0.
    # # dvalues[3] = ((6*f.values[1] + 15*f.values[2]
    # # - 20*f.values[3] + 15*f.values[4] - 6*f.values[5]
    # # + f.values[6])/(dx^6))
    #
    # for i in 3:3
    #     dvalues[i] = ((f.values[i-2] - 4*f.values[i-1] + 6*f.values[i]
    #     - 4*f.values[i+1] + f.values[i+2])/(dx^4))
    # end
    # for i in 4:(n - 5)
    #     dvalues[i] = ((-f.values[i-3] + 12*f.values[i-2] -39*f.values[i-1]
    #     + 56*f.values[i] - 39*f.values[i+1] + 12*f.values[i+2]
    #     - f.values[i+3])/(dx^4))
    # end
    # dvalues[(n-4):n] .= 0.

    dvalues[1:2] .= 0.
    for i in 3:(n - 4)
        dvalues[i] = ((f.values[i-2] - 4*f.values[i-1] + 6*f.values[i]
        - 4*f.values[i+1] + f.values[i+2])/(dx^4))
    end
    dvalues[(n-3):n] .= 0.

    return GridFun(f.grid, dvalues)
end


function init(::Type{T}, grid::Grid, param) where {T}

    # Presumably we need to specify initial conditions for
    # every variable here, replacing 0's with functions

    # Need preallocation here?
    # https://docs.julialang.org/en/v1/manual/functions/#man-vectorized
    # https://docs.julialang.org/en/v1/manual/performance-tips/#Pre-allocating-outputs

    r = param[4]

    # n = grid.ncells + 1
    #
    # α = GridFun(grid, Vector{T}(undef,n))
    # βr = GridFun(grid, Vector{T}(undef,n))
    # Br = GridFun(grid, Vector{T}(undef,n))
    # χ = GridFun(grid, Vector{T}(undef,n))
    # grr = GridFun(grid, Vector{T}(undef,n))
    # gθθ = GridFun(grid, Vector{T}(undef,n))
    # Arr = GridFun(grid, Vector{T}(undef,n))
    # K = GridFun(grid, Vector{T}(undef,n))
    # Γr = GridFun(grid, Vector{T}(undef,n))

    # Initial conditions for flat Minkowski space

    # α = project(T, grid, r -> 1)
    # βr = project(T, grid, r -> 0)
    # Br = project(T, grid, r -> 0)
    # χ = project(T, grid, r -> 1)
    # grr = project(T, grid, r -> 1)
    # gθθ = project(T, grid, r -> r^2)
    # Arr = project(T, grid, r -> 0)
    # K = project(T, grid, r -> 0)
    # Γr = project(T, grid, r -> -2/r)

    # Initial conditions for Schwarzschild metric (Isotropic Coordinates)

    #Mass

    M=1

    α = sample(T, grid, rt -> (1+M/(2*r(rt)))^(-2))
    #α = sample(T, grid, r -> 1)
    βr = sample(T, grid, rt -> 0)
    Br = sample(T, grid, rt -> 0)
    χ = sample(T, grid, rt -> (1+M/(2*r(rt)))^(-4))
    grr = sample(T, grid, rt -> 1)
    gθθreg = sample(T, grid, rt -> 0)
    Arr = sample(T, grid, rt -> 0)
    K = sample(T, grid, rt -> 0)
    Γreg = sample(T, grid, rt -> 1)

    # Initial conditions for FRW metric
    # k = -1
    # a0 = 1
    #
    # α = project(T, grid, r -> 1)
    # βr = project(T, grid, r -> 0)
    # Br = project(T, grid, r -> 0)
    # χ = project(T, grid, r -> a0^(-2))
    # grr = project(T, grid, r -> 1/(1-k*r^2))
    # gθθ = project(T, grid, r -> r^2)
    # Arr = project(T, grid, r -> 0)
    # K = project(T, grid, r -> 0)
    # Γr = project(T, grid, r -> (3*k*r^2-2)/(r*a0^2))

    α[1:2] .= 0.
    βr[1:2] .= 0.
    Br[1:2] .= 0.
    χ[1:2] .= 0.
    grr[1:2] .= 0.
    gθθreg[1:2] .= 0.
    Arr[1:2] .= 0.
    K[1:2] .= 0.
    Γreg[1:2] .= 0.

    return GBSSN_Variables(α, βr, Br, χ, grr, gθθreg, Arr, K, Γreg)

end

# Need global variable for gauge Type (v)

function rhs(state::GBSSN_Variables, param, t)

    # Variables

    α = state.α
    βr = state.βr
    Br = state.Br
    χ = state.χ
    grr = state.grr
    gθθreg = state.gθθ
    Arr = state.Arr
    K = state.K
    Γreg = state.Γr

    v = param[3]
    η = 0

    drt = spacing(α.grid)
    n = α.grid.ncells + 4

    # Boundary Conditions

    if v == 0 # Eulerian Condition
        βr[2] = (17*βr[3] + 9*βr[4] - 5*βr[5] + βr[6])/22
    elseif v == 1 # Lagrangian Condition
        gθθreg[2] = ((-315*gθθreg[3] + 210*gθθreg[4]
        - 126*gθθreg[5] + 45*gθθreg[6] - 7*gθθreg[7])/63)
        βr[2] = (-315*βr[3] + 210*βr[4] - 126*βr[5] + 45*βr[6] - 7*βr[7])/63
    end

    # Spatial Derivatives (finite differences) with respect to coordinate rt

    order = 4

    ∂rtα = deriv(α,order,1)
    ∂rtβr = deriv(βr,order,-1)
    ∂rtBr = deriv(Br,order,-1)
    ∂rtχ = deriv(χ,order,1)
    ∂rtgrr = deriv(grr,order,1)
    ∂rtgθθreg = deriv(gθθreg,order,1)
    ∂rtArr = deriv(Arr,order,1)
    ∂rtK = deriv(K,order,1)
    ∂rtΓreg = deriv(Γreg,order,-1)

    ∂2rtα = deriv2(α,order,1)
    ∂2rtβr = deriv2(βr,order,-1)
    ∂2rtχ = deriv2(χ,order,1)
    ∂2rtgrr = deriv2(grr,order,1)
    ∂2rtgθθreg = deriv2(gθθreg,order,1)

    # Convert derivatives from (d/drt) to (d/dr)

    r = sample(Float64, α.grid, param[4])
    drdrt = sample(Float64, α.grid, param[5])
    d2rdrt = sample(Float64, α.grid, param[6])

    ∂α = ∂rtα./drdrt
    ∂βr = ∂rtβr./drdrt
    ∂Br = ∂rtBr./drdrt
    ∂χ = ∂rtχ./drdrt
    ∂grr = ∂rtgrr./drdrt
    ∂gθθreg = ∂rtgθθreg./drdrt
    ∂Arr = ∂rtArr./drdrt
    ∂K = ∂rtK./drdrt
    ∂Γreg = ∂rtΓreg./drdrt

    ∂2α = (∂2rtα - d2rdrt.*∂α)./(drdrt.^2)
    ∂2βr = (∂2rtβr - d2rdrt.*∂βr)./(drdrt.^2)
    ∂2χ = (∂2rtχ - d2rdrt.*∂χ)./(drdrt.^2)
    ∂2grr = (∂2rtgrr - d2rdrt.*∂grr)./(drdrt.^2)
    ∂2gθθreg = (∂2rtgθθreg - d2rdrt.*∂gθθreg)./(drdrt.^2)

    r[1:2] .= 0.
    βr[1:2] .= 0.
    gθθreg[1:2] .= 0.
    Γreg[1:2] .= 0.

    gθθ = (r.^2).*(gθθreg .+ 1)
    ∂gθθ = (r.^2).*∂gθθreg + (2*r).*(gθθreg .+ 1)
    ∂2gθθ = (r.^2).*∂2gθθreg + (4*r).*∂gθθreg + 2*(gθθreg .+ 1)

    gθθ[1:2] .= 0.
    ∂gθθ[1:2] .= 0.
    ∂2gθθ[1:2] .= 0.

    Γr = -(2 ./r).*Γreg
    ∂Γr = -(2 ./r).*∂Γreg + (2 ./(r.^2)).*Γreg

    # Gauge Conditions

    #Superscript condition...1 is a plus, 0 is a minus
    a = 0

    #Subscript condition...1 is a plus, 0 is a minus
    b = 0

    # Zero condition, 1 includes shift, 0 for vanishing shift
    # if this is 0, a and b don't matter
    c = 1

    # Evolution Equations

    ∂tα = a*βr.*∂α - 2*α.*K

    ∂tβr = c*((3/4)*Br + b*βr.*∂βr)

    ∂tχ = ((2/3)*K.*α.*χ - (1/3)*v*βr.*χ.*∂grr./grr - (2/3)*v*βr.*χ.*∂gθθ./gθθ
     - (2/3)*v*χ.*∂βr + βr.*∂χ)

    ∂tgrr = (-2*Arr.*α - (1/3)*v*βr.*∂grr + βr.*∂grr
     - (2/3)*v*grr.*βr.*∂gθθ./gθθ + 2*grr.*∂βr - (2/3)*v*grr.*∂βr)

    ∂tgθθ = (Arr.*gθθ.*α./grr - (1/3)*v*gθθ.*βr.*∂grr./grr - (2/3)*v*βr.*∂gθθ
     + βr.*∂gθθ - (2/3)*v*gθθ.*∂βr)

    ∂tArr = (-2*α.*(Arr.^2)./grr + K.*α.*Arr - (1/3)*v*βr.*Arr.*∂grr./grr
     - (2/3)*v*βr.*Arr.*∂gθθ./gθθ - (2/3)*v*Arr.*∂βr + 2*Arr.*∂βr
     + (2/3)*α.*χ.*(∂grr./grr).^2 - (1/3)*α.*χ.*(∂gθθ./gθθ).^2
     - (1/6)*α.*(∂χ.^2)./χ - (2/3)*α.*χ.*grr./gθθ + βr.*∂Arr
     + (2/3)*α.*χ.*grr.*∂Γr - (1/2)*α.*χ.*(∂grr./grr).*(∂gθθ./gθθ)
     + (1/3)*χ.*∂grr.*∂α./grr + (1/3)*χ.*∂α.*∂gθθ./gθθ - (1/6)*α.*∂grr.*∂χ./grr
     - (1/6)*α.*∂gθθ.*∂χ./gθθ - (2/3)*∂α.*∂χ - (1/3)*α.*χ.*∂2grr./grr
     + (1/3)*α.*χ.*∂2gθθ./gθθ - (2/3)*χ.*∂2α + (1/3)*α.*∂2χ)

    ∂tK = ((3/2)*α.*(Arr./grr).^2 + (1/3)*α.*K.^2 + βr.*∂K
     + (1/2)*χ.*∂grr.*∂α./(grr.^2) - χ.*∂α.*(∂gθθ./gθθ)./grr
     + (1/2)*∂α.*∂χ./grr - χ.*∂2α./grr)

    ∂tΓr = (-v*βr.*((∂gθθ./gθθ).^2)./grr + α.*Arr.*(∂gθθ./gθθ)./(grr.^2)
     - (1/3)*v*∂βr.*(∂gθθ./gθθ)./grr + ∂βr.*(∂gθθ./gθθ)./grr
     + βr.*∂Γr + α.*Arr.*∂grr./(grr.^3)
     - (4/3)*α.*∂K./grr - 2*Arr.*∂α./(grr.^2) + (1/2)*v*∂βr.*∂grr./(grr.^2)
     - (1/2)*∂βr.*∂grr./(grr.^2) - 3*α.*Arr.*(∂χ./χ)./(grr.^2)
     + (1/6)*v*βr.*∂2grr./(grr.^2) + (1/3)*v*βr.*(∂2gθθ./gθθ)./grr
     + (1/3)*v*∂2βr./grr + ∂2βr./grr)

    ∂tΓreg = -(r/2).*∂tΓr

    ∂tgθθreg = (1 ./r.^2).*∂tgθθ

    #∂tBr = -∂tΓreg + βr.*∂Br + βr.*∂Γreg - η*Br

    ∂tBr = c*(∂tΓr + b*βr.*∂Br - b*βr.*∂Γr - η*Br)

    # Numerical Dissipation

    ∂4α = dissipation(α)
    ∂4βr = dissipation(βr)
    ∂4Br = dissipation(Br)
    ∂4χ = dissipation(χ)
    ∂4grr = dissipation(grr)
    ∂4gθθ = dissipation(gθθ)
    ∂4Arr = dissipation(Arr)
    ∂4K = dissipation(K)
    ∂4Γreg = dissipation(Γreg)

    #sign = -1 seems the best
    sign = -1
    σ = 0.3

    # ∂tα .+= (1/(2^6))*sign*σ*(drt^5)*∂6α
    # ∂tβr .+= (1/(2^6))*sign*σ*(drt^5)*∂6βr
    # ∂tBr .+= (1/(2^6))*sign*σ*(drt^5)*∂6Br
    # ∂tχ .+= (1/(2^6))*sign*σ*(drt^5)*∂6χ
    # ∂tgrr .+= (1/(2^6))*sign*σ*(drt^5)*∂6grr
    # ∂tgθθ .+= (1/(2^6))*sign*σ*(drt^5)*∂6gθθ
    # ∂tArr .+= (1/(2^6))*sign*σ*(drt^5)*∂6Arr
    # ∂tK .+= (1/(2^6))*sign*σ*(drt^5)*∂6K
    # ∂tΓreg .+= (1/(2^6))*sign*σ*(drt^5)*∂6Γreg

    ∂tα .+= (1/(16))*sign*σ*(drt^3)*∂4α
    ∂tβr .+= (1/16)*sign*σ*(drt^3)*∂4βr
    ∂tBr .+= (1/16)*sign*σ*(drt^3)*∂4Br
    ∂tχ .+= (1/16)*sign*σ*(drt^3)*∂4χ
    ∂tgrr .+= (1/16)*sign*σ*(drt^3)*∂4grr
    ∂tgθθ .+= (1/16)*sign*σ*(drt^3)*∂4gθθ
    ∂tArr .+= (1/16)*sign*σ*(drt^3)*∂4Arr
    ∂tK .+= (1/16)*sign*σ*(drt^3)*∂4K
    ∂tΓreg .+= (1/16)*sign*σ*(drt^3)*∂4Γreg

    ∂tα[1:2] .= 0.
    ∂tβr[1:2] .= 0.
    ∂tBr[1:2] .= 0.
    ∂tχ[1:2] .= 0.
    ∂tgrr[1:2] .= 0.
    ∂tgθθreg[1:2] .= 0.
    ∂tArr[1:2] .= 0.
    ∂tK[1:2] .= 0.
    ∂tΓreg[1:2] .= 0.

    #Outer Boundary Conditions
    ############################

    #Values at infinity
    # α0 = 1
    # βr0 = 0
    # Br0 = 0
    # χ0 = 1
    # grr0 = 1
    # gθθreg0 = 0
    # Arr0 = 0
    # K0 = 0
    # Γreg0 = 1
    #
    # w = 3
    #
    # hα = (r[n-2]^w)*(∂tα[n-2] + ∂α[n-2] - (α[n-2] - α0)/r[n-2] )
    # hβr = (r[n-2]^w)*(∂tβr[n-2] + ∂βr[n-2] - (βr[n-2] - βr0)/r[n-2] )
    # hBr = (r[n-2]^w)*(∂tBr[n-2] + ∂Br[n-2] - (Br[n-2] - Br0)/r[n-2] )
    # hχ = (r[n-2]^w)*(∂tχ[n-2] + ∂χ[n-2] - (χ[n-2] - χ0)/r[n-2] )
    # hgrr = (r[n-2]^w)*(∂tgrr[n-2] + ∂grr[n-2] - (grr[n-2] - grr0)/r[n-2] )
    # hgθθreg = (r[n-2]^w)*(∂tgθθreg[n-2] + ∂gθθreg[n-2] - (gθθreg[n-2] - gθθreg0)/r[n-2] )
    # hArr = (r[n-2]^w)*(∂tArr[n-2] + ∂Arr[n-2] - (Arr[n-2] - Arr0)/r[n-2] )
    # hK = (r[n-2]^w)*(∂tK[n-2] + ∂K[n-2] - (K[n-2] - K0)/r[n-2] )
    # hΓreg = (r[n-2]^w)*(∂tΓreg[n-2] + ∂Γreg[n-2] - (Γreg[n-2] - Γreg0)/r[n-2] )
    #
    # ∂tα[(n-1):n] .= (α[(n-1):n] .- α0)./r[(n-1):n] - ∂α[(n-1):n] + hα./(r[(n-1):n].^w)
    # ∂tβr[(n-1):n] .= (βr[(n-1):n] .- βr0)./r[(n-1):n] - ∂βr[(n-1):n] + hβr./(r[(n-1):n].^w)
    # ∂tBr[(n-1):n] .= (Br[(n-1):n] .- Br0)./r[(n-1):n] - ∂Br[(n-1):n] + hBr./(r[(n-1):n].^w)
    # ∂tχ[(n-1):n] .= (χ[(n-1):n] .- χ0)./r[(n-1):n] - ∂χ[(n-1):n] + hχ./(r[(n-1):n].^w)
    # ∂tgrr[(n-1):n] .= (grr[(n-1):n] .- grr0)./r[(n-1):n] - ∂grr[(n-1):n] + hgrr./(r[(n-1):n].^w)
    # ∂tgθθreg[(n-1):n] .= (gθθreg[(n-1):n] .- gθθreg0)./r[(n-1):n] - ∂gθθreg[(n-1):n] + hgθθreg./(r[(n-1):n].^w)
    # ∂tArr[(n-1):n] .= (Arr[(n-1):n] .- Arr0)./r[(n-1):n] - ∂Arr[(n-1):n] + hArr./(r[(n-1):n].^w)
    # ∂tK[(n-1):n] .= (K[(n-1):n] .- K0)./r[(n-1):n] - ∂K[(n-1):n] + hK./(r[(n-1):n].^w)
    # ∂tΓreg[(n-1):n] .= (Γreg[(n-1):n] .- Γreg0)./r[(n-1):n] - ∂Γreg[(n-1):n] + hΓreg./(r[(n-1):n].^w)

    ∂tα[(n-1):n] .= 0.
    ∂tβr[(n-1):n] .= 0.
    ∂tBr[(n-1):n] .= 0.
    ∂tχ[(n-1):n] .= 0.
    ∂tgrr[(n-1):n] .= 0.
    ∂tgθθreg[(n-1):n] .= 0.
    ∂tArr[(n-1):n] .= 0.
    ∂tK[(n-1):n] .= 0.
    ∂tΓreg[(n-1):n] .= 0.

    staterhs =
     GBSSN_Variables(∂tα,∂tβr,∂tBr,∂tχ,∂tgrr,∂tgθθreg,∂tArr,∂tK,∂tΓreg)

    return staterhs::GBSSN_Variables

end

function constraints(T,state::GBSSN_Variables,param)

    # Variables

    χ = state.χ
    grr = state.grr
    gθθreg = state.gθθ
    Arr = state.Arr
    K = state.K
    Γreg = state.Γr

    v = param[3]

    if v == 1 # Lagrangian Condition
        gθθreg[2] = ((-315*gθθreg[3] + 210*gθθreg[4] - 126*gθθreg[5]
        + 45*gθθreg[6] - 7*gθθreg[7])/63)
    end

    # Spatial Derivatives

    order = 4

    ∂rtχ = deriv(χ,order,1)
    ∂rtgrr = deriv(grr,order,1)
    ∂rtgθθreg = deriv(gθθreg,order,1)
    ∂rtArr = deriv(Arr,order,1)
    ∂rtK = deriv(K,order,1)

    ∂2rtχ = deriv2(χ,order,1)
    ∂2rtgθθreg = deriv2(gθθreg,order,1)

    # Convert derivatives from (d/drt) to (d/dr)

    r = sample(Float64, χ.grid, param[4])
    drdrt = sample(Float64, χ.grid, param[5])
    d2rdrt = sample(Float64, χ.grid, param[6])

    ∂χ = ∂rtχ./drdrt
    ∂grr = ∂rtgrr./drdrt
    ∂gθθreg = ∂rtgθθreg./drdrt
    ∂Arr = ∂rtArr./drdrt
    ∂K = ∂rtK./drdrt

    ∂2χ = (∂2rtχ - d2rdrt.*∂χ)./(drdrt.^2)
    ∂2gθθreg = (∂2rtgθθreg - d2rdrt.*∂gθθreg)./(drdrt.^2)

    gθθ = (r.^2).*(gθθreg .+ 1)
    ∂gθθ = (r.^2).*∂gθθreg + (2*r).*(gθθreg .+ 1)
    ∂2gθθ = (r.^2).*∂2gθθreg + (4*r).*∂gθθreg + 2*(gθθreg .+ 1)
    # println(gθθ)
    # println(∂gθθ)
    # println(∂2gθθ)

    Γr = -(2 ./r).*Γreg

    # Constraint Equations

    𝓗 = (-(3/2)*(Arr./grr).^2 + (2/3)*K.^2 - (5/2)*((∂χ.^2)./χ)./grr
     + 2*∂2χ./grr + 2*χ./gθθ - 2*χ.*(∂2gθθ./gθθ)./grr + 2*∂χ.*(∂gθθ./gθθ)./grr
     + χ.*(∂grr./(grr.^2)).*(∂gθθ./gθθ) - ∂χ.*∂grr./(grr.^2)
     + (1/2)*χ.*((∂gθθ./gθθ).^2)./grr)

    𝓜r = (∂Arr./grr - (2/3)*∂K - (3/2)*Arr.*(∂χ./χ)./grr
     + (3/2)*Arr.*(∂gθθ./gθθ)./grr - Arr.*∂grr./(grr.^2))

    𝓖r = -(1/2)*∂grr./(grr.^2) + Γr + (∂gθθ./gθθ)./grr

    𝓗[1:2] .= 0.
    𝓜r[1:2] .= 0.
    𝓖r[1:2] .= 0.

    return [𝓗, 𝓜r, 𝓖r]

end

function horizon(T,state::GBSSN_Variables,param)

    v = param[3]

    # Variables

    χ = state.χ
    grr = state.grr
    gθθreg = state.gθθ
    Arr = state.Arr
    K = state.K

    if v == 1 # Lagrangian Condition
        gθθreg[2] = ((-315*gθθreg[3] + 210*gθθreg[4] - 126*gθθreg[5]
        + 45*gθθreg[6] - 7*gθθreg[7])/63)
    end

    # Convert derivatives from (d/drt) to (d/dr)

    r = sample(T, χ.grid, param[4])
    drdrt = sample(T, χ.grid, param[5])

    gθθ = (r.^2).*(gθθreg .+ 1)

    Kθθ = ((1/3)*gθθ.*K - (1/2)*Arr.*gθθ./grr)./χ

    real_grr =  grr./χ

    real_gθθ =  gθθ./χ

    # Spatial Derivatives

    ∂rtreal_gθθ = deriv(real_gθθ,4,1)

    ∂real_gθθ = ∂rtreal_gθθ./drdrt

    Θ = (∂real_gθθ./real_gθθ)./real((real_grr .+ 0im).^(1/2)) - 2*Kθθ./real_gθθ

    return Θ

end


function custom_progress_message(dt,state,param,t)

    if param[1]==param[2]
        println("")
        println("| # | Time Step | Time | max α'(t) | max χ'(t) | max grr'(t) | max gθθ'(t) | max Arr'(t) | max K'(t) | max Γr'(t) |")
        println("|___|___________|______|___________|___________|_____________|_____________|_____________|___________|____________|")
        println("")
    end

    derivstate = rhs(state,param,t)

    #(𝓗, 𝓜r, 𝓖r) = constraints(state)

    println("  ",
    rpad(string(param[1]),6," "),
    rpad(string(round(dt,digits=3)),10," "),
    rpad(string(round(t,digits=3)),10," "),
    rpad(string(round(maximum(abs.(derivstate.α)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(derivstate.χ)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(derivstate.grr)),digits=3)),14," "),
    rpad(string(round(maximum(abs.(derivstate.gθθ)),digits=3)),14," "),
    rpad(string(round(maximum(abs.(derivstate.Arr)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(derivstate.K)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(derivstate.Γr)),digits=3)),14," ")
    # rpad(string(round(maximum(abs.(𝓗)),digits=3)),14," "),
    # rpad(string(round(maximum(abs.(𝓜r)),digits=3)),14," "),
    # rpad(string(round(maximum(abs.(𝓖r)),digits=3)),12," ")
    )

    #PrettyTables.jl

    param[1] += param[2]

end


function solution_saver(T,grid,sol,param,folder)

    vars = ["α","βr","Br","χ","grr","gθθ","Arr","K","Γr","H","Mr","Gr","∂tα","∂tβr","∂tBr","∂tχ","∂tgrr","∂tgθθ","∂tArr","∂tK","∂tΓreg","appHorizon"]
    #mkdir(string("data\\",folder))
    tlen = size(sol)[3]
    rlen = grid.ncells + 4
    loc = sample(T, grid, param[4])
    #loc[1] =
    cons = Array{GridFun,2}(undef,tlen,3)
    derivs = Array{GBSSN_Variables,1}(undef,tlen)
    apphorizon = Array{GridFun,1}(undef,tlen)

    for i in 1:tlen
        cons[i,1:3] .= constraints(T,sol[i],param)
        derivs[i] = rhs(sol[i],param,0)
        apphorizon[i] = horizon(T,sol[i],param)
    end


    array = Array{T,2}(undef,tlen+1,rlen+1)

    array[1,1] = 0
    array[1,2:end] .= loc

    for j = 1:22
        if j < 10
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                array[i,2:end] .= sol[:,j,i-1]
            end
        elseif j < 13
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                array[i,2:end] .= cons[i-1,j-9]
            end
        elseif j < 22
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                array[i,2:end] .= derivs[i-1][:,j-12]
            end
        else
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                array[i,2:end] .= apphorizon[i-1]
            end
        end

        CSV.write(
        string("data/",folder,"/",vars[j],".csv"),
        DataFrame(array, :auto),
        header=false
        )

    end

end

function main(points)

    T = Float64
    rspan = T[0,1000]
    rtspan = T[0,50]

    grid = setup(T, rtspan, points)
    drt = spacing(grid)
    dt = drt/4

    tspan = T[0,100]
    v = 1

    # f(b) = b*((1/(1-(rtspan[2]/b)^3))-1)^(1/3)-rspan[2]
    #
    # scale = (rspan[2]*rtspan[2])/(rspan[2]^3-rtspan[2]^3)^(1/3)
    # # println(scale)
    #
    # r(rt) = scale*real(((1/(1-(rt/scale)^3))-1+0im)^(1/3))
    # drdrt(rt) = real((1/(1-(rt/scale)^3)+0im)^(4/3))
    # d2rdrt(rt) = 4*((rt^2)/(scale^3))*real((1/(1-(rt/scale)^3)+0im)^(7/3))

    # rv = sample(Float64, grid, r)
    # println(rv)

    f(b) = b*tan(rtspan[2]/b)-rspan[2]

    scale = find_zero(f, 0.64*rtspan[2])
    #println(scale)

    r(rt) = scale*tan(rt/scale)
    drdrt(rt) = sec(rt/scale)^2
    d2rdrt(rt) = (2/scale)*(sec(rt/scale)^2)*tan(rt/scale)

    # r(rt) = rt
    # drdrt(rt) = 1
    # d2rdrt(rt) = 0


    # points = 100
    atol = eps(T)^(T(3) / 4)
    alg = RK4()
    #alg = Tsit5()
    #printlogo()

    printtimes = 1
    custom_progress_step = Int(printtimes/dt)
    step_iterator = custom_progress_step
    param = [step_iterator, custom_progress_step, v, r, drdrt, d2rdrt]
    println("Defining Initial State...")
    state = init(T, grid, param)::GBSSN_Variables
    println("Defining Problem...")
    prob = ODEProblem(rhs, state, tspan, param)
    println("Starting Solution...")
    # derivstate = rhs(state,param,0)
    # cons = constraints(state)
    # loc = T[location(grid, n) for n in 1:(grid.ncells + 4)]

    # println(loc)
    # println(size(loc))

    # return (loc,state,derivstate,cons)

    # print(derivstate.Arr)
    # print(state.Γr)

    sol = solve(
        prob, alg,
        abstol = atol,
        dt = drt/4,
        adaptive = false,
        saveat = 1,
        progress = true,
        progress_steps=custom_progress_step,
        progress_message=custom_progress_message
    )

    solution_saver(T,grid,sol,param,"Schwarzschild100L--")

    # println(sol(0.04)[:,1])
    #
    # x1 = T[location(grid, n) for n in 1:(grid.ncells + 4)]
    #
    # plot(x1, sol(0.04)[:,1])
    # print("Written to: ")
    # CSV.write("data/Minkowski_test.csv",DataFrame(sol))
    # println("\nCompleted!\n")
    # println("Written to: data/GR_test.csv")
end


end
