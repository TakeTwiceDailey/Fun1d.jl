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
    A::GridFun{S,T}
    βr::GridFun{S,T}
    Br::GridFun{S,T}
    χ::GridFun{S,T}
    γtrr::GridFun{S,T}
    γtθθ::GridFun{S,T}
    Arr::GridFun{S,T}
    K::GridFun{S,T}
    Γr::GridFun{S,T}
    𝜙::GridFun{S,T}
    K𝜙::GridFun{S,T}
end

cont(x::GBSSN_Variables) = (x.α, x.A, x.βr, x.Br, x.χ, x.γtrr, x.γtθθ, x.Arr, x.K, x.Γr, x.𝜙, x.K𝜙)
numvar = 12

# Iteration
Base.IteratorSize(::Type{<:GBSSN_Variables}) = Iterators.HasShape{2}()
Base.eltype(::Type{GBSSN_Variables{S,T}}) where {S,T} = T
Base.isempty(x::GBSSN_Variables) = isempty(x.α)
function Base.iterate(x::GBSSN_Variables, state...)
    return iterate(Iterators.flatten(cont(x)), state...)
end
Base.size(x::GBSSN_Variables) = (length(x.α), numvar)
Base.size(x::GBSSN_Variables, d) = size(x)[d]

# Indexing
function lin2cart(x::GBSSN_Variables, i::Number)
    n = length(x.α)
    return (i - 1) % n + 1, (i - 1) ÷ n + 1
end
Base.firstindex(x::GBSSN_Variables) = error("not implemented")
Base.getindex(x::GBSSN_Variables, i) = getindex(x, i.I...)
Base.getindex(x::GBSSN_Variables, i::Number) = getindex(x, lin2cart(x, i)...)
Base.getindex(x::GBSSN_Variables, i, j) = getindex(cont(x)[j], i)
Base.lastindex(x::GBSSN_Variables) = error("not implemented")
Base.setindex!(x::GBSSN_Variables, v, i) = setindex!(x, v, i.I...)
Base.setindex!(x::GBSSN_Variables, v, i::Number) = setindex!(x, v, lin2cart(x, i))
Base.setindex!(x::GBSSN_Variables, v, i, j) = setindex!(cont(x)[j], v, i)

# Abstract Array
Base.IndexStyle(::GBSSN_Variables) = IndexCartesian()
Base.similar(x::GBSSN_Variables) = GBSSN_Variables(map(similar, cont(x))...)
function Base.similar(x::GBSSN_Variables, ::Type{T}) where {T}
    return GBSSN_Variables(map(y -> similar(y,T), cont(x))...)
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
        map(fun, x.A, (y.A for y in ys)...),
        map(fun, x.βr, (y.βr for y in ys)...),
        map(fun, x.Br, (y.Br for y in ys)...),
        map(fun, x.χ, (y.χ for y in ys)...),
        map(fun, x.γtrr, (y.γtrr for y in ys)...),
        map(fun, x.γtθθ, (y.γtθθ for y in ys)...),
        map(fun, x.Arr, (y.Arr for y in ys)...),
        map(fun, x.K, (y.K for y in ys)...),
        map(fun, x.Γr, (y.Γr for y in ys)...),
        map(fun, x.𝜙, (y.𝜙 for y in ys)...),
        map(fun, x.K𝜙, (y.K𝜙 for y in ys)...)
        )
end

# function Base.rand(rng::AbstractRNG, ::Random.SamplerType{GBSSN_Variables{T}}) where {T}
#     return GBSSN_Variables{T}(rand(rng, T), rand(rng, T))
# end

Base.zero(::Type{<:GBSSN_Variables}) = error("not implemented")
Base.zero(x::GBSSN_Variables) = GBSSN_Variables(map(zero, cont(x))...)

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
"         ___\\/ / /\\\\\\\\\\\\\\\\\\  /____\\/ /\\\\\\\\_\\// //\\\\\\\\_____by Erik Schnetter_____\n",
"          ___\\/_/ / / / / /_/______\\/ /  /___\\// /  /_______and Conner Dailey____\n",
"           _____\\/_/_/_/_/__________\\/__/______\\/__/______________________________\n"
)

end

function setup(::Type{S}, rspan, points::Int) where {S}

    # Specify the radial domain and the
    # radial step amount and make a Grid object
    rmin,rmax = rspan
    domain = Domain{S}(rmin, rmax)
    grid = Grid(domain, points)
    return grid

end

function dissipation(f::GridFun{S,T}) where {S,T}

    ############################################
    # Calculates the numerical dissipation terms
    #
    # Finite differencing fails to model high
    # frequency modes in the system, and these
    # modes can lead to instabilities. Adding
    # numerical dissipation terms damps these
    # high frequency modes by subtracting off
    # a high order derivative from each dynamical
    # variable in the system.
    ############################################

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

    ############################################
    # Specifies the Initial Conditions
    ############################################

    n = grid.ncells + 4
    domain = grid.domain
    initgrid = grid
    drt = spacing(grid)
    r = param[4]
    drdrt = param[5]
    d2rdrt = param[6]
    m = param[7]

    num = 0

    # Initial conditions for Schwarzschild metric (Ker-Schild Coordinates)

    # Mass (no real reason not to use 1 here)
    M = 1

    fα(rt) = real((1+2*M/(r(rt))+0im)^(-1/2))
    fA(rt) = 0.
    fβr(rt) = (2*M/r(rt))/(1+2*M/r(rt))
    fBr(rt) = 0.
    fχ(rt) = 1.
    fγtrr(rt) = 1+2*M/r(rt)
    fγtθθ(rt) = r(rt)^2
    fArr(rt) = -(4*M/3)*(3*M+2*r(rt))/real(((r(rt)^5)*(r(rt)+2*M)+0im)^(1/2))
    fK(rt) = (2*M)*(3*M+r(rt))/real((r(rt)*(r(rt)+2*M)+0im)^(3/2))
    fΓr(rt) = -(2*r(rt)+5*M)/(r(rt)+2*M)^2

    r0 = 10
    σr = 0.5
    Amp = 0.1

    f𝜙(rt) = Amp*(1/r(rt))*exp(-(1/2)*((r(rt)-r0)/σr)^2)
    f∂𝜙(rt) = Amp*exp(-(1/2)*((r(rt)-r0)/σr)^2)*(r(rt)*r0-r(rt)^2-σr^2)/(r(rt)^2*σr^2)
    f∂t𝜙(rt) = 0.

    f∂χ(rt) = 0.
    f∂γtrr(rt) = -2*M/(r(rt)^2)
    f∂γtθθ(rt) = 2*r(rt)
    f∂Arr(rt) = (4*M/3)*(15*M^2+15*M*r(rt)+4*r(rt)^2)/real(((r(rt)^7)*((r(rt)+2*M)^3)+0im)^(1/2))
    f∂K(rt) = -2*M*(9*M^2+10*M*r(rt)+2*r(rt)^2)/real((r(rt)*(r(rt)+2*M)+0im)^(5/2))
    f∂Γr(rt) = 2*(r(rt)+3*M)/(r(rt)+2*M)^3

    f∂2γtθθ(rt) = 2

    fgtt(rt) = -(1/fα(rt)^2)
    fgtr(rt) = fβr(rt)/fα(rt)^2
    fgrr(rt,χ) = χ/fγtrr(rt) - (fβr(rt)/fα(rt))^2

    # Lagrangian Density for scalar field

    f𝓛(rt,χ) = ((1/2)*fgtt(rt)*f∂t𝜙(rt)^2 + (1/2)*fgrr(rt,χ)*f∂𝜙(rt)^2 + fgtr(rt)*f∂𝜙(rt)*f∂t𝜙(rt) - (1/2)*(m^2)*f𝜙(rt)^2)

    # Stresss Energy components (contravariant indices)

    fTtt(rt,χ) = (fgtt(rt)*f∂t𝜙(rt) + fgtr(rt)*f∂𝜙(rt)).^2 - fgtt(rt)*f𝓛(rt,χ)
    fTtr(rt,χ) = ((fgtt(rt)*f∂t𝜙(rt) + fgtr(rt)*f∂𝜙(rt))*(fgtr(rt)*f∂t𝜙(rt) + fgrr(rt,χ)*f∂𝜙(rt)) - fgtr(rt)*f𝓛(rt,χ))

    fρ(rt,χ) = (fα(rt)^2)*fTtt(rt,χ)
    fSr(rt,χ) = fα(rt)*fTtr(rt,χ)

    f∂χ(rt,(χ, X, Arr)) = X

    function f∂X(rt,(χ, X, Arr))
     -(1/2)*fγtrr(rt)*(-(3/2)*(Arr/fγtrr(rt))^2 + (2/3)*fK(rt)^2
     - (5/2)*((X^2)/χ)/fγtrr(rt)
     + 2*χ/fγtθθ(rt) - 2*χ*(f∂2γtθθ(rt)/fγtθθ(rt))/fγtrr(rt)
     + 2*X*(f∂γtθθ(rt)/fγtθθ(rt))/fγtrr(rt)
     + χ*(f∂γtrr(rt)/(fγtrr(rt)^2))*(f∂γtθθ(rt)/fγtθθ(rt))
     - X*f∂γtrr(rt)/(fγtrr(rt)^2)
     + (1/2)*χ*((f∂γtθθ(rt)/fγtθθ(rt))^2)/fγtrr(rt) - 16*pi*fρ(rt,χ))
    end

    function f∂Arr(rt,(χ, X, Arr))
     -fγtrr(rt)*(-(2/3)*f∂K(rt) - (3/2)*Arr*(X/χ)/fγtrr(rt)
     + (3/2)*Arr*(f∂γtθθ(rt)/fγtθθ(rt))/fγtrr(rt) - Arr*f∂γtrr(rt)/(fγtrr(rt)^2)
     - 8*pi*fγtrr(rt)*fSr(rt,χ)/χ)
    end

    # grid = Grid(domain,Int((2^num)*(n-5)+1))
    # n = grid.ncells + 4

    α = sample(T, grid, fα)
    A = sample(T, grid, fA)
    βr = sample(T, grid, fβr)
    Br = sample(T, grid, fBr)
    χ = sample(T, grid, fχ)
    γtrr = sample(T, grid, fγtrr)
    γtθθ = sample(T, grid, fγtθθ)
    Arr = sample(T, grid, fArr)
    K = sample(T, grid, fK)
    Γr = sample(T, grid, fΓr)
    𝜙 = sample(T, grid, f𝜙)
    ∂𝜙 = sample(T, grid, f∂𝜙)
    ∂t𝜙 = sample(T, grid, f∂t𝜙)

    ∂χ = sample(T, grid, f∂χ)
    # ∂γtrr = sample(T, grid, f∂γtrr)
    # ∂γtθθ = sample(T, grid, f∂γtθθ)
    # ∂K = sample(T, grid, f∂K)
    ∂Arr = sample(T, grid, f∂Arr)

    # ∂2γtθθ = sample(T, grid, f∂2γtθθ)
    #
    # order = 4
    #
    rr = sample(T, grid, param[4])
    # drdrt = sample(Float64, grid, param[5])
    #
    # ∂rt𝜙 = deriv(𝜙,order,1)
    # ∂𝜙 = ∂rt𝜙./drdrt

    K𝜙 = -(∂t𝜙 - βr.*∂𝜙)./(2*α)

    X = ∂χ
    ∂X = sample(T, grid, rt -> 0.)

    Kreg = real((rr .+ 0im).^(3/2)).*K
    # ∂Kreg = real((r .+ 0im).^(3/2)).*∂K + (3/2)*real((r .+ 0im).^(1/2)).*K
    # Arrreg = real((r .+ 0im).^(5/2)).*Arr
    # ∂Arrreg = real((r .+ 0im).^(5/2)).*∂Arr + (5/2)*real((r .+ 0im).^(3/2)).*Arr
    #
    γtθθreg = sample(T, grid, rt -> 0)

    # # Inverse metric (contravariant indices)
    #
    # gtt = -(1 ./α.^2)
    # gtr = βr./α.^2
    # grr = χ./γtrr - (βr./α).^2
    # gθθ = χ./γtθθ
    #
    # # Lagrangian Density for scalar field
    #
    # 𝓛 = (1/2)*gtt.*∂t𝜙.^2 + (1/2)*grr.*∂𝜙.^2 + gtr.*∂𝜙.*∂t𝜙 - (1/2)*(m^2)*𝜙.^2
    #
    # # Stresss Energy components (contravariant indices)
    #
    # Ttt = (gtt.*∂t𝜙 + gtr.*∂𝜙).^2 - gtt.*𝓛
    # Ttr = (gtt.*∂t𝜙 + gtr.*∂𝜙).*(gtr.*∂t𝜙 + grr.*∂𝜙) - gtr.*𝓛
    #
    # ρ = (α.^2).*Ttt
    # Sr = α.*Ttr

    # Constraint Equations

    rt = domain.xmin - drt
    r = param[4]

    for i = 2:n-1

        ∂χ[i] = f∂χ(rt,(χ[i], X[i], Arr[i]))
        ∂X[i] = f∂X(rt,(χ[i], X[i], Arr[i]))
        ∂Arr[i] = f∂Arr(rt,(χ[i], X[i], Arr[i]))

        χ[i+1] = χ[i] + drt*(3*∂χ[i]-∂χ[i-1])/2
        X[i+1] = X[i] + drt*(3*∂X[i]-∂X[i-1])/2
        Arr[i+1] = Arr[i] + drt*(3*∂Arr[i]-∂Arr[i-1])/2

        # k2χ = f∂χ(rt+drt/2,(χ[i]+drt*k1χ/2,X[i]+drt*k1X/2,Arr[i]+drt*k1Arr/2))
        # k2X = f∂X(rt+drt/2,(χ[i]+drt*k1χ/2,X[i]+drt*k1X/2,Arr[i]+drt*k1Arr/2))
        # k2Arr = f∂Arr(rt+drt/2,(χ[i]+drt*k1χ/2,X[i]+drt*k1X/2,Arr[i]+drt*k1Arr/2))
        #
        # k3χ = f∂χ(rt+drt/2,(χ[i]+drt*k2χ/2,X[i]+drt*k2X/2,Arr[i]+drt*k2Arr/2))
        # k3X = f∂X(rt+drt/2,(χ[i]+drt*k2χ/2,X[i]+drt*k2X/2,Arr[i]+drt*k2Arr/2))
        # k3Arr = f∂Arr(rt+drt/2,(χ[i]+drt*k2χ/2,X[i]+drt*k2X/2,Arr[i]+drt*k2Arr/2))
        #
        # k4χ = f∂χ(rt+drt,(χ[i]+drt*k3χ,X[i]+drt*k3X,Arr[i]+drt*k3Arr))
        # k4X = f∂X(rt+drt,(χ[i]+drt*k3χ,X[i]+drt*k3X,Arr[i]+drt*k3Arr))
        # k4Arr = f∂Arr(rt+drt,(χ[i]+drt*k3χ,X[i]+drt*k3X,Arr[i]+drt*k3Arr))

        # χ[i+1] = χ[i] + drt*(k1χ + 2*k2χ + 2*k3χ + k4χ)/6
        # X[i+1] = X[i] + drt*(k1X + 2*k2X + 2*k3X + k4X)/6
        # Arr[i+1] = Arr[i] + drt*(k1Arr + 2*k2Arr + 2*k3Arr + k4Arr)/6

        # Arrreg[i] = real((r(rt)+ 0im)^(5/2))*Arr[i]
        # k1Arrreg = real((r(rt)+ 0im)^(5/2))*k1Arr + (5/2)*real((r(rt)+ 0im)^(3/2))*Arr[i]
        # k2Arrreg = real((r(rt+drt/2)+ 0im)^(5/2))*k2Arr + (5/2)*real((r(rt+drt/2)+ 0im)^(3/2))*(Arr[i]+drt*k1Arr/2)
        # k3Arrreg = real((r(rt+drt/2)+ 0im)^(5/2))*k3Arr + (5/2)*real((r(rt+drt/2)+ 0im)^(3/2))*(Arr[i]+drt*k2Arr/2)
        # k4Arrreg = real((r(rt+drt)+ 0im)^(5/2))*k4Arr + (5/2)*real((r(rt+drt)+ 0im)^(3/2))*(Arr[i]+drt*k3Arr)
        #
        # Arrreg[i+1] = Arrreg[i] + drt*(k1Arrreg + 2*k2Arrreg + 2*k3Arrreg + k4Arrreg)/6
        #
        # Arr[i+1] = real((r(rt+drt)+ 0im)^(-5/2))*Arrreg[i+1]
        #
        #Arr[i+1] = Arr[i] + drt*(k1Arr + 2*k2Arr + 2*k3Arr + k4Arr)/6
        #
        # Arrreg[i+1] = real((r(rt+drt)+ 0im)^(5/2))*Arr[i+1]

        # grr[i] = χ[i]/γtrr[i] - (βr[i]/α[i])^2
        # gθθ[i] = χ[i]/γtθθ[i]
        #
        # 𝓛[i] = ((1/2)*gtt[i]*∂t𝜙[i]^2 + (1/2)*grr[i]*∂𝜙[i]^2
        # + gtr[i]*∂𝜙[i]*∂t𝜙[i] - (1/2)*(m^2)*𝜙[i]^2)
        #
        # Ttt[i] = (gtt[i]*∂t𝜙[i] + gtr[i]*∂𝜙[i])^2 - gtt[i]*𝓛[i]
        # Ttr[i] = ((gtt[i]*∂t𝜙[i] + gtr[i]*∂𝜙[i])*(gtr[i]*∂t𝜙[i] + grr[i]*∂𝜙[i])
        # - gtr[i]*𝓛[i])
        #
        # ρ[i] = (α[i]^2)*Ttt[i]
        # Sr[i] = α[i]*Ttr[i]
        #
        # ∂χ[i] = X[i]
        #
        # ∂X[i] = -(1/2)*γtrr[i]*(-(3/2)*(Arr[i]/γtrr[i])^2 + (2/3)*K[i]^2
        #  - (5/2)*((X[i]^2)/χ[i])/γtrr[i]
        #  + 2*χ[i]/γtθθ[i] - 2*χ[i]*(∂2γtθθ[i]/γtθθ[i])/γtrr[i]
        #  + 2*X[i]*(∂γtθθ[i]/γtθθ[i])/γtrr[i]
        #  + χ[i]*(∂γtrr[i]/(γtrr[i]^2))*(∂γtθθ[i]/γtθθ[i])
        #  - X[i]*∂γtrr[i]/(γtrr[i]^2)
        #  + (1/2)*χ[i]*((∂γtθθ[i]/γtθθ[i])^2)/γtrr[i] - 16*pi*ρ[i])
        #
        # ∂Arr[i] = -γtrr[i]*(-(2/3)*∂K[i] - (3/2)*Arr[i]*(X[i]/χ[i])/γtrr[i]
        #  + (3/2)*Arr[i]*(∂γtθθ[i]/γtθθ[i])/γtrr[i] - Arr[i]*∂γtrr[i]/(γtrr[i]^2)
        #  - 8*pi*γtrr[i]*Sr[i]/χ[i])
        #
        # χ[i+1] = χ[i] + dr*(3*∂χ[i]-∂χ[i-1])/2
        # X[i+1] = X[i] + dr*(3*∂X[i]-∂X[i-1])/2

        tol = 1.
        atol = eps(T)^(T(3) / 4)

        while true

            initχ = χ[i+1]
            initX = X[i+1]
            initArr = Arr[i+1]

            #Arr[i+1] = real((r(rt+drt)+ 0im)^(-5/2))*Arrreg[i+1]

            ∂χ[i+1] = f∂χ(rt+drt,(χ[i+1],X[i+1],Arr[i+1]))
            ∂X[i+1] = f∂X(rt+drt,(χ[i+1],X[i+1],Arr[i+1]))
            ∂Arr[i+1] = f∂Arr(rt+drt,(χ[i+1],X[i+1],Arr[i+1]))

            #∂Arrreg[i+1] = real((r(rt+drt)+ 0im)^(5/2))*∂Arr[i+1] + (5/2)*real((r(rt+drt)+ 0im)^(3/2))*Arr[i+1]

            χ[i+1] = χ[i] + drt*(∂χ[i] + ∂χ[i+1])/2
            X[i+1] = X[i] + drt*(∂X[i] + ∂X[i+1])/2
            Arr[i+1] = Arr[i] + drt*(∂Arr[i] + ∂Arr[i+1])/2

            global tol = maximum(abs.((initχ-χ[i+1],initX-X[i+1],initArr-Arr[i+1])))

            if tol < atol
                break
            end

        end

        rt += drt

     end

    #  grid = initgrid
    #  n = grid.ncells + 4
    #
    #  r = param[4]
    #  drdrt = param[5]
    #  d2rdrt = param[6]
    #
    #  α = sample(T, grid, fα)
    #  A = sample(T, grid, fA)
    #  βr = sample(T, grid, fβr)
    #  Br = sample(T, grid, fBr)
    #  γtrr = sample(T, grid, fγtrr)
    #  γtθθreg = sample(T, grid, rt -> 0)
    #  K = sample(T, grid, fK)
    #  Γr = sample(T, grid, fΓr)
    #  𝜙 = sample(T, grid, f𝜙)
    #  ∂t𝜙 = sample(T, grid, f∂t𝜙)
    #
    #  dsχ = sample(T, grid, fχ)
    #  dsArr = sample(T, grid, fArr)
    #
    #  r = sample(T, grid, param[4])
    #  drdrt = sample(T, grid, param[5])
    #
    #  Kreg = real((r .+ 0im).^(3/2)).*K
    #dsArrreg = real((r.+0im).^(5/2)).*dsArr
    #
    #  ∂rt𝜙 = deriv(𝜙,order,1)
    #  ∂𝜙 = ∂rt𝜙./drdrt
    #
    #  K𝜙 = -(∂t𝜙 - βr.*∂𝜙)./(2*α)
    #
    #  for i = 3:n-3
    #
    #      val = Int((i-3)*(2^num) + 3)
    #
    #      dsχ[i] = χ[val]
    #      dsArr[i] =  Arr[val]
    #      dsArrreg[i] = real((r[i]+0im)^(5/2))*dsArr[i]
    #
    #      if (i < 10 || i > n - 6 )
    #         println(val)
    #     end
    #
    # end

    #rr = sample(T, grid, param[4])

    Arrreg = real((rr.+0im).^(5/2)).*Arr

    α = sample(T, grid, fα)
    A = sample(T, grid, fA)
    βr = sample(T, grid, fβr)
    Br = sample(T, grid, fBr)
    χ = sample(T, grid, fχ)
    γtrr = sample(T, grid, fγtrr)
    γtθθ = sample(T, grid, fγtθθ)
    Arr = sample(T, grid, fArr)
    K = sample(T, grid, fK)
    Γr = sample(T, grid, fΓr)
    𝜙 = sample(T, grid, rt->0)
    K𝜙 = sample(T, grid, rt->0)

    Kreg = real((rr .+ 0im).^(3/2)).*K
    Arrreg = real((rr .+ 0im).^(5/2)).*Arr
    γtθθreg = sample(T, grid, rt -> 0)


    state = GBSSN_Variables(α, A, βr, Br, χ, γtrr, γtθθreg, Arrreg, Kreg, Γr, 𝜙, K𝜙)

    cons = constraints(T,state,param)

    plot(rr[3:n-10],cons[1][3:n-10])

    #return GBSSN_Variables(α, A, βr, Br, dsχ, γtrr, γtθθreg, dsArrreg, Kreg, Γr, 𝜙, K𝜙)

end


function rhs(state::GBSSN_Variables, param, t)

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

    # Unpack the Variables

    α = state.α
    A = state.A
    βr = state.βr
    Br = state.Br
    χ = state.χ
    γtrr = state.γtrr
    γtθθreg = state.γtθθ
    Arrreg = state.Arr
    Kreg = state.K
    Γr = state.Γr
    𝜙 = state.𝜙
    K𝜙 =state.K𝜙

    drt = spacing(α.grid)
    n = α.grid.ncells + 4

    m = param[7]

    # Boundary Conditions

    # These inner boundary conditions are necessary for stable
    # evolution for the specified gauge condition and do not
    # specify anything physical about the system.

    # Eulerian/Lagrangian condition (0/1)
    v = param[3]

    # if v == 0 # Eulerian Condition
    #     βr[2] = (17*βr[3] + 9*βr[4] - 5*βr[5] + βr[6])/22
    # elseif v == 1 # Lagrangian Condition
    #     γtθθreg[2] = ((-315*γtθθreg[3] + 210*γtθθreg[4]
    #     - 126*γtθθreg[5] + 45*γtθθreg[6] - 7*γtθθreg[7])/63)
    #     #βr[2] = (-315*βr[3] + 210*βr[4] - 126*βr[5] + 45*βr[6] - 7*βr[7])/63
    # end

    # Spatial Derivatives (finite differences) with respect to coordinate rt

    # Accuarcy order, 2 for 2nd order, 4 for 4th order
    order = 4

    # First derivatives
    ∂rtα = deriv(α,order,1)
    ∂rtβr = deriv(βr,order,-1)
    ∂rtBr = deriv(Br,order,-1)
    ∂rtχ = deriv(χ,order,1)
    ∂rtγtrr = deriv(γtrr,order,1)
    ∂rtγtθθreg = deriv(γtθθreg,order,1)
    ∂rtArrreg = deriv(Arrreg,order,1)
    ∂rtKreg = deriv(Kreg,order,1)
    ∂rtΓr = deriv(Γr,order,-1)
    ∂rt𝜙 = deriv(𝜙,order,1)
    ∂rtK𝜙 = deriv(K𝜙,order,-1)

    # Second derivatives
    ∂2rtα = deriv2(α,order,1)
    ∂2rtβr = deriv2(βr,order,-1)
    ∂2rtχ = deriv2(χ,order,1)
    ∂2rtγtrr = deriv2(γtrr,order,1)
    ∂2rtγtθθreg = deriv2(γtθθreg,order,1)
    ∂2rt𝜙 = deriv2(𝜙,order,1)

    # Coordinate transformations from computational rt coordinate
    # to physical r coordinate

    r = sample(Float64, α.grid, param[4])
    drdrt = sample(Float64, α.grid, param[5])
    d2rdrt = sample(Float64, α.grid, param[6])

    ∂α = ∂rtα./drdrt
    ∂βr = ∂rtβr./drdrt
    ∂Br = ∂rtBr./drdrt
    ∂χ = ∂rtχ./drdrt
    ∂γtrr = ∂rtγtrr./drdrt
    ∂γtθθreg = ∂rtγtθθreg./drdrt
    ∂Arrreg = ∂rtArrreg./drdrt
    ∂Kreg = ∂rtKreg./drdrt
    ∂Γr = ∂rtΓr./drdrt
    ∂𝜙 = ∂rt𝜙./drdrt
    ∂K𝜙 = ∂rtK𝜙./drdrt

    ∂2α = (∂2rtα - d2rdrt.*∂α)./(drdrt.^2)
    ∂2βr = (∂2rtβr - d2rdrt.*∂βr)./(drdrt.^2)
    ∂2χ = (∂2rtχ - d2rdrt.*∂χ)./(drdrt.^2)
    ∂2γtrr = (∂2rtγtrr - d2rdrt.*∂γtrr)./(drdrt.^2)
    ∂2γtθθreg = (∂2rtγtθθreg - d2rdrt.*∂γtθθreg)./(drdrt.^2)
    ∂2𝜙 = (∂2rt𝜙 - d2rdrt.*∂𝜙)./(drdrt.^2)

    # r[1:2] .= 0.
    # βr[1:2] .= 0.
    # γtθθreg[1:2] .= 0.
    # Γr[1:2] .= 0.

    # Conversions from regularized variables to canonical variables

    γtθθ = (r.^2).*(γtθθreg .+ 1)
    ∂γtθθ = (r.^2).*∂γtθθreg + (2*r).*(γtθθreg .+ 1)
    ∂2γtθθ = (r.^2).*∂2γtθθreg + (4*r).*∂γtθθreg + 2*(γtθθreg .+ 1)

    # γtθθ[1:2] .= 0.
    # ∂γtθθ[1:2] .= 0.
    # ∂2γtθθ[1:2] .= 0.

    K = real((r .+ 0im).^(-3/2)).*Kreg
    ∂K = real((r .+ 0im).^(-3/2)).*∂Kreg - (3/2)*real((r .+ 0im).^(-5/2)).*Kreg

    Arr = real((r .+ 0im).^(-5/2)).*Arrreg
    ∂Arr = real((r .+ 0im).^(-5/2)).*∂Arrreg - (5/2)*real((r .+ 0im).^(-7/2)).*Arrreg

    # Γr = -(2 ./r).*Γreg
    # ∂Γr = -(2 ./r).*∂Γreg + (2 ./(r.^2)).*Γreg

    # Γr = Γreg
    # ∂Γr = ∂Γreg

    # Gauge Conditions

    # Coordinate drift parameter.
    # Positive values lead to continued evolution
    # zero gives eventual steady state
    η = 0

    #Superscript condition...1 is a plus, 0 is a minus
    a = 0

    #Subscript condition...1 is a plus, 0 is a minus
    b = 0

    # Zero condition, 1 includes shift, 0 for vanishing shift
    # if this is 0, a and b don't matter as long as the
    # initial shift is zero
    c = 1

    #########################################################
    # Evolution Equations
    #
    # This is the full suite of evolution equations
    # for GR in spherical symmetry in the BSSN framework.
    # I have tried to keep them looking as close to their
    # mathematically written counterpart as possible.
    #
    # They are written in the order they appear in the
    # reference (arXiv:0705.3845v2) except for the ∂tBr
    # equation since it contains a ∂tΓr term.
    #
    #########################################################

    ∂tα = a*βr.*∂α - 2*α.*A

    ∂tβr = c*((3/4)*Br + b*βr.*∂βr)

    ∂tχ = ((2/3)*K.*α.*χ - (1/3)*v*βr.*χ.*∂γtrr./γtrr - (2/3)*v*βr.*χ.*∂γtθθ./γtθθ
     - (2/3)*v*χ.*∂βr + βr.*∂χ)

    ∂tγtrr = (-2*Arr.*α - (1/3)*v*βr.*∂γtrr + βr.*∂γtrr
     - (2/3)*v*γtrr.*βr.*∂γtθθ./γtθθ + 2*γtrr.*∂βr - (2/3)*v*γtrr.*∂βr)

    ∂tγtθθ = (Arr.*γtθθ.*α./γtrr - (1/3)*v*γtθθ.*βr.*∂γtrr./γtrr - (2/3)*v*βr.*∂γtθθ
     + βr.*∂γtθθ - (2/3)*v*γtθθ.*∂βr)

    ∂tArr = (-2*α.*(Arr.^2)./γtrr + K.*α.*Arr - (1/3)*v*βr.*Arr.*∂γtrr./γtrr
     - (2/3)*v*βr.*Arr.*∂γtθθ./γtθθ - (2/3)*v*Arr.*∂βr + 2*Arr.*∂βr
     + (2/3)*α.*χ.*(∂γtrr./γtrr).^2 - (1/3)*α.*χ.*(∂γtθθ./γtθθ).^2
     - (1/6)*α.*(∂χ.^2)./χ - (2/3)*α.*χ.*γtrr./γtθθ + βr.*∂Arr
     + (2/3)*α.*χ.*γtrr.*∂Γr - (1/2)*α.*χ.*(∂γtrr./γtrr).*(∂γtθθ./γtθθ)
     + (1/3)*χ.*∂γtrr.*∂α./γtrr + (1/3)*χ.*∂α.*∂γtθθ./γtθθ - (1/6)*α.*∂γtrr.*∂χ./γtrr
     - (1/6)*α.*∂γtθθ.*∂χ./γtθθ - (2/3)*∂α.*∂χ - (1/3)*α.*χ.*∂2γtrr./γtrr
     + (1/3)*α.*χ.*∂2γtθθ./γtθθ - (2/3)*χ.*∂2α + (1/3)*α.*∂2χ)

    ∂tK = ((3/2)*α.*(Arr./γtrr).^2 + (1/3)*α.*K.^2 + βr.*∂K
     + (1/2)*χ.*∂γtrr.*∂α./(γtrr.^2) - χ.*∂α.*(∂γtθθ./γtθθ)./γtrr
     + (1/2)*∂α.*∂χ./γtrr - χ.*∂2α./γtrr)

    ∂tΓr = (-v*βr.*((∂γtθθ./γtθθ).^2)./γtrr + α.*Arr.*(∂γtθθ./γtθθ)./(γtrr.^2)
     - (1/3)*v*∂βr.*(∂γtθθ./γtθθ)./γtrr + ∂βr.*(∂γtθθ./γtθθ)./γtrr
     + βr.*∂Γr + α.*Arr.*∂γtrr./(γtrr.^3)
     - (4/3)*α.*∂K./γtrr - 2*Arr.*∂α./(γtrr.^2) + (1/2)*v*∂βr.*∂γtrr./(γtrr.^2)
     - (1/2)*∂βr.*∂γtrr./(γtrr.^2) - 3*α.*Arr.*(∂χ./χ)./(γtrr.^2)
     + (1/6)*v*βr.*∂2γtrr./(γtrr.^2) + (1/3)*v*βr.*(∂2γtθθ./γtθθ)./γtrr
     + (1/3)*v*∂2βr./γtrr + ∂2βr./γtrr)

    ∂tA = ∂tK

    ∂tBr = c*(∂tΓr + b*βr.*∂Br - b*βr.*∂Γr - η*Br)

    #########################################################
    # Source Terms and Source Evolution
    #
    # This currently includes the addition of source terms
    # to GR that come from a Klein-Gordon scalar field
    #
    #########################################################

    # Klein-Gordon System

    ∂t𝜙 = βr.*∂𝜙 - 2*α.*K𝜙
    ∂tK𝜙 = (βr.*∂K𝜙 + α.*K.*K𝜙 - (1/2)*α.*χ.*∂2𝜙./γtrr
        + (1/4)*α.*χ.*∂γtrr.*∂𝜙./γtrr.^2 - (1/4)*α.*∂χ.*∂𝜙./γtrr
        - (1/2)*χ.*∂α.*∂𝜙./γtrr - (1/2)*χ.*∂γtθθ.*∂𝜙./(γtrr.*γtθθ)
        + (1/2)*∂χ.*∂𝜙./(γtrr) + (1/2)*m^2*𝜙)

    # Inverse metric (contravariant indices)

    gtt = -(1 ./α.^2)
    gtr = βr./α.^2
    grr = χ./γtrr - (βr./α).^2
    gθθ = χ./γtθθ

    # Lagrangian Density for scalar field

    𝓛 = (1/2)*gtt.*∂t𝜙.^2 + (1/2)*grr.*∂𝜙.^2 + gtr.*∂𝜙.*∂t𝜙 - (1/2)*(m^2)*𝜙.^2

    # Stresss Energy components (contravariant indices)

    Ttt = (gtt.*∂t𝜙 + gtr.*∂𝜙).^2 - gtt.*𝓛
    Trr = (gtr.*∂t𝜙 + grr.*∂𝜙).^2 - grr.*𝓛
    Ttr = (gtt.*∂t𝜙 + gtr.*∂𝜙).*(gtr.*∂t𝜙 + grr.*∂𝜙) - gtr.*𝓛
    Tθθ = -gθθ.*𝓛

    # Source Terms to GR
    # Sr here is a contravariant vector component
    # Srr here is a covariant tensor component

    ρ = (α.^2).*Ttt
    Sr = α.*Ttr
    Srr = ((γtrr.^2)./(χ.^2)).*Trr
    S = (γtrr.*Trr + 2*γtθθ.*Tθθ)./χ

    ∂tArr .+= -8*pi*α.*(χ.*Srr - (1/3)*S.*γtrr)
    ∂tK .+= 4*pi*α.*(ρ + S)
    ∂tΓr .+= -16*pi*α.*Sr./χ

    # Convert back to regularized variables

    # ∂tΓreg = -(r/2).*∂tΓr
    # ∂tγtθθreg = (1 ./r.^2).*∂tγtθθ

    ∂tγtθθreg = (1 ./r.^2).*∂tγtθθ
    ∂tArrreg = real((r .+ 0im).^(5/2)).*∂tArr
    ∂tKreg = real((r .+ 0im).^(3/2)).*∂tK

    # Numerical Dissipation terms

    ∂4α = dissipation(α)
    ∂4A = dissipation(A)
    ∂4βr = dissipation(βr)
    ∂4Br = dissipation(Br)
    ∂4χ = dissipation(χ)
    ∂4γtrr = dissipation(γtrr)
    ∂4γtθθreg = dissipation(γtθθreg)
    ∂4Arrreg = dissipation(Arrreg)
    ∂4Kreg = dissipation(Kreg)
    ∂4Γr = dissipation(Γr)
    ∂4𝜙 = dissipation(𝜙)
    ∂4K𝜙 = dissipation(K𝜙)

    #sign = -1 seems the best
    sign = -1
    σ = 0.3

    # ∂tα .+= (1/(2^6))*sign*σ*(drt^5)*∂6α
    # ∂tβr .+= (1/(2^6))*sign*σ*(drt^5)*∂6βr
    # ∂tBr .+= (1/(2^6))*sign*σ*(drt^5)*∂6Br
    # ∂tχ .+= (1/(2^6))*sign*σ*(drt^5)*∂6χ
    # ∂tγtrr .+= (1/(2^6))*sign*σ*(drt^5)*∂6γtrr
    # ∂tγtθθ .+= (1/(2^6))*sign*σ*(drt^5)*∂6γtθθ
    # ∂tArr .+= (1/(2^6))*sign*σ*(drt^5)*∂6Arr
    # ∂tK .+= (1/(2^6))*sign*σ*(drt^5)*∂6K
    # ∂tΓreg .+= (1/(2^6))*sign*σ*(drt^5)*∂6Γreg

    ∂tα .+= (1/(16))*sign*σ*(drt^3)*∂4α
    ∂tA .+= (1/(16))*sign*σ*(drt^3)*∂4A
    ∂tβr .+= (1/16)*sign*σ*(drt^3)*∂4βr
    ∂tBr .+= (1/16)*sign*σ*(drt^3)*∂4Br
    ∂tχ .+= (1/16)*sign*σ*(drt^3)*∂4χ
    ∂tγtrr .+= (1/16)*sign*σ*(drt^3)*∂4γtrr
    ∂tγtθθreg .+= (1/16)*sign*σ*(drt^3)*∂4γtθθreg
    ∂tArrreg .+= (1/16)*sign*σ*(drt^3)*∂4Arrreg
    ∂tKreg .+= (1/16)*sign*σ*(drt^3)*∂4Kreg
    ∂tΓr .+= (1/16)*sign*σ*(drt^3)*∂4Γr
    ∂t𝜙 .+= (1/16)*sign*σ*(drt^3)*∂4𝜙
    ∂tK𝜙 .+= (1/16)*sign*σ*(drt^3)*∂4K𝜙

    # Inner temporal boundary Conditions

    ∂tα[1:2] .= 0.
    ∂tA[1:2] .= 0.
    ∂tβr[1:2] .= 0.
    ∂tBr[1:2] .= 0.
    ∂tχ[1:2] .= 0.
    ∂tγtrr[1:2] .= 0.
    ∂tγtθθreg[1:2] .= 0.
    ∂tArrreg[1:2] .= 0.
    ∂tKreg[1:2] .= 0.
    ∂tΓr[1:2] .= 0.
    ∂t𝜙[1:2] .= 0.
    ∂tK𝜙[1:2] .= 0.

    # Outer temporal boundary conditions

    ∂tα[(n-1):n] .= 0.
    ∂tA[(n-1):n] .= 0.
    ∂tβr[(n-1):n] .= 0.
    ∂tBr[(n-1):n] .= 0.
    ∂tχ[(n-1):n] .= 0.
    ∂tγtrr[(n-1):n] .= 0.
    ∂tγtθθreg[(n-1):n] .= 0.
    ∂tArrreg[(n-1):n] .= 0.
    ∂tKreg[(n-1):n] .= 0.
    ∂tΓr[(n-1):n] .= 0.
    ∂t𝜙[(n-1):n] .= 0.
    ∂tK𝜙[(n-1):n] .= 0.

    # In case you want to freeze all GR variables

    # ∂tα[2:(n-1)] .= 0.
    # ∂tβr[2:(n-1)] .= 0.
    # ∂tBr[2:(n-1)] .= 0.
    # ∂tχ[2:(n-1)] .= 0.
    # ∂tγtrr[2:(n-1)] .= 0.
    # ∂tγtθθreg[2:(n-1)] .= 0.
    # ∂tArr[2:(n-1)] .= 0.
    # ∂tK[2:(n-1)] .= 0.
    # ∂tΓreg[2:(n-1)] .= 0.

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
    # ∂tα[(n-1):n] .= (α[(n-1):n] .- α0)./r[(n-1):n] - ∂α[(n-1):n] + hα./(r[(n-1):n].^w)
    # ∂tβr[(n-1):n] .= (βr[(n-1):n] .- βr0)./r[(n-1):n] - ∂βr[(n-1):n] + hβr./(r[(n-1):n].^w)
    # ∂tBr[(n-1):n] .= (Br[(n-1):n] .- Br0)./r[(n-1):n] - ∂Br[(n-1):n] + hBr./(r[(n-1):n].^w)
    # ∂tχ[(n-1):n] .= (χ[(n-1):n] .- χ0)./r[(n-1):n] - ∂χ[(n-1):n] + hχ./(r[(n-1):n].^w)
    # ∂tgrr[(n-1):n] .= (grr[(n-1):n] .- grr0)./r[(n-1):n] - ∂grr[(n-1):n] + hgrr./(r[(n-1):n].^w)
    # ∂tgθθreg[(n-1):n] .= (gθθreg[(n-1):n] .- gθθreg0)./r[(n-1):n] - ∂gθθreg[(n-1):n] + hgθθreg./(r[(n-1):n].^w)
    # ∂tArr[(n-1):n] .= (Arr[(n-1):n] .- Arr0)./r[(n-1):n] - ∂Arr[(n-1):n] + hArr./(r[(n-1):n].^w)
    # ∂tK[(n-1):n] .= (K[(n-1):n] .- K0)./r[(n-1):n] - ∂K[(n-1):n] + hK./(r[(n-1):n].^w)
    # ∂tΓreg[(n-1):n] .= (Γreg[(n-1):n] .- Γreg0)./r[(n-1):n] - ∂Γreg[(n-1):n] + hΓreg./(r[(n-1):n].^w)

    return GBSSN_Variables(∂tα,∂tA,∂tβr,∂tBr,∂tχ,∂tγtrr,∂tγtθθreg,∂tArrreg,∂tKreg,∂tΓr,∂t𝜙,∂tK𝜙)

end

function constraints(T,state::GBSSN_Variables,param)

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

    α = state.α
    βr = state.βr
    χ = state.χ
    γtrr = state.γtrr
    γtθθreg = state.γtθθ
    Arrreg = state.Arr
    Kreg = state.K
    Γr = state.Γr
    𝜙 = state.𝜙
    K𝜙 = state.K𝜙

    m = param[7]

    # Gauge conditions

    v = param[3]

    if v == 1 # Lagrangian Condition
        γtθθreg[2] = ((-315*γtθθreg[3] + 210*γtθθreg[4] - 126*γtθθreg[5]
        + 45*γtθθreg[6] - 7*γtθθreg[7])/63)
    end

    # Spatial Derivatives

    order = 4

    # First derivatives
    ∂rtχ = deriv(χ,order,1)
    ∂rtγtrr = deriv(γtrr,order,1)
    ∂rtγtθθreg = deriv(γtθθreg,order,1)
    ∂rtArrreg = deriv(Arrreg,order,1)
    ∂rtKreg = deriv(Kreg,order,1)
    ∂rt𝜙 = deriv(𝜙,order,1)

    # Second derivatives
    ∂2rtχ = deriv2(χ,order,1)
    ∂2rtγtθθreg = deriv2(γtθθreg,order,1)

    # Coordinate transformations from computational rt coordinate
    # to physical r coordinate

    r = sample(Float64, χ.grid, param[4])
    drdrt = sample(Float64, χ.grid, param[5])
    d2rdrt = sample(Float64, χ.grid, param[6])

    ∂χ = ∂rtχ./drdrt
    ∂γtrr = ∂rtγtrr./drdrt
    ∂γtθθreg = ∂rtγtθθreg./drdrt
    ∂Arrreg = ∂rtArrreg./drdrt
    ∂Kreg = ∂rtKreg./drdrt
    ∂𝜙 = ∂rt𝜙./drdrt

    ∂2χ = (∂2rtχ - d2rdrt.*∂χ)./(drdrt.^2)
    ∂2γtθθreg = (∂2rtγtθθreg - d2rdrt.*∂γtθθreg)./(drdrt.^2)

    ∂t𝜙 = βr.*∂𝜙 - 2*α.*K𝜙

    # Conversions from regularized variables to canonical variables

    γtθθ = (r.^2).*(γtθθreg .+ 1)
    ∂γtθθ = (r.^2).*∂γtθθreg + (2*r).*(γtθθreg .+ 1)
    ∂2γtθθ = (r.^2).*∂2γtθθreg + (4*r).*∂γtθθreg + 2*(γtθθreg .+ 1)

    K = real((r .+ 0im).^(-3/2)).*Kreg
    ∂K = real((r .+ 0im).^(-3/2)).*∂Kreg - (3/2)*real((r .+ 0im).^(-5/2)).*Kreg

    Arr = real((r .+ 0im).^(-5/2)).*Arrreg
    ∂Arr = real((r .+ 0im).^(-5/2)).*∂Arrreg - (5/2)*real((r .+ 0im).^(-7/2)).*Arrreg

    # Γr = -(2 ./r).*Γreg

    # Inverse metric (contravariant indices)

    gtt = -(1 ./α.^2)
    gtr = βr./α.^2
    grr = χ./γtrr - (βr./α).^2
    gθθ = χ./γtθθ

    # Lagrangian Density for scalar field

    𝓛 = (1/2)*gtt.*∂t𝜙.^2 + (1/2)*grr.*∂𝜙.^2 + gtr.*∂𝜙.*∂t𝜙 - (1/2)*(m^2)*𝜙.^2

    # Stresss Energy components (contravariant indices)

    Ttt = (gtt.*∂t𝜙 + gtr.*∂𝜙).^2 - gtt.*𝓛
    Ttr = (gtt.*∂t𝜙 + gtr.*∂𝜙).*(gtr.*∂t𝜙 + grr.*∂𝜙) - gtr.*𝓛

    # Source Terms to the constraints

    ρ = (α.^2).*Ttt
    Sr = α.*Ttr

    # Constraint Equations

    𝓗 = (-(3/2)*(Arr./γtrr).^2 + (2/3)*K.^2 - (5/2)*((∂χ.^2)./χ)./γtrr
     + 2*∂2χ./γtrr + 2*χ./γtθθ - 2*χ.*(∂2γtθθ./γtθθ)./γtrr + 2*∂χ.*(∂γtθθ./γtθθ)./γtrr
     + χ.*(∂γtrr./(γtrr.^2)).*(∂γtθθ./γtθθ) - ∂χ.*∂γtrr./(γtrr.^2)
     + (1/2)*χ.*((∂γtθθ./γtθθ).^2)./γtrr - 16*pi*ρ)

    𝓜r = (∂Arr./γtrr - (2/3)*∂K - (3/2)*Arr.*(∂χ./χ)./γtrr
     + (3/2)*Arr.*(∂γtθθ./γtθθ)./γtrr - Arr.*∂γtrr./(γtrr.^2)
     - 8*pi.*γtrr.*Sr./χ)

    𝓖r = -(1/2)*∂γtrr./(γtrr.^2) + Γr + (∂γtθθ./γtθθ)./γtrr

    # 𝓗[1:2] .= 0.
    # 𝓜r[1:2] .= 0.
    # 𝓖r[1:2] .= 0.

    return (𝓗, 𝓜r, 𝓖r, ρ)

end

function horizon(T,state::GBSSN_Variables,param)

    ############################################
    # Caculates the apparent horizon
    #
    # Where the function crosses zero is the
    # apparent horizon of the black hole.
    ############################################

    v = param[3]

    # Unpack Variables

    χ = state.χ
    γtrr = state.γtrr
    γtθθreg = state.γtθθ
    Arrreg = state.Arr
    Kreg = state.K

    # Gauge condition

    # if v == 1 # Lagrangian Condition
    #     γtθθreg[2] = ((-315*γtθθreg[3] + 210*γtθθreg[4] - 126*γtθθreg[5]
    #     + 45*γtθθreg[6] - 7*γtθθreg[7])/63)
    # end

    r = sample(T, χ.grid, param[4])
    drdrt = sample(T, χ.grid, param[5])

    # Conversions from regularized variables to canonical variables

    γtθθ = (r.^2).*(γtθθreg .+ 1)

    K = real((r .+ 0im).^(-3/2)).*Kreg

    Arr = real((r .+ 0im).^(-5/2)).*Arrreg

    # Intermediate calculations

    Kθθ = ((1/3)*γtθθ.*K - (1/2)*Arr.*γtθθ./γtrr)./χ

    grr =  γtrr./χ

    gθθ =  γtθθ./χ

    # Spatial Derivatives

    ∂rtgθθ = deriv(gθθ,4,1)

    # Coordinate transformations from computational rt coordinate
    # to physical r coordinate

    ∂gθθ = ∂rtgθθ./drdrt

    # Apparent horizon function

    Θ = (∂gθθ./gθθ)./real((grr .+ 0im).^(1/2)) - 2*Kθθ./gθθ

    # cross = GridFun(χ.grid, sign.(Θ))

    return Θ

end

# function crossings(Θ::GridFun)
#
#     GridFun(χ.grid, sign.(Θ))
#
#     for
#
# end


function custom_progress_message(dt,state,param,t)

    ###############################################
    # Outputs status numbers while the program runs
    ###############################################

    if param[1]==param[2]
        println("")
        println("| # | Time Step | Time | max α'(t) | max χ'(t) | max γtrr'(t) | max γtθθ'(t) | max Arr'(t) | max K'(t) | max Γr'(t) |")
        println("|___|___________|______|___________|___________|______________|______________|_____________|___________|____________|")
        println("")
    end

    derivstate = rhs(state,param,t)

    println("  ",
    rpad(string(param[1]),6," "),
    rpad(string(round(dt,digits=3)),10," "),
    rpad(string(round(t,digits=3)),10," "),
    rpad(string(round(maximum(abs.(derivstate.α)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(derivstate.χ)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(derivstate.γtrr)),digits=3)),14," "),
    rpad(string(round(maximum(abs.(derivstate.γtθθ)),digits=3)),14," "),
    rpad(string(round(maximum(abs.(derivstate.Arr)),digits=3)),12," "),
    # rpad(string(round(maximum(abs.(derivstate.𝜙)),digits=3)),12," "),
    # rpad(string(round(maximum(abs.(derivstate.K𝜙)),digits=3)),14," ")
    rpad(string(round(maximum(abs.(derivstate.K)),digits=3)),12," "),
    rpad(string(round(maximum(abs.(derivstate.Γr)),digits=3)),14," ")
    )

    #PrettyTables.jl

    param[1] += param[2]

end


function solution_saver(T,grid,sol,param,folder)

    ###############################################
    # Saves all of the variables in nice CSV files
    # in the choosen data folder directory
    ###############################################

    vars = (["α","A","βr","Br","χ","γtrr","γtθθ","Arr","K","Γreg","𝜙","K𝜙",
    "H","Mr","Gr","ρ","∂tα","∂tβr","∂tBr","∂tχ","∂tγtrr","∂tγtθθ",
    "∂tArr","∂tK","∂tΓreg","∂t𝜙","∂tK𝜙","appHorizon"])
    varlen = length(vars)
    #mkdir(string("data\\",folder))
    tlen = size(sol)[3]
    rlen = grid.ncells + 4
    loc = sample(T, grid, param[4])
    #loc[1] =
    cons = Array{GridFun,2}(undef,tlen,4)
    derivs = Array{GBSSN_Variables,1}(undef,tlen)
    apphorizon = Array{GridFun,1}(undef,tlen)

    for i in 1:tlen
        derivs[i] = rhs(sol[i],param,0)
        cons[i,1:4] .= constraints(T,sol[i],derivs[i],param)
        apphorizon[i] = horizon(T,sol[i],param)
    end


    array = Array{T,2}(undef,tlen+1,rlen+1)

    array[1,1] = 0
    array[1,2:end] .= loc

    for j = 1:varlen
        if j < 13
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                array[i,2:end] .= sol[:,j,i-1]
            end
        elseif j < 17
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                array[i,2:end] .= cons[i-1,j-12]
            end
        elseif j < 28
            for i = 2:tlen+1
                array[i,1] = sol.t[i-1]
                array[i,2:end] .= derivs[i-1][:,j-16]
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

    T = Float64
    rspan = T[1,210]
    rtspan = T[1,21]

    grid = setup(T, rtspan, points)
    drt = spacing(grid)
    dt = drt/4

    tspan = T[0,20]
    v = 1

    m = 0

    # f(b) = b*tan(rtspan[2]/b)-rspan[2]
    #
    # scale = find_zero(f, 0.64*rtspan[2])
    #
    # r(rt) = scale*tan(rt/scale)
    # drdrt(rt) = sec(rt/scale)^2
    # d2rdrt(rt) = (2/scale)*(sec(rt/scale)^2)*tan(rt/scale)

    r(rt) = rt
    drdrt(rt) = 1
    d2rdrt(rt) = 0

    atol = eps(T)^(T(3) / 4)
    alg = RK4()

    #printlogo()

    printtimes = 1
    custom_progress_step = round(Int, printtimes/dt)
    step_iterator = custom_progress_step
    param = [step_iterator, custom_progress_step, v, r, drdrt, d2rdrt, m]
    println("Defining Initial State...")
    #state = init(T, grid, param)::GBSSN_Variables
    println("Defining Problem...")
    #prob = ODEProblem(rhs, state, tspan, param)
    println("Starting Solution...")

    init(T, grid, param)

    # sol = solve(
    #     prob, alg,
    #     abstol = atol,
    #     dt = drt/4,
    #     adaptive = false,
    #     saveat = 1,
    #     progress = true,
    #     progress_steps=custom_progress_step,
    #     progress_message=custom_progress_message
    # )
    #
    # solution_saver(T,grid,sol,param,"ScalarTests")

end


end
