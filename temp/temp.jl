module temp

using DifferentialEquations
using BoundaryValueDiffEq
using OrdinaryDiffEq
using RecursiveArrayTools

using BenchmarkTools
using InteractiveUtils
using RecursiveArrayTools

struct Param{S}
    state::S
    dtstate2::S
end

@inline function Base.zeros(::Type{ArrayPartition},::Type{T},n::Int) where T
    return ArrayPartition(
        zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),
        zeros(n),zeros(n),zeros(n),zeros(n),zeros(n),zeros(n)
    )
end

function rhs!(dtstate::S, regstate::S, param::Param{S}, t) where S

    #Unpack the parameters

    dtstate2 = param.dtstate2
    state = param.state

    ## Store the current state in the container
    ## so it can be modified during the calculations
    # for i in 1:12
    #     state.x[i] .= regstate.x[i]
    # end
    state .= regstate

    # Unpack the states so I can refer to individual things
    a,b,c,d,e,f,g,h,i,j,k,l = state.x

    A,B,C,D,E,F,G,H,I,J,K,L = dtstate.x

    ## Do all your calculations here....
    ## here is an example

    @. A = a + b + c + d + e + f + g + h + i + j + k + l
    @. B = a + b + c + d + e + f + g + h + i + j + k + l
    @. C = a + b + c + d + e + f + g + h + i + j + k + l
    @. D = a + b + c + d + e + f + g + h + i + j + k + l
    @. E = a + b + c + d + e + f + g + h + i + j + k + l
    @. F = a + b + c + d + e + f + g + h + i + j + k + l
    @. G = a + b + c + d + e + f + g + h + i + j + k + l
    @. H = a + b + c + d + e + f + g + h + i + j + k + l
    @. I = a + b + c + d + e + f + g + h + i + j + k + l
    @. J = a + b + c + d + e + f + g + h + i + j + k + l
    @. K = a + b + c + d + e + f + g + h + i + j + k + l
    @. L = a + b + c + d + e + f + g + h + i + j + k + l


    ## Store the current time derivatives in the container
    # for use elsewhere
    # for i in 1:12
    #     dtstate2.x[i] .= dtstate.x[i]
    # end
    dtstate2 .= dtstate



end

function main()

    n = 4000

    numvar = 12

    T = Float64

    tspan = T[0.,1.]

    cont = ArrayPartition{T, NTuple{numvar, Vector{T}}}

    regstate = zeros(ArrayPartition,T,n)::cont
    state = zeros(ArrayPartition,T,n)::cont
    dtstate = zeros(ArrayPartition,T,n)::cont

    param = Param(state,dtstate)

    prob = ODEProblem(rhs!, regstate, tspan, param)

    atol = eps(T)^(T(3) / 4)

    alg = RK4()

    # @btime sol = solve(
    #     $prob, $alg,
    #     abstol = $atol,
    #     dt = 0.1,
    #     adaptive = false,
    #     saveat = 0.1,
    #     alias_u0 = true
    #     # progress = true,
    #     # progress_steps = custom_progress_step,
    #     # progress_message = custom_progress_message
    # )

    @code_warntype rhs!(dtstate,regstate,param,0.)

    return

end

end
