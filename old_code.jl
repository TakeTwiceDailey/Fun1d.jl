
# ∂tKrr[1]  = Krrrhs
# ∂tfrrr[1] = frrrrhs
# ∂tKθθ[1]  = Kθθrhs
# ∂tfrθθ[1] = frθθrhs

#Umr = Krr[1] - frrr[1]/sqrt(γrr[1])
#Upθ = Kθθ[1] + frθθ[1]/sqrt(γrr[1])

#∂rUmθ = ∂rKθθ[1] - ∂rfrθθ[1]/sqrt(γrr[1]) + frθθ[1]*∂rγrr[1]/(2*sqrt(γrr[1])^3)
#
# ∂rUpθ = (1/sqrt(γθθ[1]) - 1)*∂rγθθ[1]/Umθ[1] - (2*sqrt(γθθ[1]) - γθθ[1])*∂rUmθ/Umθ[1]^2

# cp = -βr[1] + α[1]/sqrt(γrr[1])
# cm = -βr[1] - α[1]/sqrt(γrr[1])
#
# ∂rcp = -∂rβr[1] + ∂rᾶ[1]*γθθ[1] + ᾶ[1]*∂rγθθ[1]
# ∂rcm = -∂rβr[1] - ∂rᾶ[1]*γθθ[1] - ᾶ[1]*∂rγθθ[1]
#
# ∂rUmθ = ∂rKθθ[1] - ∂rfrθθ[1]/sqrt(γrr[1]) + frθθ[1]*∂rγrr[1]/(2*sqrt(γrr[1])^3)
#
# ∂rUpθ = cm*∂rUmθ/cp + ∂rcm*Umθ[1]/cp - cm*Umθ[1]*∂rcp/cp^2
#
# #∂rUpθ = (q1 ⋅ Upθ[1:7])/dr̃/drdr̃[1]
# #∂rUmθ = ∂rKθθ[1] - ∂rfrθθ[1]/sqrt(γrr[1]) + frθθ[1]*∂rγrr[1]/(2*sqrt(γrr[1])^3)
# #∂rUpθ = (1/sqrt(γθθ[1]) - 1.)*∂rγθθ[1]/Umθ[1] - (2*sqrt(γθθ[1]) - γθθ[1])*∂rUmθ/Umθ[1]^2
# #∂rUpθ = ∂rKθθ[1] + ∂rfrθθ[1]/sqrt(γrr[1]) - frθθ[1]*∂rγrr[1]/(2*sqrt(γrr[1])^3)
# #∂rUpθ = (-25*Upθ[1] + 48*Upθ[2] - 36*Upθ[3] + 16*Upθ[4] - 3*Upθ[5])/(12*dr̃)/drdr̃[1]
# #∂rUpθ = (-137*Upθ[1] + 300*Upθ[2] - 300*Upθ[3] + 200*Upθ[4] - 75*Upθ[5] + 12*Upθ[6])/(60*dr̃)/drdr̃[1]
#
# Upr = -Umr - γrr[1]*Upθ[1]/γθθ[1] + (2*∂rUpθ*sqrt(γrr[1]) - γrr[1])/Upθ[1]
#
# ∂tKrr[1]  += s*( (Upr - Krr[1])/2 - frrr[1]/sqrt(γrr[1])/2 )/(dr̃*σ00)
# ∂tfrrr[1] += s*( (Upr - Krr[1])*sqrt(γrr[1])/2 - frrr[1]/2 )/(dr̃*σ00)

# dtU0r = 0.
# dtU0θ = 0.

#dtUmr = ∂tKrr[n] - ∂tfrrr[n]/sqrt(γrr[n]) + frrr[n]*∂tγrr[n]/2/sqrt(γrr[n])^3
# ∂tUmr = 0.
#
# ∂tKrr[n]  += s*(  (Umr - Krr[n])/2 + frrr[n]/sqrt(γrr[n])/2 )/(dr̃*σ00)
# ∂tfrrr[n] += s*( -(Umr - Krr[n])*sqrt(γrr[n])/2 - frrr[n]/2 )/(dr̃*σ00)
#
# ∂tγrr[n]  += -s*( ∂rγrr[n] + 8*frθθ[n]*γrr[n]/γθθ[n] - 2*frrr[n] )/(dr̃*σ00)/500
# ∂tγθθ[n]  += -s*( ∂rγθθ[n] - 2*frθθ[n] )/(dr̃*σ00)/500
#
# ∂rUmθ = ∂rKθθ[n] - ∂rfrθθ[n]/sqrt(γrr[n]) + frθθ[n]*∂rγrr[n]/(2*sqrt(γrr[n])^3)
#
# con = (2*∂rUmθ + (Umr + Upr)*Umθ/sqrt(γrr[n]) + (1. + Umθ^2/γθθ[n])*sqrt(γrr[n]))/500
#
# ∂tKθθ[n]  += -s*( con/2 )/(dr̃*σ00)
# ∂tfrθθ[n] += -s*( -con*sqrt(γrr[n])/2 )/(dr̃*σ00)

# ∂tKθθ[n]  = ∂tUmθ/2 + Kθθrhs/2 + frθθrhs/sqrt(γrr[n])/2
# ∂tfrθθ[n] = frθθrhs/2 - ∂tUmθ*sqrt(γrr[n])/2 + Kθθrhs*sqrt(γrr[n])/2

# @. Upθ = Kθθ + frθθ/sqrt(γrr)
#
# @. Umθ = ((-βr + α/sqrt(γrr))/(-βr - α/sqrt(γrr)))*Upθ
# #@. Umθ = Kθθ - frθθ/sqrt(γrr)
#
# cp = -βr[n] + α[n]/sqrt(γrr[n])
# cm = -βr[n] - α[n]/sqrt(γrr[n])
#
# ∂rcp = -∂rβr[n] + ∂rᾶ[n]*γθθ[n] + ᾶ[n]*∂rγθθ[n]
# ∂rcm = -∂rβr[n] - ∂rᾶ[n]*γθθ[n] - ᾶ[n]*∂rγθθ[n]
#
# ∂rUpθ = ∂rKθθ[n] + ∂rfrθθ[n]/sqrt(γrr[n]) - frθθ[n]*∂rγrr[n]/(2*sqrt(γrr[n])^3)
# #∂rUmθ = ∂rKθθ[n] - ∂rfrθθ[n]/sqrt(γrr[n]) + frθθ[n]*∂rγrr[n]/(2*sqrt(γrr[n])^3)
#
# ∂rUmθ = cp*∂rUpθ/cm + ∂rcp*Upθ[n]/cm - cp*Upθ[n]*∂rcm/cm^2
#
# #∂rUmθ = ∂rKθθ[n] - ∂rfrθθ[n]/sqrt(γrr[n]) + frθθ[n]*∂rγrr[n]/(2*sqrt(γrr[n])^3)
#
# Upr = Krr[n] + frrr[n]/sqrt(γrr[n])
#
# Umr = -Upr - γrr[n]*Umθ[n]/γθθ[n] - (2*∂rUmθ*sqrt(γrr[n]) + γrr[n])/Umθ[n]
#
# # U0r = (2*frrr[n]-∂rγrr[n])*γθθ[n]/frθθ[n]/8
# # U0θ = 8*γrr[n]*frθθ[n]/(2*frrr[n]-∂rγrr[n])
#
# U0r = γrr[n]
# U0θ = γθθ[n]

#Umθ[n] = Kθθi[n] - frθθi[n]/sqrt(γrri[n])

#Umr = Krri[n] - frrri[n]/sqrt(γrri[n])

# ∂tKθθ[n]  += s*(  (Umθ[n] - Kθθ[n])/2 + frθθ[n]/sqrt(γrr[n])/2 )/(dr̃*σ00)
# ∂tfrθθ[n] += s*( -(Umθ[n] - Kθθ[n])*sqrt(γrr[n])/2 - frθθ[n]/2 )/(dr̃*σ00)
# #+ frθθ[n]*U0r/γrr[n]/2
#
#
# # ∂tγrr[n]  += s*( U0r - γrr[n] )/(dr̃*σ00)
# # ∂tγθθ[n]  += s*( U0θ - γθθ[n] )/(dr̃*σ00)
#
# ∂tγrr[n] = (2*frrr[n] - 8*frθθ[n]*γrr[n]/γθθ[n])*βr[n] + 2*∂rβr[n]*γrr[n] - 2*α[n]*Krr[n]
# ∂tγθθ[n] = 0.
# #∂tγθθ[n] = 2*frθθ[n]*βr[n] - 2*α[n]*Kθθ[n]
#
# ∂tKrr[n]  += s*(  (Umr - Krr[n])/2 + frrr[n]/sqrt(γrr[n])/2 )/(dr̃*σ00)
# ∂tfrrr[n] += s*( -(Umr - Krr[n])*sqrt(γrr[n])/2 - frrr[n]/2 )/(dr̃*σ00)
# + frrr[n]*U0r/γrr[n]/2

# @inline function deriv!(df::Vector{T}, f::Vector{T}, n::Int64, dx::T) where T
#
#     #######################################################
#     # Calculates derivatives using a 4th order SBP operator
#     #######################################################
#
#     # @inbounds @fastmath @simd
#
#     df[1] = (-48*f[1] + 59*f[2] - 8*f[3] - 3*f[4])/(34*dx)
#
#     df[2] = (-f[1] + f[3])/(2*dx)
#
#     df[3] = (8*f[1] - 59*f[2] + 59*f[4] - 8*f[5])/(86*dx)
#
#     df[4] = (3*f[1] - 59*f[3] + 64*f[5] - 8*f[6])/(98*dx)
#
#     for i in 5:(n - 4)
#         df[i] = (f[i-2] - 8*f[i-1] + 8*f[i+1] - f[i+2])/(12*dx)
#     end
#
#     df[n-3] = -(3*f[n] - 59*f[n-2] + 64*f[n-4] - 8*f[n-5])/(98*dx)
#
#     df[n-2] = -(8*f[n] - 59*f[n-1] + 59*f[n-3] - 8*f[n-4])/(86*dx)
#
#     df[n-1] = -(-f[n] + f[n-2])/(2*dx)
#
#     df[n] = -(-48*f[n] + 59*f[n-1] - 8*f[n-2] - 3*f[n-3])/(34*dx)
#
# end

#########################


# fᾶ(M,r,r̃) = 1
# f∂rᾶ(M,r,r̃) = 0
# f∂r2ᾶ(M,r,r̃) = 0
#
# fβr(M,r,r̃) = 0
# f∂rβr(M,r,r̃) = 0
# f∂r2βr(M,r,r̃) = 0
#
# fγrr(M,r,r̃) = 1
# f∂rγrr(M,r,r̃) = 0
#
# fγθθ(M,r,r̃) = 1
# f∂rγθθ(M,r,r̃) = 0
#
# fKrr(M,∂M,r,r̃) = 0
# f∂rKrr(M,r,r̃) = 0
#
# fKθθ(M,r,r̃) = 0
# f∂rKθθ(M,r,r̃) = 0
#
# ffrrr(M,∂M,r,r̃) = 0
# f∂rfrrr(M,r,r̃) = 0
#
# ffrθθ(M,r,r̃) = 0
# f∂rfrθθ(M,r,r̃) = 0


# function M_init(::Type{T}, grid::Grid, param) where {T}
#
#     ############################################
#     # Specifies the Initial Conditions
#     ############################################
#
#     n = grid.ncells + 4
#     domain = grid.domain
#     initgrid = grid
#     drt = spacing(grid)
#     r = param[4]
#     drdrt = param[5]
#     d2rdrt = param[6]
#     m = param[7]
#     rtspan = param[8]
#
#     num = 0
#
#     fρ(M,rt) = (2*fK𝜙(rt)^2 + (1/2)*(fχ(rt)/fγtrr(M,rt))*f∂𝜙(rt)^2
#         + (1/2)*m^2*f𝜙(rt)^2)
#
#     fSr(rt) = 2*fK𝜙(rt)*f∂𝜙(rt)
#
#     #f∂M(M,rt) = 4*pi*(r(rt)^2)*fρ(M,rt)
#
#     function f∂M(M,rt)
#          if rt < 2
#              return 0.
#          else
#              4*pi*(r(rt)^2)*fρ(M,rt)
#          end
#     end
#
#     function f𝓗(M,∂M,rt)
#          (-(3/2)*(fArr(M,∂M,rt)/fγtrr(M,rt))^2 + (2/3)*fK(M,∂M,rt)^2
#          - (5/2)*((f∂χ(rt)^2)/fχ(rt))/fγtrr(M,rt) + 2*f∂2χ(rt)/fγtrr(M,rt)
#          + 2*fχ(rt)/fγtθθ(rt) - 2*fχ(rt)*(f∂2γtθθ(rt)/fγtθθ(rt))/fγtrr(M,rt)
#          + 2*f∂χ(rt)*(f∂γtθθ(rt)/fγtθθ(rt))/fγtrr(M,rt)
#          + fχ(rt)*(f∂γtrr(M,∂M,rt)/(fγtrr(M,rt)^2))*(f∂γtθθ(rt)/fγtθθ(rt))
#          - f∂χ(rt)*f∂γtrr(M,∂M,rt)/(fγtrr(M,rt)^2)
#          + (1/2)*fχ(rt)*((f∂γtθθ(rt)/fγtθθ(rt))^2)/fγtrr(M,rt) - 16*pi*fρ(M,rt))
#     end
#
#     fαreg(M,rt) = 2*M
#     fγtrrreg(M,rt) = 2*M
#     fArrreg(M,∂M,rt) = real((r(rt)+ 0im)^(5/2))*fArr(M,∂M,rt)
#     fKreg(M,∂M,rt) = real((r(rt)+ 0im)^(3/2))*fK(M,∂M,rt)
#
#     # Constraint Equations
#
#     rtspan = (rtspan[1], rtspan[2])
#     #rtspan = (rtspan[2], 0.5)
#
#     function constraintSystem(M, param, rt)
#         f∂M(M,rt)
#     end
#
#     # function boundaryCondition!(residual, M, param, rt)
#     #     residual = M[end] - 1. #inner boundary condition
#     # end
#
#     atol = 1e-15
#
#     BVP = ODEProblem(constraintSystem, 1., rtspan, param)
#     M = solve(BVP, Tsit5(), abstol=atol, dt=drt, adaptive=false)
#
#     ∂M(rt) = f∂M(M(rt),rt)
#
#     # M(rt) = 1.
#     # ∂M(rt) = 0
#
#
#     state = GridFun(grid, M)
#
#     return
#
# end
#
# function M_rhs(M::GridFun, param, t)
#
#     #########################################################
#     # Source Terms and Source Evolution
#     #
#     # This currently includes the addition of source terms
#     # to GR that come from a Klein-Gordon scalar field
#     #
#     #########################################################
#
#     # Klein-Gordon System
#
#     ∂t𝜙 = βr.*∂𝜙 - 2*α.*K𝜙
#     ∂tK𝜙 = (βr.*∂K𝜙 + α.*K.*K𝜙 - (1/2)*α.*χ.*∂2𝜙./γtrr
#         + (1/4)*α.*χ.*∂γtrr.*∂𝜙./γtrr.^2 - (1/4)*α.*∂χ.*∂𝜙./γtrr
#         - (1/2)*χ.*∂α.*∂𝜙./γtrr - (1/2)*χ.*∂γtθθ.*∂𝜙./(γtrr.*γtθθ)
#         + (1/2)*∂χ.*∂𝜙./(γtrr) + (1/2)*m^2*𝜙)
#
#     ρ = 2*K𝜙.^2 + (1/2)*(χ./γtrr).*∂𝜙.^2 + (1/2)*m^2*𝜙.^2
#     #Lower Index
#     Sr = 2*γtrr.*K𝜙.*∂𝜙./χ
#     # S = 6*K𝜙.^2 - (1/2)*(χ./γtrr).*∂𝜙.^2 - (3/2)*m^2*𝜙.^2
#     # Srr = (γtrr./χ).*(2*K𝜙.^2 + (1/2)*(χ./γtrr).*∂𝜙.^2 - (1/2)*m^2*𝜙.^2)
#
#     # ∂tArr .+= -8*pi*α.*(χ.*Srr - (1/3)*S.*γtrr)
#     # ∂tK .+= 4*pi*α.*(ρ + S)
#     # ∂tΓtr .+= -16*pi*α.*Sr./γtrr
#
#     # Inner temporal boundary Conditions
#
#     # ∂tα[1:2] .= 0.
#     # ∂tA[1:2] .= 0.
#     # ∂tβr[1:2] .= 0.
#     # ∂tBr[1:2] .= 0.
#     # ∂tχ[1:2] .= 0.
#     # ∂tγtrr[1:2] .= 0.
#     # ∂tγtθθ[1:2] .= 0.
#     # ∂tArr[1:2] .= 0.
#     # ∂tK[1:2] .= 0.
#     # ∂tΓtr[1:2] .= 0.
#     # ∂t𝜙[1:2] .= 0.
#     # ∂tK𝜙[1:2] .= 0.
#
#     return GBSSN_Variables(∂tαreg,∂tA,∂tβr,∂tBr,∂tχ,∂tγtrrreg,∂tγtθθreg,∂tArrreg,∂tKreg,∂tΓtr,∂t𝜙,∂tK𝜙)
#
# end

# @. ∂tArr += -8*pi*α*(χ*Srr - (1/3)*S*γtrr)
# @. ∂tK += 4*pi*α*(ρ + S)
# @. ∂tΓtr += -16*pi*α*Sr/γtrr

# fr = param.r
#
# fα(M,rt) = real((1+2*M/(fr(rt))+0im)^(-1/2))
# fβr(M,rt) = (2*M/fr(rt))*fα(M,rt)^2
# fγtrr(M,rt) = 1+2*M/fr(rt)
# fγtθθ(rt) = fr(rt)^2
# fArr(M,∂M,rt) = (4/3)*(fr(rt)*(M+fr(rt))*∂M-M*(3*M+2*fr(rt)))/real(((fr(rt)^5)*(fr(rt)+2*M)+0im)^(1/2))
# fK(M,∂M,rt) = (2*M*(3*M+fr(rt))+2*fr(rt)*∂M*(M+fr(rt)))/real((fr(rt)*(fr(rt)+2*M)+0im)^(3/2))
# fΓtr(M,∂M,rt) = (fr(rt)*∂M-2*fr(rt)-5*M)/(fr(rt)+2*M)^2
#
# f∂α(M,rt) = M*real((fr(rt)*(fr(rt)+2*M+0im)^3)^(-1/2))
# f∂βr(M,rt) = -2*M/(fr(rt)+2*M)^2
# f∂γtrr(M,rt) = -2*M/(fr(rt)^2)
# f∂γtθθ(rt) = 2*fr(rt)
# f∂Arr(M,rt) = (4*M/3)*(15*M^2+15*M*fr(rt)+4*fr(rt)^2)/real(((fr(rt)^7)*((fr(rt)+2*M)^3)+0im)^(1/2))
# f∂K(M,rt) = -2*M*(9*M^2+10*M*fr(rt)+2*fr(rt)^2)/real((fr(rt)*(fr(rt)+2*M)+0im)^(5/2))
# f∂Γtr(M,rt) = 2*(fr(rt)+3*M)/(fr(rt)+2*M)^3
#
# rt = sample(Float64, A.grid, rt->rt)
#
# # for i=1:2
#
# ∂tα[1:2] .= (α[1:2] .- fα.(1.,rt[1:2]))./r[1:2] + ∂α[1:2] - f∂α.(1.,rt[1:2])
# ∂tA[1:2] .= (A[1:2] .- 0.)./r[1:2] + ∂A[1:2]
# ∂tβr[1:2] .= (βr[1:2] .- fβr.(1.,rt[1:2]))./r[1:2] + ∂βr[1:2] - f∂βr.(1.,rt[1:2])
# ∂tBr[1:2] .= (Br[1:2] .- 0.)./r[1:2] + ∂Br[1:2]
# ∂tχ[1:2] .= (χ[1:2] .- 1.)./r[1:2] + ∂χ[1:2]
# ∂tγtrr[1:2] .= (γtrr[1:2] .- fγtrr.(1.,rt[1:2]))./r[1:2] + ∂γtrr[1:2] - f∂γtrr.(1.,rt[1:2])
# ∂tγtθθ[1:2] .= (γtθθ[1:2] .- fγtθθ.(rt[1:2]))./r[1:2] + ∂γtθθ[1:2] - f∂γtθθ.(rt[1:2])
# ∂tArr[1:2] .= (Arr[1:2] .- fArr.(1.,0.,rt[1:2]))./r[1:2] + ∂Arr[1:2] - f∂Arr.(1.,rt[1:2])
# ∂tK[1:2] .= (K[1:2] .- fK.(1.,0.,rt[1:2]))./r[1:2] + ∂K[1:2] - f∂K.(1.,rt[1:2])
# ∂tΓtr[1:2] .= (Γtr[1:2] .- fΓtr.(1.,0.,rt[1:2]))./r[1:2] + ∂Γtr[1:2] - f∂Γtr.(1.,rt[1:2])
# ∂t𝜙[1:2] .= (𝜙[1:2] .- 0.)./r[1:2] + ∂𝜙[1:2]
# ∂tK𝜙[1:2] .= (K𝜙[1:2] .- 0.)./r[1:2] + ∂K𝜙[1:2]

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
# equation since it contains a ∂tΓtr term.
#
#########################################################

# Lagrangian Gauge condition
# v = 1.
#
# @. ∂tχ = ((2/3)*K*α*χ - (1/3)*v*βr*χ*∂γtrr/γtrr - (2/3)*v*βr*χ*∂γtθθ/γtθθ
#  - (2/3)*v*χ*∂βr + βr*∂χ)
#
# @. ∂tγtrr = (-2*Arr*α - (1/3)*v*βr*∂γtrr + βr*∂γtrr
#  - (2/3)*v*γtrr*βr*∂γtθθ/γtθθ + 2*γtrr*∂βr - (2/3)*v*γtrr*∂βr)
#
# @. ∂tγtθθ = (Arr*γtθθ*α/γtrr - (1/3)*v*γtθθ*βr*∂γtrr/γtrr - (2/3)*v*βr*∂γtθθ
#  + βr*∂γtθθ - (2/3)*v*γtθθ*∂βr)
#
# @. ∂tArr = (-2*α*(Arr^2)/γtrr + K*α*Arr - (1/3)*v*βr*Arr*∂γtrr/γtrr
#  - (2/3)*v*βr*Arr*∂γtθθ/γtθθ - (2/3)*v*Arr*∂βr + 2*Arr*∂βr
#  + (2/3)*α*χ*(∂γtrr/γtrr)^2 - (1/3)*α*χ*(∂γtθθ/γtθθ)^2
#  - (1/6)*α*(∂χ^2)/χ - (2/3)*α*χ*γtrr/γtθθ + βr*∂Arr
#  + (2/3)*α*χ*γtrr*∂Γtr - (1/2)*α*χ*(∂γtrr/γtrr)*(∂γtθθ/γtθθ)
#  + (1/3)*χ*∂γtrr*∂α/γtrr + (1/3)*χ*∂α*∂γtθθ/γtθθ - (1/6)*α*∂γtrr*∂χ/γtrr
#  - (1/6)*α*∂γtθθ*∂χ/γtθθ - (2/3)*∂α*∂χ - (1/3)*α*χ*∂2γtrr/γtrr
#  + (1/3)*α*χ*∂2γtθθ/γtθθ - (2/3)*χ*∂2α + (1/3)*α*∂2χ)
#
# @. ∂tK = ((3/2)*α*(Arr/γtrr)^2 + (1/3)*α*K^2 + βr*∂K
#  + (1/2)*χ*∂γtrr*∂α/(γtrr^2) - χ*∂α*(∂γtθθ/γtθθ)/γtrr
#  + (1/2)*∂α*∂χ/γtrr - χ*∂2α/γtrr)
#
# @. ∂tΓtr = (-v*βr*((∂γtθθ/γtθθ)^2)/γtrr + α*Arr*(∂γtθθ/γtθθ)/(γtrr^2)
#  - (1/3)*v*∂βr*(∂γtθθ/γtθθ)/γtrr + ∂βr*(∂γtθθ/γtθθ)/γtrr
#  + βr*∂Γtr + α*Arr*∂γtrr/(γtrr^3) - (4/3)*α*∂K/γtrr
#  - 2*Arr*∂α/(γtrr^2) + (1/2)*v*∂βr*∂γtrr/(γtrr^2)
#  - (1/2)*∂βr*∂γtrr/(γtrr^2) - 3*α*Arr*(∂χ/χ)/(γtrr^2)
#  + (1/6)*v*βr*∂2γtrr/(γtrr^2) + (1/3)*v*βr*(∂2γtθθ/γtθθ)/γtrr
#  + (1/3)*v*∂2βr/γtrr + ∂2βr/γtrr)

# Gauge choices for the evolution of the
# determinant of the conformal metric
# (must have v = 1 to use this)

# ∂tlnγt = temp.x[5]
# ∂rt∂tlnγt = temp.x[6]
#
# ∂tlnγt .= 0
#
# #∂tlnγt = -8*pi*Sr.*real((γtθθ./γtrr .+ 0im).^(1/2))
#
# deriv!(∂rt∂tlnγt,∂tlnγt,n,drt)
#
# @. ∂r∂tlnγt = ∂rt∂tlnγt/drdrt
#
# # ∂tα = -(1/2)*α.*∂tlnγt
# # ∂tβr = (χ./γtrr).*∂tlnγt
#
# @. ∂tχ += (1/3)*χ*∂tlnγt
# @. ∂tγtrr += (1/3)*γtrr*∂tlnγt
# @. ∂tγtθθ += (1/3)*γtθθ*∂tlnγt
# @. ∂tArr += (1/3)*Arr*∂tlnγt
# @. ∂tΓtr += -(1/3)*Γtr*∂tlnγt - (1/6)*(χ/γtrr)*∂r∂tlnγt
# s = 1
#
# fα(M,r,rt) = real((1+2*M/(r(rt))+0im)^(-1/2))
# f∂α(M,r,rt) = (M/r(rt)^2)*fα(M,r,rt)^3
# f∂2α(M,r,rt) = -(M/r(rt)^4)*(M+2*r(rt))*fα(M,r,rt)^5
#
# fβr(M,r,rt) = s*(2*M/r(rt))*fα(M,r,rt)^2
# f∂βr(M,r,rt) = -s*2*M/(r(rt)+2*M)^2
# f∂2βr(M,r,rt) = s*4*M/(r(rt)+2*M)^3
#
# fχ(M,r,rt) = 1.
# f∂χ(M,r,rt) = 0.
# f∂2χ(M,r,rt) = 0.
#
# fγtrr(M,r,rt) = 1 + 2*M/r(rt)
# f∂γtrr(M,r,rt) = -2*M/r(rt)^2
# f∂2γtrr(M,r,rt) = 4*M/r(rt)^3
#
# fγtθθ(M,r,rt) = r(rt)^2
# f∂γtθθ(M,r,rt) = 2*r(rt)
# f∂2γtθθ(M,r,rt) = 2
#
# fK(M,r,rt) = s*(2*M/r(rt)^3)*(3*M+r(rt))*fα(M,r,rt)^3
# f∂K(M,r,rt) = -s*(2*M/r(rt)^5)*(9*M^2+10*M*r(rt)+2*r(rt)^2)*fα(M,r,rt)^5
#
# fArr(M,r,rt) = -s*(4/3)*(M/r(rt)^3)*(2*r(rt)+3*M)*fα(M,r,rt)
# f∂Arr(M,r,rt) = s*(4/3)*(M/r(rt)^5)*(15*M^2+15*M*r(rt)+4*r(rt)^2)*fα(M,r,rt)^3
#
# fΓtr(M,r,rt) = -(5*M+2*r(rt))/(r(rt)+2*M)^2
# f∂Γtr(M,r,rt) = 2*(r(rt)+3*M)/(r(rt)+2*M)^3
######################################################
# @inline function deriv2!(df::Vector{T}, f::Vector{T}, n::Int64, dx::T) where T
#
#     # @inbounds @fastmath @simd
#
#     df[1] = (2*f[1] - 5*f[2] + 4*f[3] - f[4])/(dx^2)
#
#     df[2] = (f[1] - 2*f[2] + f[3])/(dx^2)
#
#     df[3] = (-4*f[1] + 59*f[2] - 110*f[3] + 59*f[4] - 4*f[5])/(43*dx^2)
#
#     df[4] = (-f[1] + 59*f[3] - 118*f[4] + 64*f[5] - 4*f[6])/(49*dx^2)
#
#     for i in 5:(n - 4)
#         df[i] = (-f[i-2] + 16*f[i-1] - 30*f[i] + 16*f[i+1] - f[i+2])/(12*dx^2)
#     end
#
#     df[n-3] = (-f[n] + 59*f[n-2] - 118*f[n-3] + 64*f[n-4] - 4*f[n-5])/(49*dx^2)
#
#     df[n-2] = (-4*f[n] + 59*f[n-1] - 110*f[n-2] + 59*f[n-3] - 4*f[n-4])/(43*dx^2)
#
#     df[n-1] = (f[n] - 2*f[n-1] + f[n-2])/(dx^2)
#
#     df[n] = (2*f[n] - 5*f[n-1] + 4*f[n-2] - f[n-3])/(dx^2)
#
# end
