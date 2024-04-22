module GR3D

const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using CellArrays

ParallelStencil.@reset_parallel_stencil()

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using Plots, Printf, Statistics, BenchmarkTools, ForwardDiff

using Tensorial, Roots

# Alias for SymmetricSecondOrderTensor 2x2
const TwoTensor{T} = SymmetricSecondOrderTensor{2,T,3}

# Alias for non-symmetric 3 tensor
const ThreeTensor{T} = SecondOrderTensor{3,T,9}

# Alias for non-symmetric 3 tensor
const StateTensor = SymmetricSecondOrderTensor{4,Data.Number,10}

# Alias for StateVector of a scalar field
const StateScalar{T} = Vec{4,T}

# Alias for tensor to hold metric derivatives and Christoffel Symbols
# Defined to be symmetric in the last two indices
const Symmetric3rdOrderTensor{T} = Tensor{Tuple{3,@Symmetry{3,3}},T,3,18}


@CellType WaveCell fieldnames=(ψ,ψx,ψy,ψz,Ψ)

# const q11 = -24/17.; const q21 = 59/34. ;
# const q31 = -4/17. ; const q41 = -3/34. ;
# const q51 = 0.     ; const q61 =  0.    ;

# const q12 = -1/2.  ; const q22 = 0.     ;
# const q32 = 1/2.   ; const q42 = 0.     ;
# const q52 = 0.     ; const q62 = 0.     ;

# const q13 =  4/43. ; const q23 = -59/86.;
# const q33 =  0.    ; const q43 = 59/86. ;
# const q53 =  -4/43.; const q63 = 0.     ;

# const q14 = 3/98.  ; const q24 = 0.     ;
# const q34 = -59/98.; const q44 = 0.     ;
# const q54 = 32/49. ; const q64 = -4/49. ;

##################################################################
# Coefficent functions for fourth order diagonal norm 
# embedded boundary SBP operator

# Boundary interpolation coefficents
@inline el1(a) = (a+2)*(a+1)/2
@inline el2(a) = -a*(a+2)
@inline el3(a) = a*(a+1)/2

# Norm coefficents
@inline h11(a) = 17/48 + a + 11/12*a^2 + 1/3*a^3 + 1/24*a^4
@inline h22(a) = 59/48 - 3/2*a^2 - 5/6*a^3 - 1/8*a^4
@inline h33(a) = 43/48 + 3/4*a^2 + 2/3*a^3 + 1/8*a^4
@inline h44(a) = 49/48 - 1/6*a^2 - 1/6*a^3 - 1/24*a^4

# Q + Q^T = 0 coefficents
@inline Q12(a) = 7/12*a^2 + a + 1/48*a^4 + 1/6*a^3 + 59/96 
@inline Q13(a) = -1/12*a^4 - 5/12*a^3 - 7/12*a^2 - 1/4*a - 1/12 
@inline Q14(a) = 1/16*a^4 + 1/4*a^3 + 1/4*a^2 - 1/32

@inline Q23(a) = 3/16*a^4 + 5/6*a^3 + 3/4*a^2 + 59/96
@inline Q24(a) = -1/6*a^2*(a + 2)^2

@inline Q34(a) = 5/48*a^4 + 5/12*a^3 + 5/12*a^2 + 59/96

# Finite difference coefficents 
# (I have kept the norm) part out for speed
@inline q11(a) = -el1(a)^2/2
@inline q12(a) = Q12(a) - el1(a)*el2(a)/2
@inline q13(a) = Q13(a) - el1(a)*el3(a)/2
@inline q14(a) = Q14(a)

@inline q21(a) = -Q12(a) - el1(a)*el2(a)/2
@inline q22(a) = -el2(a)^2/2
@inline q23(a) = Q23(a) - el2(a)*el3(a)/2
@inline q24(a) = Q24(a)

@inline q31(a) = -Q13(a) - el1(a)*el3(a)/2
@inline q32(a) = -Q23(a) - el2(a)*el3(a)/2
@inline q33(a) = -el3(a)^2/2
@inline q34(a) = Q34(a)

@inline q41(a) = -Q14(a)
@inline q42(a) = -Q24(a)
@inline q43(a) = -Q34(a)

##################################################################
# Coefficent functions for second order diagonal norm 
# embedded boundary SBP operator

@inline el2_1(a) = (a+1)
@inline el2_2(a) = -a

@inline h2_11(a) = (a + 1)^2/2
@inline h2_22(a) = 1 - a^2/2

@inline Q2_12(a) = (a+1)/2

@inline q2_21(a) = -Q2_12(a) - el2_1(a)*el2_2(a)/2
@inline q2_22(a) = -el2_2(a)^2/2

##################################################################
# Coefficent functions for 3-point embedded boundary SBP operator

@inline el1(a,b) = 1+a-b/2
@inline el2(a,b) = -a+b
@inline el3(a,b) = -b/2

@inline er1(a,b) = -a/2
@inline er2(a,b) = a-b
@inline er3(a,b) = 1+b-a/2

@inline h11(a,b) = a^2/4 - b^2/4 + a + 3/4
@inline h22(a,b) = 1/2
@inline h33(a,b) = b^2/4 - a^2/4 + b + 3/4

@inline Q12(a,b) = a^2/4 + b^2/4 - a*b/2 + a/2 - b/2 + 1/4
@inline Q13(a,b) = -a^2/4 - b^2/4 + a*b/2 + a/4 + b/4 + 1/4
@inline Q23(a,b) = Q12(a,b)

@inline q11(a,b) = er1(a,b)^2/2 - el1(a,b)^2/2
@inline q12(a,b) = Q12(a,b) + er1(a,b)*er2(a,b)/2 - el1(a,b)*el2(a,b)/2
@inline q13(a,b) = Q13(a,b) + er1(a,b)*er3(a,b)/2 - el1(a,b)*el3(a,b)/2

@inline q31(a,b) = -Q13(a,b) + er1(a,b)*er3(a,b)/2 - el1(a,b)*el3(a,b)/2
@inline q32(a,b) = -Q23(a,b) + er3(a,b)*er2(a,b)/2 - el3(a,b)*el2(a,b)/2
@inline q33(a,b) = er3(a,b)^2/2 - el3(a,b)^2/2

# @inline function Dx(f::Function,U::Data.CellArray,ns::Tuple,i::Int,j::Int,k::Int)
#     n = ns[1]
#     if i in 5:n-4
#         (f(U[i-2,j,k]) - 8*f(U[i-1,j,k]) + 8*f(U[i+1,j,k]) - f(U[i+2,j,k]))/12
#     elseif i==1
#         (q11*f(U[1,j,k]) + q21*f(U[2,j,k]) + q31*f(U[3,j,k]) + q41*f(U[4,j,k]))
#     elseif i==2
#         (q12*f(U[1,j,k]) + q32*f(U[3,j,k]))
#     elseif i==3
#         (q13*f(U[1,j,k]) + q23*f(U[2,j,k]) + q43*f(U[4,j,k]) + q53*f(U[5,j,k]))
#     elseif i==4
#         (q14*f(U[1,j,k]) + q34*f(U[3,j,k]) + q54*f(U[5,j,k]) + q64*f(U[6,j,k]))
#     elseif i==n
#         -(q11*f(U[n,j,k]) + q21*f(U[n-1,j,k]) + q31*f(U[n-2,j,k]) + q41*f(U[n-3,j,k]))
#     elseif i==n-1
#         -(q12*f(U[n,j,k]) + q32*f(U[n-2,j,k]))
#     elseif i==n-2
#         -(q13*f(U[n,j,k]) + q23*f(U[n-1,j,k]) + q43*f(U[n-3,j,k]) + q53*f(U[n-4,j,k]))
#     else#if i==n-3
#         -(q14*f(U[n,j,k]) + q34*f(U[n-2,j,k]) + q54*f(U[n-4,j,k]) + q64*f(U[n-5,j,k]))
#     end

# end

# @inline function ψu(U::WaveCell) # Scalar gradient-flux

#     # Give names to stored arrays from the state vector
#     Ψ = U.Ψ

#     # gi = inverse(g)

#     # α = 1/sqrt(-gi[1,1])

#     # βr = -gi[1,2]/gi[1,1]
#     # βθ = -gi[1,3]/gi[1,1]

#     return Ψ

# end

# @inline function Dx(f,U,ns,i,j,k) 
#     n = ns[1]
#     if i in 5:n-4
#         -f(U[i-2,j,k]) + 4*f(U[i-1,j,k]) - 6*f(U[i,j,k]) + 4*f(U[i+1,j,k]) - f(U[i+2,j,k])
#     elseif i==1
#         (a11*f(U[1,j,k]) + a21*f(U[2,j,k]) + a31*f(U[3,j,k]))
#     elseif i==2
#         (a12*f(U[1,j,k]) + a22*f(U[2,j,k]) + a32*f(U[3,j,k]) + a42*f(U[4,j,k]))
#     elseif i==3
#         (a13*f(U[1,j,k]) + a23*f(U[2,j,k]) + a33*f(U[3,j,k]) + a43*f(U[4,j,k]) + a53*f(U[5,j,k]))
#     elseif i==4
#         (a24*f(U[2,j,k]) + a34*f(U[3,j,k]) + a44*f(U[4,j,k]) + a54*f(U[5,j,k]) + a64*f(U[6,j,k]))
#     elseif i==n
#         (a11*f(U[n,j,k]) + a21*f(U[n-1,j,k]) + a31*f(U[n-2,j,k]))
#     elseif i==n-1
#         (a12*f(U[n,j,k]) + a22*f(U[n-1,j,k]) + a32*f(U[n-2,j,k]) + a42*f(U[n-3,j,k]))
#     elseif i==n-2
#         (a13*f(U[n,j,k]) + a23*f(U[n-1,j,k]) + a33*f(U[n-2,j,k]) + a43*f(U[n-3,j,k]) + a53*f(U[n-4,j,k]))
#     elseif i==n-3
#         (a24*f(U[n-1,j,k]) + a34*f(U[n-2,j,k]) + a44*f(U[n-3,j,k]) + a54*f(U[n-4,j,k]) + a64*f(U[n-5,j,k]))
#     end
# end

# @parallel_indices (i,y) function rhs!(Type,U1,U2,U3,U_init,C1,C2,C3,H,∂H,rm,θm,t,ns,dt,_ds,iter)

#     #Explicit slices from main memory
#     # At each iteration in an Runge-Kutta algorithm,
#     # a U-read (U) and U-write (Uw) are defined
#     if iter == 1
#         # U3 has past iteration.
#         U = U1
#         Uw = U2
#         Uxy = U[x,y]
#     elseif iter == 2
#         # U1 has past iteration.
#         U = U2
#         Uw = U3
#         Uxy = U[x,y]
#     else
#         # U2 has past iteration.
#         U = U3
#         Uw = U1
#         Uxy = U[x,y]
#     end

#     Hxy = H[x,y]; ∂Hxy = ∂H[x,y];

#     r = rm[x,y]; θ = θm[x,y];

#     ρ = Uxy.ρ

#     ψ,ψr,ψθ,Ψ,g,dr,dθ,P = unpack(Uxy,false)

#     # Calculate inverse metric components
#     gi = inverse(g)

#     # Calculate lapse and shift
#     α  = 1/sqrt(-gi[1,1])
#     βr = -gi[1,2]/gi[1,1]
#     βθ = -gi[1,3]/gi[1,1]

#     # Time derivatives of the metric
#     ∂tg = βr*dr + βθ*dθ - α*P

#     nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 

#     n = @Vec [nt,nr,nθ]

#     n_ = @Vec [-α,0.0,0.0]

#     γi = gi + symmetric(@einsum n[μ]*n[ν])

#     δ = one(ThreeTensor)

#     γm = δ + (@einsum n_[μ]*n[ν])

#     γ = g + symmetric(@einsum n_[μ]*n_[ν])

#     #Derivatives of the lapse and the shift 

#     # ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])
#     # ∂tβ = α*(@einsum γi[α,μ]*n[ν]*∂tg[μ,ν]) # result is a 3-vector

#     # Metric derivatives
#     ∂g = Symmetric3rdOrderTensor{Type}(
#         (σ,μ,ν) -> (σ==1 ? ∂tg[μ,ν] : σ==2 ? dr[μ,ν] : σ==3 ? dθ[μ,ν] : @assert false)
#         )

#     # Chistoffel Symbols (of the first kind, i.e. all covariant indices)
#     Γ  = Symmetric3rdOrderTensor{Type}((σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν]))

#     C_ = @einsum gi[ϵ,σ]*Γ[μ,ϵ,σ] - Hxy[μ] #- ∂ψ[α]

#     Cr = (Dρ2T(fg,U,r,θ,ns,_ds,x,y) - dr)
#     Cθ = (Dz2T(fg,U,r,θ,ns,_ds,x,y) - dθ)

#     Cψr = (Dρ2(fψ,U,r,θ,ns,_ds,x,y) - ψr)
#     Cψθ = (Dz2(fψ,U,r,θ,ns,_ds,x,y) - ψθ)

#     # Scalar Evolution
#     ######################################################################

#     γ1 = -1.;
#     γ0 =  1.# + 9/(ρ+1)
#     γ2 = 1.;

#     ∂tψ  = βr*ψr + βθ*ψθ - α*Ψ

#     ∂ψ   = @Vec [∂tψ,ψr,ψθ]
    
#     ∂trootγ = @einsum 0.5*γi[i,j]*∂tg[i,j]

#     #######################################################################
#     # Define Stress energy tensor and trace 
#     # T = zero(StateTensor{Type})
#     # Tt = 0.

#     ∂tP = -2*α*S  # + 8*pi*Tt*g - 16*pi*T 

#     #∂tP += -2*α*symmetric(@einsum (μ,ν) -> 2*∂ψ[μ]*∂ψ[ν])  # + 8*pi*Tt*g - 16*pi*T 

#     ∂tP += 2*α*symmetric(∂Hxy)

#     #∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*Hxy[ϵ]*∂g[μ,ν,σ])

#     ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*Hxy[ϵ]*Γ[σ,μ,ν])

#     ∂tP -=  α*symmetric(@einsum (μ,ν) -> gi[λ,γ]*gi[ϵ,σ]*Γ[λ,ϵ,σ]*∂g[γ,μ,ν])

#     ∂tP += 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*∂g[λ,ϵ,μ]*∂g[ρ,σ,ν])

#     ∂tP -= 2*α*symmetric(@einsum (μ,ν) -> gi[ϵ,σ]*gi[λ,ρ]*Γ[μ,ϵ,λ]*Γ[ν,σ,ρ])

#     # Constraint damping term for C_

#     ∂tP += γ0*α*symmetric(@einsum (μ,ν) -> 2C_[μ]*n_[ν] - g[μ,ν]*n[ϵ]*C_[ϵ])

#     # ρv = @Vec [0.,1.,0.]
#     # ρv_ = @einsum g[μ,ν]*ρv[μ]
    
#     # ∂tP += -2*γ0*α*exp(-ψ)*symmetric(@einsum (μ,ν) -> ρv_[μ]*n_[ν]*ρv[i]*C_[i])

#     # if (y==1 || y==ns[2])
    
#     # end

#     #∂tP += -2*α*symmetric(@einsum (μ,ν) -> 2*C_[μ]*∂ψ[ν] - g[μ,ν]*gi[α,β]*∂ψ[α]*C_[β])

#     #∂tP += γ0*α*exp(-ψ)*symmetric(@einsum (μ,ν) -> 2*n_[μ]*C_[ν] - g[μ,ν]*n[ϵ]*C_[ϵ])

#     #vt = @Vec [0.,ρ,0.]

#     #∂tP -= 2*α*g*(@einsum gi[μ,ν]*vt[μ]*C_[ν])

#     #∂tP -= 2*α*g*(@einsum gi[μ,ν]*∂ψ[μ]*C_[ν])

#     #∂tP -= γ0*α*g*exp(-ψ)*(@einsum n[ϵ]*C_[ϵ])

#     ∂tP -= ∂trootγ*P

#     ∂tP += γ1*γ2*(βr*Cr + βθ*Cθ)

#     ###########################################
#     # All finite differencing occurs here

#     # mask1 = StateTensor{Type}((1.,ρ,1.,1.,ρ,1.))
#     # mask2 = StateTensor{Type}((0.,1.,0.,0.,1.,0.))

#     ∂tP += Div4T(vr,vθ,U,r,θ,ns,_ds,x,y) 

#     ∂tdr = symmetric(Dρ4T(u,U,r,θ,ns,_ds,x,y)) + α*γ2*Cr #+ mask2.*u_odd(U) 

#     ∂tdθ = symmetric(Dz4T(u,U,r,θ,ns,_ds,x,y)) + α*γ2*Cθ

#     ##################################################

#     ∂tP = symmetric(∂tP)

#     ##################################################

#     ∂tψr = Dρ4(ψu,U,r,θ,ns,_ds,x,y) + α*γ2*Cψr

#     ∂tψθ = Dz4(ψu,U,r,θ,ns,_ds,x,y) + α*γ2*Cψθ

#     ∂tΨ  = Div4(ψvr,ψvθ,U,r,θ,ns,_ds,x,y) #- (α/4)*St 

#     #∂tΨ  -= α*(@einsum gi[i,j]*∂ψ[i]*C_[j]) 

#     #∂tΨ  += α*γ0*(@einsum n[i]*C_[i]) 

#     ∂tΨ  -= ∂trootγ*Ψ

#     # ∂tψ  = 0.
#     # ∂tψr = 0.
#     # ∂tψθ = 0.
#     # ∂tΨ  = 0.

#     ######################################################

#     ∂ρg = Dρ2T(fg,U,r,θ,ns,_ds,x,y)
#     ∂zg = Dz2T(fg,U,r,θ,ns,_ds,x,y)
#     ∂ρψ = Dρ2(fψ,U,r,θ,ns,_ds,x,y)
#     ∂zψ = Dz2(fψ,U,r,θ,ns,_ds,x,y)

#     # ∂tg = βr*∂ρg + βθ*∂zg - α*P

#     ###################################################
#     #Boundary conditions

#     c1 = (x==1); c2 = (x==ns[1]);

#     if (c1 || c2) #&& false

#         if c1 
#             b=1; p=-1
#         else 
#             b=2; p=1
#         end

#         Cy = C1[b,y]

#         h = g
#         hi = gi
#         ∂h = ∂g
    
#         α  = 1/sqrt(-hi[1,1])
#         βr = -hi[1,2]/hi[1,1]
#         βθ = -hi[1,3]/hi[1,1]
    
#         nt = 1.0/α; nr = -βr/α; nθ = -βθ/α; 
    
#         n = @Vec [nt,nr,nθ]
    
#         n_ = @Vec [-α,0.0,0.0]

#         # Form the unit normal vector to the boundary

#         s = @Vec [0.0,p*sin(θ),p*cos(θ)]

#         snorm = @einsum h[μ,ν]*s[μ]*s[ν]
    
#         s = s/sqrt(snorm) 

#         s_ = @einsum h[μ,ν]*s[ν]
    
#         # Form the unit tangent to the boundary

#         # if y == 1 || y== ns[2]
#         #     Θ_ = @Vec [0.,0.,0.]
#         # else

#         # end
#         #Θ = @Vec [0.0,r*cos(θ),-r*sin(θ)]

#         Θ_ = @Vec [βr*cos(θ)-βθ*sin(θ),cos(θ),-sin(θ)]
#         Θnorm = @einsum hi[μ,ν]*Θ_[μ]*Θ_[ν]
#         Θ_ = Θ_/sqrt(Θnorm)

#         Θ = @einsum hi[μ,ν]*Θ_[ν]

#         # Form ingoing and outgoing null vectors

#         ℓ = @einsum (n[α] + s[α])/sqrt(2)
#         k = @einsum (n[α] - s[α])/sqrt(2)

#         ℓ_ = @einsum h[μ,α]*ℓ[α]
#         k_ = @einsum h[μ,α]*k[α]

#         # σ = StateTensor{Type}((μ,ν) -> gi[μ,ν] + k[μ]*ℓ[ν] + ℓ[μ]*k[ν])
        
#         # σm = @einsum g[μ,α]*σ[α,ν] # mixed indices (raised second index)

#         # σ_ = @einsum g[μ,α]*σm[ν,α]

#         σ = @einsum Θ[μ]*Θ[ν]
        
#         σm = @einsum Θ_[μ]*Θ[ν] # mixed indices (raised second index)

#         σ_ = @einsum Θ_[μ]*Θ_[ν]

#         cp =  α - βr*s_[2] - βθ*s_[3]
#         cm = -α - βr*s_[2] - βθ*s_[3]
#         c0 =    - βr*s_[2] - βθ*s_[3]

#         # if y==1 && x==ns[1]
#         #     println(cp," ",cm," ",c0)
#         # end

#         βdotθ = βr*Θ_[2] + βθ*Θ_[3]
    
#         Up = @einsum StateTensor{Type} k[α]*∂h[α,μ,ν]
#         #Um = @einsum StateTensor{Type} ℓ[α]*∂h[α,μ,ν]
#         U0 = @einsum StateTensor{Type} Θ[α]*∂h[α,μ,ν]

#         # Up = -(P + s[2]*dr + s[3]*dθ)/sqrt(2)
#         # U0 = Θ[2]*dr + Θ[3]*dθ

#         Uψp = @einsum k[α]*∂ψ[α]
#         Uψ0 = @einsum Θ[α]*∂ψ[α]


#         # Q4 = SymmetricFourthOrderTensor{3,Type}(
#         #     (μ,ν,α,β) -> σ_[μ,ν]*σ[α,β] - 2*ℓ_[μ]*σm[ν,α]*k[β] + ℓ_[μ]*ℓ_[ν]*k[α]*k[β]
#         # ) # Four index constraint projector (indices down down up up)
    
#         Q3 = Symmetric3rdOrderTensor{Type}(
#             (α,μ,ν) -> ℓ_[μ]*σm[ν,α]/2 + ℓ_[ν]*σm[μ,α]/2 - σ_[μ,ν]*ℓ[α] - ℓ_[μ]*ℓ_[ν]*k[α]/2
#         ) # Three index constraint projector (indices up down down)
#         # Note order of indices here

#         # # O = SymmetricFourthOrderTensor{4}(
#         # #     (μ,ν,α,β) -> σm[μ,α]*σm[ν,β] - σ_[μ,ν]*σ[α,β]/2
#         # # ) # Gravitational wave projector

#         G = FourthOrderTensor{3,Type}(
#             (μ,ν,α,β) -> (2k_[μ]*ℓ_[ν]*k[α]*ℓ[β] - 2k_[μ]*σm[ν,α]*ℓ[β] + k_[μ]*k_[ν]*ℓ[α]*ℓ[β])
#         ) # Four index gauge projector (indices down down up up)

#         # G = minorsymmetric(G)

#         Amp = 0.00001
#         #Amp = 0.0
#         σ0 = 0.5
#         μ0 = 10.5

#         #f(t,z) = (μ0-t-σ0)<z<(μ0-t+σ0) ? (Amp/σ0^8)*(z-((μ0-t)-σ0))^4*(z-((μ0-t)+σ0))^4 : 0.

#         #f(t,z) = (μ0-t-σ0)<z<(μ0-t+σ0) ? Amp : 0.

#         f(t,ρ,z) = (μ0-t-σ0)<ρ<(μ0-t+σ0) ? Amp : 0.

#         if c2
#             Cf = @Vec [f(t,r*sin(θ),r*cos(θ)),r*sin(θ)*f(t,r*sin(θ),r*cos(θ)),f(t,r*sin(θ),r*cos(θ))]
#             #Cf = @Vec [0.,0.,0.] 
#             #Cf = C_
#         else
#             #Cf = Cy

#             # CBC = constraints(U[2,y]) 
#             Cf = @Vec [0.,0.,0.] 
#             #Cf = C_

#             #Cf = @Vec [f(-t,r*cos(θ)),0.,f(-t,r*cos(θ))]
#         end
    
#         A_ = @einsum (2*ℓ[μ]*Up[μ,α] - hi[μ,ν]*Up[μ,ν]*ℓ_[α] + hi[μ,ν]*U0[μ,ν]*Θ_[α] - 2*Θ[μ]*U0[μ,α] + 2*Hxy[α] + 2*Cf[α])
#         # # index down

#         # # Condition ∂tgμν = 0 on the boundary
#         Umb2 = (cp/cm)*Up + sqrt(2)*(βdotθ/cm)*U0
#         #Umb2 = ℓ[1]*∂tg + ℓ[2]*∂ρg + ℓ[3]*∂zg
#         #Umb2 = zero(StateTensor)


#         Umbh = @einsum StateTensor{Type} (Q3[α,μ,ν]*A_[α] + G[μ,ν,α,β]*Umb2[α,β])


#         Umb = symmetric(Umbh)

#         Pb  = -(Up + Umb)/sqrt(2)
#         # dxb = ∂ρg
#         # dyb = ∂zg
#         dxb = Θ_[2]*U0 - k_[2]*Umb - ℓ_[2]*Up
#         dyb = Θ_[3]*U0 - k_[3]*Umb - ℓ_[3]*Up 
        

#         ∂χ = @Vec [∂tχ,∂ρχ,∂zχ]

#         Uχp = @einsum k[α]*∂χ[α]
#         Uχ0 = @einsum Θ[α]*∂χ[α]

#         #Uχmb = (cp/cm)*Uχp + 2*(βdotθ/cm)*Uχ0
#         Uχmb = (cp/cm)*Uχp + 2*(βdotθ/cm)*Uχ0
#         #Uχmb = 0. #(Θ_[2]*Uχ0 - ℓ_[2]*Uχp)/k_[2]

#         Uψmb = Umb[2,2]/g[2,2]/4 + Uχmb/2

#         #Uψmb = (cp/cm)*Uψp + 2*(βdotθ/cm)*Uψ0

#         Ψb  = -(Uψp + Uψmb)/sqrt(2)
#         ψxb = Θ_[2]*Uψ0 - k_[2]*Uψmb - ℓ_[2]*Uψp
#         ψyb = Θ_[3]*Uψ0 - k_[3]*Uψmb - ℓ_[3]*Uψp 

#         #∂tψ = (Uψmb - ℓ[2]*ψxb - ℓ[3]*ψyb)/ℓ[1]
#         #∂tψ = (Uψmb - ℓ[2]*∂ρψ - ℓ[3]*∂zψ)/ℓ[1]

#         ##########################################################################

#         # ∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])

#         # ∂tβ = α*(@einsum γi[α,μ]*n[ν]*∂tg[μ,ν]) # result is a 3-vector

#         # ∂t∂tg = (βr*∂tdr + βθ*∂tdθ - α*∂tP) + (∂tβ[2]*dr + ∂tβ[3]*dθ - ∂tα*P)

#         # ∂t∂g = Symmetric3rdOrderTensor{Type}(
#         #     (σ,μ,ν) -> (σ==1 ? ∂t∂tg[μ,ν] : σ==2 ? ∂tdr[μ,ν] : σ==3 ? ∂tdθ[μ,ν] : @assert false))

#         # ∂tΓ  = Symmetric3rdOrderTensor{Type}(
#         #     (σ,μ,ν) -> 0.5*(∂t∂g[ν,μ,σ] + ∂t∂g[μ,ν,σ] - ∂t∂g[σ,μ,ν])
#         #     )   

#         # ∂tH = Vec{3}((∂Hxy[1,:]...))
#         # ∂xH = Vec{3}((∂Hxy[2,:]...))
#         # ∂zH = Vec{3}((∂Hxy[3,:]...))

#         # ∂tC = (@einsum gi[ϵ,σ]*∂tΓ[λ,ϵ,σ] - gi[μ,ϵ]*gi[ν,σ]*Γ[λ,μ,ν]*∂tg[ϵ,σ]) - ∂tH

#         # # set up finite differencing for the constraints, by defining a function
#         # # that calculates the constraints for any x and y index. This
#         # # might not be the best idea, but should work.

#         # dxC = DρC(constraints,U,r,θ,ns,_ds,x,y) - ∂xH 
#         # dyC = DzC(constraints,U,r,θ,ns,_ds,x,y) - ∂zH 

#         # ∂C = ThreeTensor{Type}(
#         #     (σ,ν) ->  (σ==1 ? ∂tC[ν] : σ==2 ? dxC[ν] : σ==3 ? dyC[ν] : @assert false)
#         #     )

#         # UpC = @einsum k[α]*∂C[α,μ]
#         # U0C = @einsum Θ[α]*∂C[α,μ]

#         # UmbC = @Vec [0.,0.,0.] #(U0C).^(2)./UpC

#         # ∂tCb = zeroST(ρ)# Θ_[1]*U0C - k_[1]*UmbC - ℓ_[1]*UpC

#         #∂tCb = -γ0*C_

#         ε = 2*abs(cm)*_ds[1]

#         ∂tP  += ε*(Pb - P)
#         ∂tdr += ε*(dxb - dr)
#         ∂tdθ += ε*(dyb - dθ)
    
#         ∂tΨ  += ε*(Ψb - Ψ)
#         ∂tψr += ε*(ψxb - ψr)
#         ∂tψθ += ε*(ψyb - ψθ)

#     end

#     ∂tψv = @Vec [∂tψ,∂tψr,∂tψθ,∂tΨ]

#     ∂tU = StateVector{Type}(ρ,∂tψv,∂tg,∂tdr,∂tdθ,∂tP)

#     Dis = Dissipation(U,r,θ,x,y,ns)
#     Dis = StateVector{Type}(ρ,Dis.ψ,Dis.g,Dis.dr,Dis.dθ,Dis.P)

#     ∂tU += Dis

#     ##########################################################

#     # if ρ == 0.
#     #     mask1 = StateTensor{Type}((1.,0.,1.,1.,0.,1.))
#     #     mask2 = StateTensor{Type}((0.,1.,0.,0.,1.,0.))
#     #     mask3 = @Vec [1.,0.,1.,1.]
#     # else
#     #     mask1 = StateTensor{Type}((1.,1/ρ,1.,1.,1/ρ,1.))
#     #     mask2 = StateTensor{Type}((1/ρ,1.,1/ρ,1/ρ,1.,1/ρ))
#     #     mask3 = @Vec [1.,1/ρ,1.,1.]
#     # end

#     # ∂tU = StateVector{Type}(ρ,∂tψv,mask1.*∂tg,mask2.*∂tdr,mask1.*∂tdθ,mask1.*∂tP) #mask2.*mask3.*

#     #∂tU = StateVector{Type}(ρ,∂tψv,∂tg,∂tdr,∂tdθ,∂tP) #mask2.*mask3.*

#     #########################################################

#     # if iter == 1
#     #     U1t = unpack(Uxy)
#     #     Uwxy = U1t + dt*∂tU
#     # elseif iter == 2
#     #     U1t = unpack(U1[x,y])
#     #     U2t = unpack(Uxy)
#     #     Uwxy = (3/4)*U1t + (1/4)*U2t + (1/4)*dt*∂tU
#     # elseif iter == 3
#     #     U1t = unpack(U1[x,y])
#     #     U2t = unpack(Uxy)
#     #     Uwxy = (1/3)*U1t + (2/3)*U2t + (2/3)*dt*∂tU
#     # end

#     if iter == 1
#         U1t = Uxy
#         Uwxy = U1t + dt*∂tU
#     elseif iter == 2
#         U1t = U1[x,y]
#         U2t = Uxy
#         Uwxy = (3/4)*U1t + (1/4)*U2t + (1/4)*dt*∂tU
#     elseif iter == 3
#         U1t = U1[x,y]
#         U2t = Uxy
#         Uwxy = (1/3)*U1t + (2/3)*U2t + (2/3)*dt*∂tU
#     end

#     #Uw[x,y] = pack(Uwxy)

#     Uw[x,y] = Uwxy

#     return
    
# end

@inline function ψu(U::WaveCell) # Scalar gradient-flux
    
    # Give names to stored arrays from the state vector
    Ψ = U.Ψ

    # gi = inverse(g)

    # α = 1/sqrt(-gi[1,1])

    # βr = -gi[1,2]/gi[1,1]
    # βθ = -gi[1,3]/gi[1,1]

    return -Ψ

end

@inline function vx(U::WaveCell) # Scalar gradient-flux
    
    # Give names to stored arrays from the state vector
    ψx = U.ψx

    # gi = inverse(g)

    # α = 1/sqrt(-gi[1,1])

    # βr = -gi[1,2]/gi[1,1]
    # βθ = -gi[1,3]/gi[1,1]

    return ψx

end

@inline function vy(U::WaveCell) # Scalar gradient-flux
    
    # Give names to stored arrays from the state vector
    ψy = U.ψy

    # gi = inverse(g)

    # α = 1/sqrt(-gi[1,1])

    # βr = -gi[1,2]/gi[1,1]
    # βθ = -gi[1,3]/gi[1,1]

    return ψy

end

@inline function vz(U::WaveCell) # Scalar gradient-flux
    
    # Give names to stored arrays from the state vector
    ψz = U.ψz

    # gi = inverse(g)

    # α = 1/sqrt(-gi[1,1])

    # βr = -gi[1,2]/gi[1,1]
    # βθ = -gi[1,3]/gi[1,1]

    return ψz

end

# Base.@propagate_inbounds @inline function Dx(f,U,ns,i,j,k)
#     n = ns[1]
#     if i in 5:n-4
#         (f(U[i-2,j,k]) - 8*f(U[i-1,j,k]) + 8*f(U[i+1,j,k]) - f(U[i+2,j,k]))/12
#     elseif i==1
#         (q11*f(U[1,j,k]) + q21*f(U[2,j,k]) + q31*f(U[3,j,k]) + q41*f(U[4,j,k]))
#     elseif i==2
#         (q12*f(U[1,j,k]) + q32*f(U[3,j,k]))
#     elseif i==3
#         (q13*f(U[1,j,k]) + q23*f(U[2,j,k]) + q43*f(U[4,j,k]) + q53*f(U[5,j,k]))
#     elseif i==4
#         (q14*f(U[1,j,k]) + q34*f(U[3,j,k]) + q54*f(U[5,j,k]) + q64*f(U[6,j,k]))
#     elseif i==n
#         -(q11*f(U[n,j,k]) + q21*f(U[n-1,j,k]) + q31*f(U[n-2,j,k]) + q41*f(U[n-3,j,k]))
#     elseif i==n-1
#         -(q12*f(U[n,j,k]) + q32*f(U[n-2,j,k]))
#     elseif i==n-2
#         -(q13*f(U[n,j,k]) + q23*f(U[n-1,j,k]) + q43*f(U[n-3,j,k]) + q53*f(U[n-4,j,k]))
#     else#if i==n-3
#         -(q14*f(U[n,j,k]) + q34*f(U[n-2,j,k]) + q54*f(U[n-4,j,k]) + q64*f(U[n-5,j,k]))
#     end
# end

# Base.@propagate_inbounds @inline function Dx(f,Um2,Um1,U1,U2,ns,i,j,k)
#     (f(Um2) - 8*f(Um1) + 8*f(U1) - f(U2))/12
# end

# Base.@propagate_inbounds @inline function Dy(f,U,ns,i,j,k)
#     n = ns[2]
#     if j in 5:n-4
#         (f(U[i,j-2,k]) - 8*f(U[i,j-1,k]) + 8*f(U[i,j+1,k]) - f(U[i,j+2,k]))/12
#     elseif j==1
#         (q11*f(U[i,1,k]) + q21*f(U[i,2,k]) + q31*f(U[i,3,k]) + q41*f(U[i,4,k]))
#     elseif j==2
#         (q12*f(U[i,1,k]) + q32*f(U[i,3,k]))
#     elseif j==3
#         (q13*f(U[i,1,k]) + q23*f(U[i,2,k]) + q43*f(U[i,4,k]) + q53*f(U[i,5,k]))
#     elseif j==4
#         (q14*f(U[i,1,k]) + q34*f(U[i,3,k]) + q54*f(U[i,5,k]) + q64*f(U[i,6,k]))
#     elseif j==n
#         -(q11*f(U[i,n,k]) + q21*f(U[i,n-1,k]) + q31*f(U[i,n-2,k]) + q41*f(U[i,n-3,k]))
#     elseif j==n-1
#         -(q12*f(U[i,n,k]) + q32*f(U[i,n-2,k]))
#     elseif j==n-2
#         -(q13*f(U[i,n,k]) + q23*f(U[i,n-1,k]) + q43*f(U[i,n-3,k]) + q53*f(U[i,n-4,k]))
#     else#if i==n-3
#         -(q14*f(U[i,n,k]) + q34*f(U[i,n-2,k]) + q54*f(U[i,n-4,k]) + q64*f(U[i,n-5,k]))
#     end
# end

# Base.@propagate_inbounds @inline function Dy(f,Um2,Um1,U1,U2,ns,i,j,k)
#     (f(Um2) - 8*f(Um1) + 8*f(U1) - f(U2))/12
# end

# Base.@propagate_inbounds @inline function Dz(f,U,ns,i,j,k)
#     n = ns[3]
#     if k in 5:n-4
#         (f(U[i,j,k-2]) - 8*f(U[i,j,k-1]) + 8*f(U[i,j,k+1]) - f(U[i,j,k+2]))/12
#     elseif k==1
#         (q11*f(U[i,j,1]) + q21*f(U[i,j,2]) + q31*f(U[i,j,3]) + q41*f(U[i,j,4]))
#     elseif k==2
#         (q12*f(U[i,j,1]) + q32*f(U[i,j,3]))
#     elseif k==3
#         (q13*f(U[i,j,1]) + q23*f(U[i,j,2]) + q43*f(U[i,j,4]) + q53*f(U[i,j,5]))
#     elseif k==4
#         (q14*f(U[i,j,1]) + q34*f(U[i,j,3]) + q54*f(U[i,j,5]) + q64*f(U[i,j,6]))
#     elseif k==n
#         -(q11*f(U[i,j,n]) + q21*f(U[i,j,n-1]) + q31*f(U[i,j,n-2]) + q41*f(U[i,j,n-3]))
#     elseif k==n-1
#         -(q12*f(U[i,j,n]) + q32*f(U[i,j,n-2]))
#     elseif k==n-2
#         -(q13*f(U[i,j,n]) + q23*f(U[i,j,n-1]) + q43*f(U[i,j,n-3]) + q53*f(U[i,j,n-4]))
#     else#if i==n-3
#         -(q14*f(U[i,j,n]) + q34*f(U[i,j,n-2]) + q54*f(U[i,j,n-4]) + q64*f(U[i,j,n-5]))
#     end
# end

Base.@propagate_inbounds @inline function Dx(f,Um,ns,αs,i,j,k)
    nsx = ns[1]
    nl,nr = nsx
    αsx = αs[1]
    @inline U(x) = f(getindex(Um,x,j,k))
    if nr-nl>=7
        D_4_2(U,nsx,αsx,i)
    elseif nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
        D_2_1(U,nsx,αsx,i)
    elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
        D_3point(U,nsx,αsx,i)
    elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
        -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    else # only one grid point, extrapolate derivative
        @assert false # Not implemented
        return 0.
        # nlx,nrx = ns[1]
        # nly,nry = ns[2]
        # if i == nlx && i ≠ nrx
        #     3*Dz(f,Um,ns,αs,nlx+1,j,k) - 3*Dz(f,Um,ns,αs,nlx+2,j,k) + Dz(f,Um,ns,αs,nlx+3,j,k)
        # elseif i == nrx && i ≠ nlx
        #     3*Dz(f,Um,ns,αs,nrx-1,j,k) - 3*Dz(f,Um,ns,αs,nrx-2,j,k) + Dz(f,Um,ns,αs,nrx-3,j,k)
        # elseif j == nly && j ≠ nry
        #     3*Dz(f,Um,ns,αs,i,nly+1,k) - 3*Dz(f,Um,ns,αs,i,nly+2,k) + Dz(f,Um,ns,αs,i,nly+3,k)
        # else#if j == nry && j ≠ nly
        #     3*Dz(f,Um,ns,αs,i,nrx-1,k) - 3*Dz(f,Um,ns,αs,i,nrx-2,k) + Dz(f,Um,ns,αs,i,nrx-3,k)
        # end
    end
    
end

Base.@propagate_inbounds @inline function Dy(f,Um,ns,αs,i,j,k)
    nsy = ns[2]
    nl,nr = nsy
    αsy = αs[2]
    @inline U(x) = f(getindex(Um,i,x,k))
    if nr-nl>=7
        D_4_2(U,nsy,αsy,j)
    elseif nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
        D_2_1(U,nsy,αsy,j)
    elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
        D_3point(U,nsy,αsy,j)
    elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
        -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    else # only one grid point, extrapolate derivative
        @assert false
        return 0.
        # nlx,nrx = ns[1]
        # nly,nry = ns[2]
        # if i == nlx && i ≠ nrx
        #     3*Dz(f,Um,ns,αs,nlx+1,j,k) - 3*Dz(f,Um,ns,αs,nlx+2,j,k) + Dz(f,Um,ns,αs,nlx+3,j,k)
        # elseif i == nrx && i ≠ nlx
        #     3*Dz(f,Um,ns,αs,nrx-1,j,k) - 3*Dz(f,Um,ns,αs,nrx-2,j,k) + Dz(f,Um,ns,αs,nrx-3,j,k)
        # elseif j == nly && j ≠ nry
        #     3*Dz(f,Um,ns,αs,i,nly+1,k) - 3*Dz(f,Um,ns,αs,i,nly+2,k) + Dz(f,Um,ns,αs,i,nly+3,k)
        # else#if j == nry && j ≠ nly
        #     3*Dz(f,Um,ns,αs,i,nrx-1,k) - 3*Dz(f,Um,ns,αs,i,nrx-2,k) + Dz(f,Um,ns,αs,i,nrx-3,k)
        # end
    end
    
end

Base.@propagate_inbounds @inline function Dz(f,Um,ns,αs,i,j,k)
    nsz = ns[3]
    nl,nr = nsz
    αsz = αs[3]
    @inline U(x) = f(getindex(Um,i,j,x))
    if nr-nl>=7
        D_4_2(U,nsz,αsz,k)
    elseif nr-nl>=3 # not enough points for 4th order stencil, go to 2nd order
        D_2_1(U,nsz,αsz,k)
    elseif nr-nl==2 # not enough points for 2nd order stencil, go to 1st order 3 point
        D_3point(U,nsz,αsz,k)
    elseif nr-nl==1 # not enough points for 2nd order stencil, go to 1st order 2 point
        -U(nl) + U(nr) # note the two point operator happens to be the same for both points
    else # only one grid point, extrapolate derivative
        @assert false
        return 0.
        # nlx,nrx = ns[1]
        # nly,nry = ns[2]
        # if i == nlx && i ≠ nrx
        #     3*Dz(f,Um,ns,αs,nlx+1,j,k) - 3*Dz(f,Um,ns,αs,nlx+2,j,k) + Dz(f,Um,ns,αs,nlx+3,j,k)
        # elseif i == nrx && i ≠ nlx
        #     3*Dz(f,Um,ns,αs,nrx-1,j,k) - 3*Dz(f,Um,ns,αs,nrx-2,j,k) + Dz(f,Um,ns,αs,nrx-3,j,k)
        # elseif j == nly && j ≠ nry
        #     3*Dz(f,Um,ns,αs,i,nly+1,k) - 3*Dz(f,Um,ns,αs,i,nly+2,k) + Dz(f,Um,ns,αs,i,nly+3,k)
        # else#if j == nry && j ≠ nly
        #     3*Dz(f,Um,ns,αs,i,nrx-1,k) - 3*Dz(f,Um,ns,αs,i,nrx-2,k) + Dz(f,Um,ns,αs,i,nrx-3,k)
        # end
    end
    
end

Base.@propagate_inbounds @inline function D_4_2(U,ns,αs,k)
    nl,nr = ns
    αl,αr = αs
    if k in nl+4:nr-4
        (U(k-2) - 8*U(k-1) + 8*U(k+1) - U(k+2))/12
    elseif k==nl
        (q11(αl)*U(nl) + q12(αl)*U(nl+1) + q13(αl)*U(nl+2) + q14(αl)*U(nl+3))/h11(αl)
    elseif k==nl+1
        (q21(αl)*U(nl) + q22(αl)*U(nl+1) + q23(αl)*U(nl+2) + q24(αl)*U(nl+3))/h22(αl)
    elseif k==nl+2
        (q31(αl)*U(nl) + q32(αl)*U(nl+1) + q33(αl)*U(nl+2) + q34(αl)*U(nl+3) - U(nl+4)/12)/h33(αl)
    elseif k==nl+3
        (q41(αl)*U(nl) + q42(αl)*U(nl+1) + q43(αl)*U(nl+2) + (2/3)*U(nl+4) - U(nl+5)/12)/h44(αl)
    elseif k==nr
        -(q11(αr)*U(nr) + q12(αr)*U(nr-1) + q13(αr)*U(nr-2) + q14(αr)*U(nr-3))/h11(αr)
    elseif k==nr-1
        -(q21(αr)*U(nr) + q22(αr)*U(nr-1) + q23(αr)*U(nr-2) + q24(αr)*U(nr-3))/h22(αr)
    elseif k==nr-2
        -(q31(αr)*U(nr) + q32(αr)*U(nr-1) + q33(αr)*U(nr-2) + q34(αr)*U(nr-3) - U(nr-4)/12)/h33(αr)
    else#if k==nr-3
        -(q41(αr)*U(nr) + q42(αr)*U(nr-1) + q43(αr)*U(nr-2) + (2/3)*U(nr-4) - U(nr-5)/12)/h44(αr)
    end
end

Base.@propagate_inbounds @inline function D_2_1(U,ns,αs,k)
    nl,nr = ns
    αl,αr = αs
    if k in nl+2:nr-2
        (-U(k-1) + U(k+1))/2
    elseif k==nl
        -U(nl) + U(nl+1)
    elseif k==nl+1
        (q2_21(αl)*U(nl) + q2_22(αl)*U(nl+1) + U(nl+2)/2)/h2_22(αl)
    elseif k==nr
        U(nr) - U(nr-1)
    else#if k==nr-1
        -(q2_21(αr)*U(nr) + q2_22(αr)*U(nr-1) + U(nr-2)/2)/h2_22(αr)
    end
end

Base.@propagate_inbounds @inline function D_3point(U,ns,αs,k)
    nl,nr = ns
    αl,αr = αs
    if k == nl
        (q11(αl,αr)*U(nl) + q12(αl,αr)*U(nl+1) + q13(αl,αr)*U(nr))/h11(αl,αr)
    elseif k==nr
        (q31(αl,αr)*U(nl) + q32(αl,αr)*U(nl+1) + q33(αl,αr)*U(nr))/h33(αl,αr)
    else
        (-U(nl) + U(nr))/2
    end
end

# Base.@propagate_inbounds @inline function Dz(f,Um2,Um1,U1,U2,ns,i,j,k)
#     (f(Um2) - 8*f(Um1) + 8*f(U1) - f(U2))/12
# end

Base.@propagate_inbounds @inline function Div(vx,vy,vz,U,ns,αs,ds,i,j,k)
    dx,dy,dz,_ = ds
    Dx(vx,U,ns,αs,i,j,k)/dx + Dy(vy,U,ns,αs,i,j,k)/dy + Dz(vz,U,ns,αs,i,j,k)/dz
end

# @inline function Div(vx,vy,vz,
#                     Uxm2,Uxm1,Ux1,Ux2,
#                     Uym2,Uym1,Uy1,Uy2,
#                     Uzm2,Uzm1,Uz1,Uz2,
#                     ns,ds,i,j,k)
#     dx,dy,dz = ds
#     (  Dx(vx,Uxm2,Uxm1,Ux1,Ux2,ns,i,j,k)/dx 
#      + Dy(vy,Uym2,Uym1,Uy1,Uy2,ns,i,j,k)/dy 
#      + Dz(vz,Uzm2,Uzm1,Uz1,Uz2,ns,i,j,k)/dz )
# end

Base.@propagate_inbounds @inline function energy_cell(U,ds)

    ψx = U.ψx
    ψy = U.ψy
    ψz = U.ψz
    Ψ  = U.Ψ

    return (Ψ^2 + ψx^2 + ψy^2 + ψz^2)*ds[1]*ds[2]*ds[3]

    #return U.Ψ

end

Base.@propagate_inbounds @inline function vectors(rb)
    # Returns the normal vector to the boundary.
    # Forms the normal vector to the boundary depending
    # If you are on a face, edge, or corner

    # lx,ly,lz=ls

    # X,Y,Z = -lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2

    # x,y,z = X[xi],Y[yi],Z[zi]

    # if xi==1; (sx = -1) elseif xi==ns[1]; (sx = 1) else (sx = 0) end
    # if yi==1; (sy = -1) elseif yi==ns[2]; (sy = 1) else (sy = 0) end
    # if zi==1; (sz = -1) elseif zi==ns[3]; (sz = 1) else (sz = 0) end

    x,y,z = rb

    sx = x; sy = y; sz = z; 

    norm = sqrt(sx^2 + sy^2 + sz^2)

    # if norm == 0.; println("what") end

    s = @Vec [0.,sx/norm,sy/norm,sz/norm]

    n = @Vec [1.,0.,0.,0.]

    ℓ = (n + s)/sqrt(2)
    k = (n - s)/sqrt(2)

    δ = one(StateTensor)

    σ = StateTensor((μ,ν) -> δ[μ,ν] + k[μ]*ℓ[ν] + ℓ[μ]*k[ν])

    return (k,ℓ,σ)
end

@inline function find_boundary(ls,ds,xi,yi,zi)

    #f(x,y,z) = x^2 + y^2 + z^2 - 25^2

    lx,ly,lz=ls
    dx,dy,dz=ds

    X,Y,Z = -lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2

    x,y,z = X[xi],Y[yi],Z[zi]

    r = lx/2

    xbr =  sqrt(r^2 - y^2 - z^2) + lx/2
    xbl = -sqrt(r^2 - y^2 - z^2) + lx/2

    ybr =  sqrt(r^2 - x^2 - z^2) + ly/2
    ybl = -sqrt(r^2 - x^2 - z^2) + ly/2

    zbr =  sqrt(r^2 - y^2 - x^2) + lz/2
    zbl = -sqrt(r^2 - y^2 - x^2) + lz/2

    xibr,αrx = divrem(xbr,dx,RoundNearest)
    αrx /= dx;
    xibl,αlx = divrem(xbl,dx,RoundNearest)
    αlx /= dx;

    yibr,αry = divrem(ybr,dy,RoundNearest)
    αry /= dy;
    yibl,αly = divrem(ybl,dy,RoundNearest)
    αly /= dy;

    zibr,αrz = divrem(zbr,dz,RoundNearest)
    αrz /= dz;
    zibl,αlz = divrem(zbl,dz,RoundNearest)
    αlz /= dz;

    inside = !((xibl<=xi<=xibr) && (yibl<=yi<=yibr) && (zibl<=zi<=zibr))

    return (inside, ((xibl,xibr),(yibl,yibr),(zibl,zibr)) , ((αlx,αrx),(αly,αry),(αlz,αrz)))

end

@parallel_indices (xi,yi,zi) function rhs!(U1::Data.CellArray,U2::Data.CellArray,U3::Data.CellArray,ls::Tuple,ds::Tuple,iter::Int)

    T = Data.Number

    if iter == 1
        U = U1
        Uw = U2
    elseif iter == 2
        U = U2
        Uw = U3
    else
        U = U3
        Uw = U1
    end

    lx,ly,lz=ls
    X,Y,Z = -lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2
    #coords = (X,Y,Z)

    inside,ns,αs = find_boundary(ls,ds,xi,yi,zi)

    if inside

    dx,dy,dz,dt = ds

    Uxyz = U[xi,yi,zi]

    ψx = Uxyz.ψx
    ψy = Uxyz.ψy
    ψz = Uxyz.ψz
    Ψ  = Uxyz.Ψ


    # if (i in 5:ns[1]-4 && j in 5:ns[2]-4 && k in 5:ns[3]-4 )
    #     Uxm2 = U[i-2,j,k]; Uxm1 = U[i-1,j,k]; Ux1 = U[i+1,j,k]; Ux2 = U[i+2,j,k]; 
    #     Uym2 = U[i,j-2,k]; Uym1 = U[i,j-1,k]; Uy1 = U[i,j+1,k]; Uy2 = U[i,j+2,k]; 
    #     Uzm2 = U[i,j,k-2]; Uzm1 = U[i,j,k-1]; Uz1 = U[i,j,k+1]; Uz2 = U[i,j,k+2]; 

    #     ∂tψ = Ψ

    #     ∂tdx = Dx(ψu,Uxm2,Uxm1,Ux1,Ux2,ns,i,j,k)/dx

    #     ∂tdy = Dy(ψu,Uym2,Uym1,Uy1,Uy2,ns,i,j,k)/dy
    
    #     ∂tdz = Dz(ψu,Uzm2,Uzm1,Uz1,Uz2,ns,i,j,k)/dz
    
    #     ∂tΨ  = 0. 
    #     # Div(vx,vy,vz,
    #     #         Uxm2,Uxm1,Ux1,Ux2,
    #     #         Uym2,Uym1,Uy1,Uy2,
    #     #         Uzm2,Uzm1,Uz1,Uz2,ns,ds,i,j,k)

    # else

    #     ∂tψ = Ψ

    #     ∂tdx = 0. #Dx(ψu,U,ns,i,j,k)/dx
    
    #     ∂tdy = 0. #Dy(ψu,U,ns,i,j,k)/dy
    
    #     ∂tdz = 0. #Dz(ψu,U,ns,i,j,k)/dz
    
    #     ∂tΨ  = 0. #Div(vx,vy,vz,U,ns,ds,i,j,k)

    # end

    ∂tψ = -Ψ

    ∂tψx = Dx(ψu,U,ns,αs,xi,yi,zi)/dx

    ∂tψy = Dy(ψu,U,ns,αs,xi,yi,zi)/dy

    ∂tψz = Dz(ψu,U,ns,αs,xi,yi,zi)/dz

    ∂tΨ  = -Div(vx,vy,vz,U,ns,αs,ds,xi,yi,zi)

    # Apply Boundary Conditions (test if in any of the exterior regions)
    #(xi in (1,ns[1]) || yi in (1,ns[2]) || zi in (1,ns[3]))

    ri = (xi,yi,zi)

    for i in 1:3

        nl,nr = ns[i] 

        if nr-nl<3 @assert false end

        if ri[i] in nl:nl+2 # in the boundary region on the left side of the line

            αl = αs[i][1]

            # Interpolate the solution vector on the boundary 
            # and determine the boundary position on the coordinate line
            if i == 1 # On an x-line
                rb = (X[nl]-αl*dx,Y[yi],Z[zi])
                Ub = el1(αl)*U[nl,yi,zi] + el2(αl)*U[nl+1,yi,zi] + el3(αl)*U[nl+2,yi,zi]
            elseif i == 2 # On a y-line
                rb = (X[xi],Y[nl]-αl*dy,Z[zi])
                Ub = el1(αl)*U[xi,nl,zi] + el2(αl)*U[xi,nl+1,zi] + el3(αl)*U[xi,nl+2,zi]
            elseif i == 3 # On a z-line
                rb = (X[xi],Y[yi],Z[nl]-αl*dz)
                Ub = el1(αl)*U[xi,yi,nl] + el2(αl)*U[xi,yi,nl+1] + el3(αl)*U[xi,yi,nl+2]
            end

            k,ℓ,σ = vectors(rb) # Form boundary basis

            ψxb = Ub.ψx
            ψyb = Ub.ψy
            ψzb = Ub.ψz
            Ψb  = Ub.Ψ

            ∂ψb = @Vec [-Ψb,ψxb,ψyb,ψzb]

            Upb = @einsum k[α]*∂ψb[α]
            U0b = @einsum σ[α,β]*∂ψb[β]

            UmBC = -Upb

            ΨBC  = -(Upb + UmBC)/sqrt(2)
            ∂ψBC =  U0b - k*UmBC - ℓ*Upb

            if ri[i] == nl 
                ε = el1(αl)/h11(αl)/ds[i]
            elseif ri[i] == nl-1
                ε = el2(αl)/h22(αl)/ds[i]
            else#if ri[i] == nl-2
                ε = el3(αl)/h33(αl)/ds[i]
            end

            ∂tΨ  += ε*(ΨBC - Ψb)
            ∂tψx += ε*(∂ψBC[2] - ψxb)
            ∂tψy += ε*(∂ψBC[3] - ψyb)
            ∂tψz += ε*(∂ψBC[4] - ψzb)

        end

        if ri[i] in (nr-2):nr # in the boundary region on the right side of the line

            # Interpolate the solution vector on the boundary 
            # and determine the boundary position on the coordinate line
            if i == 1 # On an x-line
                rb = (X[nr]+αr*dx,Y[yi],Z[zi])
                Ub = el1(αr)*U[nr,yi,zi] + el2(αr)*U[nr-1,yi,zi] + el3(αr)*U[nr-2,yi,zi]
            elseif i == 2 # On a y-line
                rb = (X[xi],Y[nr]+αr*dy,Z[zi])
                Ub = el1(αr)*U[xi,nr,zi] + el2(αr)*U[xi,nr-1,zi] + el3(αr)*U[xi,nr-2,zi]
            elseif i == 3 # On a z-line
                rb = (X[xi],Y[yi],Z[nr]+αr*dz)
                Ub = el1(αr)*U[xi,yi,nr] + el2(αr)*U[xi,yi,nr-1] + el3(αr)*U[xi,yi,nr-2]
            end

            k,ℓ,σ = vectors(rb) # Form boundary basis

            ψxb = Ub.ψx
            ψyb = Ub.ψy
            ψzb = Ub.ψz
            Ψb  = Ub.Ψ

            ∂ψb = @Vec [-Ψb,ψxb,ψyb,ψzb]

            Upb = @einsum k[α]*∂ψb[α]
            U0b = @einsum σ[α,β]*∂ψb[β]

            UmBC = -Upb

            ΨBC  = -(Upb + UmBC)/sqrt(2)
            ∂ψBC =  U0b - k*UmBC - ℓ*Upb

            if ri[i] == nr
                ε = el1(αr)/h11(αr)/ds[i]
            elseif ri[i] == nr-1
                ε = el2(αr)/h22(αr)/ds[i]
            else#if ri[i] == nl-2
                ε = el3(αr)/h33(αr)/ds[i]
            end

            ∂tΨ  += ε*(ΨBC - Ψb)
            ∂tψx += ε*(∂ψBC[2] - ψxb)
            ∂tψy += ε*(∂ψBC[3] - ψyb)
            ∂tψz += ε*(∂ψBC[4] - ψzb)

        end
        
    end

    # # Apply Boundary Conditions in a special fashion on the corners
    # if i in (1,ns[1]) ⊻ j in (1,ns[2]) ⊻ k in (1,ns[3])

    # end

    ∂tU = WaveCell(∂tψ,∂tψx,∂tψy,∂tψz,∂tΨ)

    if iter == 1
        U1t = Uxyz
        Uwxyz = U1t + dt*∂tU
    elseif iter == 2
        U1t = U1[xi,yi,zi]
        U2t = Uxyz
        Uwxyz = (3/4)*U1t + (1/4)*U2t + (1/4)*dt*∂tU
    elseif iter == 3
        U1t = U1[xi,yi,zi]
        U2t = Uxyz
        Uwxyz = (1/3)*U1t + (2/3)*U2t + (2/3)*dt*∂tU
    end

    Uw[xi,yi,zi] = Uwxyz

    else # don't do anything if outside of the boundary

    end


    return

end

##################################################
@views function main()
    # Physics
    lx, ly, lz = 50.0, 50.0, 50.0  # domain extends
    ls = (lx,ly,lz)
    t  = 0.0               # physical start time

    # Numerics
    size = 31*8
    ns = (size,size,size)     # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    nx, ny, nz = ns
    nt         = 1500              # number of timesteps
    nout       = 10                # plotting frequency

    # Derived numerics
    dx, dy, dz = lx/(nx-1), ly/(ny-1), lz/(nz-1) # cell sizes
    dt         = min(dx,dy,dz)/5.1
    ds = (dx,dy,dz,dt)

    # Initial Conditions
    #σ = 2.; x0 = lx/2; y0 = ly/2; z0 = lz/2;
    σ = 2.; x0 = 0.; y0 = 0.; z0 = 0.;
    @inline ψ_init(x,y,z) = exp(-((x-x0)^2+(y-y0)^2+(z-z0)^2)/σ^2)

    @inline ∂xψ(x,y,z) = ForwardDiff.derivative(x -> ψ_init(x,y,z), x)
    @inline ∂yψ(x,y,z) = ForwardDiff.derivative(y -> ψ_init(x,y,z), y)
    @inline ∂zψ(x,y,z) = ForwardDiff.derivative(z -> ψ_init(x,y,z), z)

    @inline ∂tψ(x,y,z) = 0.

    # Array allocations

    U1 = @zeros(ns..., celltype=WaveCell)
    U2 = @zeros(ns..., celltype=WaveCell)
    U3 = @zeros(ns..., celltype=WaveCell)

    coords = (-lx/2:dx:lx/2, -ly/2:dy:ly/2, -lz/2:dz:lz/2)
    X,Y,Z = coords
    #X, Y, Z = 0:dx:lx, 0:dy:ly, 0:dz:lz

    temp  = zeros(5,ns...)

    for xi in 1:ns[1], yi in 1:ns[2], zi in 1:ns[3]

        x = X[xi]
        y = Y[yi]
        z = Z[zi]

        ψ  = ψ_init(x,y,z)
        ψx =    ∂xψ(x,y,z)
        ψy =    ∂yψ(x,y,z)
        ψz =    ∂zψ(x,y,z)
        Ψ  =    ∂tψ(x,y,z)

        temp[:,xi,yi,zi] .= [ψ,ψx,ψy,ψz,Ψ]

    end

    for i in 1:5
        CellArrays.field(U1,i) .= Data.Array(temp[i,:,:,:])
    end

    copy!(U2.data, U1.data)
    copy!(U3.data, U1.data)

    # ns = ((10,26),(1,ny),(1,nz))

    # αs = ((0.25,0.25),(0.25,0.25),(0.25,0.25))

    # return [-Dx(ψu,U1,ns,αs,i,1,1)/dx - 2*(-lx/2 + (i-1)*dx) for i in ns[1][1]:ns[1][2] ]

    #return 0.

    # Preparation of visualisation
    ENV["GKSwstype"]="nul"; if isdir("viz3D_out")==false mkdir("viz3D_out") end; loadpath = "./viz3D_out/"; anim = Animation(loadpath,String[])
    old_files = readdir(loadpath; join=true)
    for i in 1:length(old_files) rm(old_files[i]) end
    println("Animation directory: $(anim.dir)")
    y_sl       = Int(ceil(ny/2))

    #return @benchmark @parallel $bulk rhs!($U1,$U2,$U3,$ns,$ds,1)

    evec = zeros(0)

    # Time loop
    for it = 1:nt

        if (it==11)  global wtime0 = Base.time()  end

        bulk = (1:nx,1:ny,1:nz)

        # First stage (iter=1)

        @parallel bulk rhs!(U1,U2,U3,ls,ns,ds,1) 
 
        # Second stage (iter=2)
    
        @parallel bulk rhs!(U1,U2,U3,ls,ns,ds,2) 
    
        # Third stage (iter=3)
    
        @parallel bulk rhs!(U1,U2,U3,ls,ns,ds,3) 

        t = t + dt

        # Visualisation
        if mod(it,nout)==0
            A = Array(CellArrays.field(U1,1))[:,y_sl,:]
            heatmap(X, Z, A, aspect_ratio=1, xlims=(X[1],X[end])
            ,ylims=(Z[1],Z[end]),clim=(-0.1,0.1), c=:viridis)
            frame(anim)

            append!(evec,sum(U -> energy_cell(U,ds),U1))
        end


    end

    # Performance
    wtime    = Base.time() - wtime0
    A_eff    = (4*2)/1e9*nx*ny*nz*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: H and dHdτ have to be read and written (dHdτ for damping): 4 whole-array memaccess; B has to be read: 1 whole-array memaccess)
    wtime_it = wtime/(nt-10)                           # Execution time per iteration [s]
    T_eff    = A_eff/wtime_it                          # Effective memory throughput [GB/s]

    @printf("Total steps=%d, time=%1.3e sec (@ T_eff = %1.2f GB/s) \n", nt, wtime, round(T_eff, sigdigits=3))
    gif(anim, "acoustic3D.gif", fps = 15)

    # if USE_GPU GC.gc(true) end

    GC.gc(true)

    return plot((evec)./evec[1], ylim = (0, 2))

end

end