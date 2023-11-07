
    # if iter == 1
    #     U2[x,y] = Uxy + dt*∂tU
    # elseif iter == 2
    #     U3[x,y] = (3/4)*U1[x,y] + (1/4)*Uxy + (1/4)*dt*∂tU
    # elseif iter == 3
    #     U1[x,y] = (1/3)*U1[x,y] + (2/3)*Uxy + (2/3)*dt*∂tU
    # end
# End bit for RK4 low storage

# if iter in (1,2)

#     Un1[x,y] = dt*∂tU

# elseif iter == 3

#     Un1[x,y] = dt*∂tU - 0.5*Un1[x,y]

# elseif iter == 4

#     Un1[x,y] = dt*∂tU + 2*Un1[x,y]

# end

#@parallel_indices (x,y) function add!(A,B::Tuple,b::Tuple)

#     @inbounds A[x,y] = sum(getindex.(B,x,y).*b)

#     return

# end

# @parallel_indices (x,y) function copy!(A,B)

#     @inbounds A[x,y] = B[x,y]

#     return

# end

# @parallel_indices (x,y) function add!(A,B,b::Number)

#     @inbounds A[x,y] += b*B[x,y]

#     return

# end

# @parallel_indices (x,y) function combine!(A,B,C,b::Number,c::Number)

#     @inbounds A[x,y] = b*B[x,y] + c*C[x,y]

#     return

# end

# @parallel_indices (x,y) function update!(A,B,C,b::Number,c::Number)

#     @inbounds A[x,y] += b*B[x,y] + c*C[x,y]

#     return

# end
# @inline function u(x,y,U) # scalar gradient-flux

#     # Slice the State Vector 
#     Uxy = U[x,y]

#     # Give names to stored arrays from the state vector
#     g = Uxy.g 
#     dr = Uxy.dr   
#     dθ = Uxy.dθ  
#     P = Uxy.P 

#     # Unpack the metric into indiviual components
#     gtt,gtr,gtθ,_,grr,grθ,_,gθθ,_,_ = g.data

#     # Calculate lapse and shift
#     det  = grr*gθθ - grθ^2
#     βr = (gtr*gθθ-gtθ*grθ)/det
#     βθ = (gtθ*grr-gtr*grθ)/det
#     α  = sqrt(-gtt + grr*βr^2 + 2*grθ*βr*βθ + gθθ*βθ^2)

#     return βr*dr + βθ*dθ - α*P

# end

# @inline function vr(x,y,U) # r component of the divergence-flux

#     # Slice the State Vector 
#     Uxy = U[x,y]

#     # Give names to stored arrays from the state vector
#     g  = Uxy.g 
#     dr = Uxy.dr   
#     dθ = Uxy.dθ  
#     P  = Uxy.P 

#     # Unpack the metric into indiviual components
#     gtt,gtr,gtθ,_,grr,grθ,_,gθθ,_,gϕϕ = g.data

#     # Calculate lapse and shift
#     det  = grr*gθθ - grθ^2
#     βr = (gtr*gθθ-gtθ*grθ)/det
#     βθ = (gtθ*grr-gtr*grθ)/det
#     α  = sqrt(-gtt + grr*βr^2 + 2*grθ*βr*βθ + gθθ*βθ^2)

#     # Calculate inverse components
#     γirr = gθθ/det; γirθ = -grθ/det;

#     return Aθ2(rootγ,U,x,y)*(βr*P - α*(γirr*dr + γirθ*dθ))
    
# end

# @inline function vθ(x,y,U) # θ component of the divergence-flux

#     # Slice the State Vector 
#     Uxy = U[x,y]

#     # Give names to stored arrays from the state vector
#     g  = Uxy.g 
#     dr = Uxy.dr   
#     dθ = Uxy.dθ  
#     P  = Uxy.P 

#     # Unpack the metric into indiviual components
#     gtt,gtr,gtθ,_,grr,grθ,_,gθθ,_,gϕϕ = g.data

#     # Calculate lapse and shift
#     det  = grr*gθθ - grθ^2
#     βr = (gtr*gθθ-gtθ*grθ)/det
#     βθ = (gtθ*grr-gtr*grθ)/det
#     α  = sqrt(-gtt + grr*βr^2 + 2*grθ*βr*βθ + gθθ*βθ^2)

#     # Calculate inverse components
#     γirθ = -grθ/det; γiθθ = grr/det;

#     #return (βθ*P - α*(γirθ*dr + γiθθ*dθ))
#     return Aθ2(rootγ,U,x,y)*(βθ*P - α*(γirθ*dr + γiθθ*dθ))
    
# end

# @inline function rootγ(x,y,U)

#     # Slice the State Vector 
#     Uxy = U[x,y]

#     # Give names to stored arrays from the state vector
#     g  = Uxy.g 

#     # Unpack the metric into indiviual components
#     _,_,_,_,grr,grθ,_,gθθ,_,gϕϕ = g.data

#     det  = grr*gθθ - grθ^2

#     return sqrt(det*gϕϕ)
# end

# @inline function Base.:+(A::StateVector{T},B::StateVector{T}) where T
#     g  = A.g  .+ B.g
#     dx = A.dx .+ B.dx
#     dθ = A.dθ .+ B.dθ
#     P  = A.P  .+ B.P
#     return StateVector{T}(g,dr,dθ,P)
# end

# @inline function Base.:-(A::StateVector{T},B::StateVector{T}) where T
#     g  = A.g  .- B.g
#     dr = A.dr .- B.dr
#     dθ = A.dθ .- B.dθ
#     P  = A.P  .- B.P
#     return StateVector{T}(g,dr,dθ,P)
# end

# @inline function Base.:*(a::Number,A::StateVector{T}) where T
#     g  = a*A.g
#     dr = a*A.dr
#     dθ = a*A.dθ
#     P  = a*A.P
#     return StateVector{T}(g,dr,dθ,P)
# end

# @inline function Base.:*(A::StateVector{T},B::StateVector{T}) where T
#     g  = A.g.*B.g
#     dr = A.dr.*B.dr
#     dθ = A.dθ.*B.dθ
#     P  = A.P.*B.P
#     return StateVector{T}(g,dr,dθ,P)
# end
if c1 p=-1 else p=1 end

st_ = @Vec [0.0,p*sin(θ[x,y]),0.0,p*cos(θ[x,y])]

snorm = @einsum gi[μ,ν]*st_[μ]*st_[ν]

s_ = st_/sqrt(snorm)

s = @einsum gi[μ,ν]*s_[ν]

r_vec = @Vec [p*sin(θ[x,y]),0.0,p*cos(θ[x,y])]

rvec = @einsum γi3[i,j]*r_vec[j]

rnorm = @einsum γ[i,j]*rvec[i]*rvec[j]

rhat = rvec/sqrt(rnorm)

#s = @Vec [0.0,rhat[1],rhat[2],rhat[3]]

r_hat = r_vec/sqrt(rnorm)

Θt = @Vec [0.0,s_[4],0.0,-s_[2]]

Θnorm = @einsum g[μ,ν]*Θt[μ]*Θt[ν]

Θ = Θt/sqrt(Θnorm)

θvec = @Vec [r_vec[3],0.0,-r_vec[1]]

θnorm = @einsum γ[i,j]*θvec[i]*θvec[j]

θhat = θvec/sqrt(θnorm)

#θ4 = @Vec [0.0,θhat[1],θhat[2],θhat[3]]

θ_hat = @einsum γ[i,j]*θhat[j]

#cp =  α - βx*r_hat[1] - βz*r_hat[3]
cm = -α - βx*r_hat[1] - βz*r_hat[3]
c0 =    - βx*r_hat[1] - βz*r_hat[3]

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
Θ_  = @einsum g[μ,α]*Θ[α]
#k_ = @einsum g[μ,α]*k[α]

#σ = StateTensor((μ,ν) -> gi[μ,ν] + k[μ]*l[ν] + l[μ]*k[ν])

σ_ = StateTensor((μ,ν) -> g[μ,ν] + n_[μ]*n_[ν] - s_[μ]*s_[ν])

σ = @einsum gi[μ,α]*gi[ν,β]*σ_[α,β]

σm = @einsum gi[μ,α]*σ_[ν,α] # mixed indices (raised second index)

#δ4 = one(SymmetricFourthOrderTensor{4})
δ = one(SymmetricSecondOrderTensor{4})

γp = @einsum δ[μ,ν] + n_[μ]*n[ν] 

Q4 = SymmetricFourthOrderTensor{4}(
    (μ,ν,α,β) -> σ_[μ,ν]*σ[α,β]/2 - 2*l_[μ]*σm[ν,α]*k[β] + l_[μ]*l_[ν]*k[α]*k[β]
) # Four index constraint projector (indices down down up up)

Q3 = Tensor{Tuple{@Symmetry{4,4},4}}(
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

∂tα = -0.5*α*(@einsum n[μ]*n[ν]*∂tg[μ,ν])

∂tβ = α*(@einsum γi[α,μ]*n[ν]*∂tg[μ,ν]) # result is a 4-vector

∂t∂tg = (βx*∂tdx + βz*∂tdz - α*∂tP) + (∂tβ[2]*dx + ∂tβ[4]*dz - ∂tα*P)

∂t∂g = Tensor{Tuple{4,@Symmetry{4,4}}}((σ,μ,ν) -> (σ==1 ? ∂t∂tg[μ,ν] : σ==2 ? ∂tdx[μ,ν] : σ==3 ? 0.0 : σ==4 ? ∂tdz[μ,ν] : @assert false))

# ∂∂g = SymmetricFourthOrderTensor{4}(
#     (ϵ,σ,μ,ν) -> ϵ==1 ? (σ==1 ? ∂t∂tg[μ,ν] : σ==2 ? ∂tdx[μ,ν] : σ==3 ? 0.0 : σ==4 ? ∂tdz[μ,ν] : @assert false) :
#                  ϵ==2 ? (σ==1 ? ∂tdx[μ,ν]  : σ==2 ? ∂xdx[μ,ν] : σ==3 ? 0.0 : σ==4 ? ∂xdz[μ,ν] : @assert false) :
#                  ϵ==3 ? (σ==1 ? 0.0        : σ==2 ? 0.0       : σ==3 ? 0.0 : σ==4 ? 0.0       : @assert false) :
#                  ϵ==4 ? (σ==1 ? ∂tdz[μ,ν]  : σ==2 ? ∂xdz[μ,ν] : σ==3 ? 0.0 : σ==4 ? ∂zdz[μ,ν] : @assert false) : @assert false
# ) # use this to do the spatial derivatives of the constraints the way the paper does it

∂tΓ  = Tensor{Tuple{4,@Symmetry{4,4}}}((σ,μ,ν) -> 0.5*(∂t∂g[ν,μ,σ] + ∂t∂g[μ,ν,σ] - ∂t∂g[σ,μ,ν]))   

∂tH = Vec{4}((∂Hxy[1,:]...))
∂xH = Vec{4}((∂Hxy[2,:]...))
∂zH = Vec{4}((∂Hxy[4,:]...))

∂tC = (@einsum gi[ϵ,σ]*∂tΓ[λ,ϵ,σ] - gi[μ,ϵ]*gi[ν,σ]*Γ[λ,μ,ν]*∂tg[ϵ,σ]) - ∂tH

# set up finite differencing for the constraints, by defining a function
# that calculates the constraints for any x and y index. This
# might not be the best idea, but should work.

∂xC = DxC(constraints,U,r,θ,ns,_ds,x,y) - ∂xH + 0.5*γ2*(@einsum (n_[σ]*gi[μ,ν]*Cx[μ,ν] - n[ν]*Cx[σ,ν]))
∂zC = DzC(constraints,U,r,θ,ns,_ds,x,y) - ∂zH + 0.5*γ2*(@einsum (n_[σ]*gi[μ,ν]*Cx[μ,ν] - n[ν]*Cx[σ,ν]))

F = (∂tC - βx*∂xC - βz*∂zC)/α + γ2*(@einsum γi[μ,ν]*C2[μ,ν,λ] - 0.5*γp[λ,σ]*gi[μ,ν]*C2[σ,μ,ν])

∂Cm = F + rhat[1]*∂xC + rhat[3]*∂zC

c4xz = Dx2(fdz,U,r,θ,ns,_ds,x,y) - Dz2(fdx,U,r,θ,ns,_ds,x,y)
c4zx = -c4xz

∂tUp = ∂tP + rhat[1]*∂tdx + rhat[3]*∂tdz# - γ2*∂tg   
∂tUm = ∂tP - rhat[1]*∂tdx - rhat[3]*∂tdz# - γ2*∂tg
∂tU0 = θhat[1]*∂tdx + θhat[3]*∂tdz

#∂tU0 = ()∂tdx + ∂tdz

∂tUmb = @einsum Q4[μ,ν,α,β]*∂tUm[α,β]
∂tUmb -= sqrt(2)*cm*(@einsum Q3[μ,ν,α]*∂Cm[α]) # Constraint preserving BCs

∂tU0b = ∂tU0 + c0*(rhat[1]*θhat[3]*c4zx + rhat[3]*θhat[1]*c4xz)

#∂tUmb = @einsum O[μ,ν,α,β]*∂th[α,β] # Incoming Gravitational waveform

# Time derivatives are OVERWRITTEN here, but still depends on evolution values
∂tP  = 0.5*(∂tUp + ∂tUmb)
∂tdx = 0.5*(∂tUp - ∂tUmb)*r_hat[1] + ∂tU0b*θ_hat[1] 
∂tdz = 0.5*(∂tUp - ∂tUmb)*r_hat[3] + ∂tU0b*θ_hat[3] 

# @parallel_indices (x,y) function rhs!(U1,U2,U3,H,∂H_sym,r,θ,ns,dt,_ds,iter)

#     # Explicit slices from main memory
#     if iter == 1
#         U = U1
#         Uxy = U[x,y]
#     elseif iter == 2
#         U = U2
#         Uxy = U[x,y]
#     elseif iter == 3
#         U = U3
#         Uxy = U[x,y]
#     end

#     Hxy = H[x,y]; ∂H_symxy = ∂H_sym[x,y];

#     # Machinery for rescaling

#     # Λ
#     # Λi

#     # Give names to stored arrays from the state vector
#     g  = Uxy.g 
#     dr = Uxy.dr   
#     dθ = Uxy.dθ  
#     P  = Uxy.P 

#     # Unpack the tensor type into indiviual components
#     gtt,  gtr, gtθ,_, grr, grθ,_, gθθ,_, gϕϕ = g.data
#     drtt,drtr,drtθ,_,drrr,drrθ,_,drθθ,_,drϕϕ = dr.data
#     dθtt,dθtr,dθtθ,_,dθrr,dθrθ,_,dθθθ,_,dθϕϕ = dθ.data
#     Ptt,  Ptr, Ptθ,_, Prr, Prθ,_, Pθθ,_, Pϕϕ = P.data

#     # Calculate lapse and shift
#     det  = grr*gθθ - grθ^2
#     βr   = (gtr*gθθ-gtθ*grθ)/det
#     βθ   = (gtθ*grr-gtr*grθ)/det

#     β = @Vec [0.0,βr,βθ,0.0]

#     # if x==1 && y==1
#     #     println(x," ",y," ",grr," ",gθθ," ",gϕϕ," ")
#     # end
#     if -gtt + grr*βr^2 + 2*grθ*βr*βθ + gθθ*βθ^2 < 0
#         println(x," ",y," ",grr," ",gθθ," ",gϕϕ," ")
#     end
#     α  = sqrt(-gtt + grr*βr^2 + 2*grθ*βr*βθ + gθθ*βθ^2)

#     γ1 = -1.
#     γ2 = 0.

#     # Calculate time derivative of the metric
#     ∂tgμν = ((1+γ1)*(βr*Dr2(fg,U,ns,x,y)*_ds[1] + βθ*Dθ2(fg,U,ns,x,y)*_ds[2]) 
#             - γ1*(βr*dr + βθ*dθ) - α*P)

#     # Time derivative of the metric (get rid of this?)
#     ∂tgtt = βr*drtt + βθ*dθtt - α*Ptt; ∂tgtr = βr*drtr + βθ*dθtr - α*Ptr;
#     ∂tgtθ = βr*drtθ + βθ*dθtθ - α*Ptθ; ∂tgrr = βr*drrr + βθ*dθrr - α*Prr;
#     ∂tgrθ = βr*drrθ + βθ*dθrθ - α*Prθ; ∂tgθθ = βr*drθθ + βθ*dθθθ - α*Pθθ;
#     ∂tgϕϕ = βr*drϕϕ + βθ*dθϕϕ - α*Pϕϕ; 

#     ∂tg =  βr*dr + βθ*dθ - α*P
    
#     # Calculate inverse components
#     γirr = gθθ/det; γirθ = -grθ/det; γiθθ = grr/det; γiϕϕ = 1.0/gϕϕ;

#     γi = StateTensor((0.0,0.0,0.0,0.0,γirr,γirθ,0.0,γiθθ,0.0,γiϕϕ))

#     nt = 1.0/α; nx = -βr/α; ny = -βθ/α; 

#     n = @Vec [nt,nx,ny,0.0]

#     n_ = @Vec [-α,0.0,0.0,0.0]

#     # gitt = -nt^2; gitr = -nt*nx; girr = γirr-nx^2;
#     # gitθ = -nt*ny; girθ = γirθ-nx*ny; giθθ = γiθθ-ny^2;
#     # giϕϕ = γiϕϕ

#     # rootγ = sqrt(det*gϕϕ)

#     gi = symmetric(@einsum γi[μ,ν] - n[μ]*n[ν])

#     #gi = StateTensor((gitt,gitr,gitθ,0.0,girr,girθ,0.0,giθθ,0.0,giϕϕ))

#     ∂g = Tensor{Tuple{4,@Symmetry{4,4}}}(
#         (∂tgtt,drtt,dθtt,0.0,∂tgtr,drtr,dθtr,0.0,∂tgtθ,drtθ,dθtθ,0.0,0.0,0.0,0.0,0.0,
#          ∂tgrr,drrr,dθrr,0.0,∂tgrθ,drrθ,dθrθ,0.0,0.0,0.0,0.0,0.0,
#          ∂tgθθ,drθθ,dθθθ,0.0,0.0,0.0,0.0,0.0,∂tgϕϕ,drϕϕ,dθϕϕ,0.0))

#     Γ = Tensor{Tuple{4,@Symmetry{4,4}}}(
#         (σ,μ,ν) -> 0.5*(∂g[ν,μ,σ] + ∂g[μ,ν,σ] - ∂g[σ,μ,ν])
#         )

#     C = @einsum Hxy[μ] - gi[ϵ,σ]*Γ[μ,ϵ,σ]

#     # if (x == 100 && y == 3 && iter==4) 
#     #     display(C) 
#     # end

#     # Define Stress energy tensor and trace 
#     T = zero(StateTensor)
#     Tt = 0.

#     δ = one(StateTensor)
#     γ0 = 1.

#     ∂tPμν  = 8*pi*Tt*g - 16*pi*T + 2*∂H_symxy

#     ∂tPμν -= @einsum gi[ϵ,σ]*Hxy[ϵ]*∂g[μ,ν,σ]

#     ∂tPμν -= @einsum gi[ϵ,σ]*Hxy[ϵ]*∂g[ν,μ,σ]

#     ∂tPμν += @einsum 2*gi[ϵ,σ]*gi[λ,ρ]*∂g[λ,ϵ,μ]*∂g[ρ,σ,ν]

#     ∂tPμν -= @einsum 2*gi[ϵ,σ]*gi[λ,ρ]*Γ[μ,ϵ,λ]*Γ[ν,σ,ρ]

#     # ∂tPμν -= @einsum 0.5*γi[i,j]*∂g[k,i,j]*γi[k,l]*∂g[l,μ,ν]

#     #∂tPμν += @einsum -1.0*(δ[ϵ,μ]*n_[ν] + δ[ϵ,ν]*n_[μ] - g[μ,ν]*n[ϵ])*C[ϵ]

#     ∂tPμν *= α

#     ∂tPμν -= @einsum 0.5*γi[i,j]*∂tg[i,j]*P[μ,ν]

#     # ∂tPμν += @einsum 0.5*γi[i,j]*β[k]*∂g[k,i,j]*P[μ,ν]

#     # ∂tPμν -= @einsum (0.5*α)*γi[i,j]*∂g[k,i,j]*γi[k,l]*∂g[l,μ,ν]

#     #∂tPμν += Div_∂(vr,vθ,k,ns,_ds,x,y)

#     #∂tPμν += Div(vr,vθ,U,ns,_ds,x,y)

#     ∂tPμν += Div(vr,vθ,U,ns,_ds,x,y)

#     ∂tdrμν = Dr2(u,U,ns,x,y)*_ds[1] + α*γ2*(Dr2(fg,U,ns,x,y)*_ds[1] - dr)

#     ∂tdθμν = Dθ2(u,U,ns,x,y)*_ds[2] + α*γ2*(Dθ2(fg,U,ns,x,y)*_ds[2] - dθ)

#     ∂tPμν = symmetric(∂tPμν)

#     ∂tU = StateVector(∂tgμν,∂tdrμν,∂tdθμν,∂tPμν)

#     if x==100 && y==1 && iter == 1
#         println("")
#         #display(vθ(x,y,U))
#         display(∂tgμν)
#         display(∂tdrμν)
#         display(∂tdθμν)
#         display(∂tPμν)
#         println("")
#     end

#     #∂tU = zero(StateVector{Float64})

#     # End bit for SSP RK3

#     if iter == 1
#         U2[x,y] = Uxy + dt*∂tU
#     elseif iter == 2
#         U3[x,y] = (3/4)*U1[x,y] + (1/4)*Uxy + (1/4)*dt*∂tU
#     elseif iter == 3
#         U1[x,y] = (1/3)*U1[x,y] + (2/3)*Uxy + (2/3)*dt*∂tU
#     end

#     # End bit for RK4 low storage

#     # if iter in (1,2)

#     #     Un1[x,y] = dt*∂tU

#     # elseif iter == 3

#     #     Un1[x,y] = dt*∂tU - 0.5*Un1[x,y]

#     # elseif iter == 4

#     #     Un1[x,y] = dt*∂tU + 2*Un1[x,y]

#     # end


#     return
    
# end



macro sum(indices, expr)
    (parse_sum(expr,indices))
end

function parse_sum(expr,indices)
    @assert indices isa Expr
    @assert indices.head === :tuple
    #@assert expr.head === :+=
    inds = indices.args
    @assert !isempty(inds)
    @assert all(ind isa Symbol for ind in inds)
    n = length(inds)
    terms = []
    iters = inds
    for iters in Iterators.product([1:4 for i in 1:n]...)
        assignments = Expr(:block,[Expr(:(=), esc(inds[i]), iters[i]) for i in 1:n]...)
        body = esc(expr)
        expr′ = Expr(:let, assignments, body)
        push!(terms, expr′)
    end
    # terms2 = []
    # for part in Iterators.partition(terms,4)
    #     push!(terms2,Expr(:call, :+, part...))
    # end
    # res = Expr(:call, :+, terms2...)
    # if n <= 2
    #     res = Expr(:call, :+, terms...)
    # else
    #     terms2 = []
    #     for part in Iterators.partition(terms,16)
    #         push!(terms2,Expr(:call, :+, part...))
    #     end
    #     res = Expr(:call, :+, terms2...)
    # end
    res = Expr(:block, terms...)
    return res
end


# function parse_sum(expr,indices)
#     @assert indices isa Expr
#     @assert indices.head === :tuple
#     inds = indices.args
#     @assert !isempty(inds)
#     @assert all(ind isa Symbol for ind in inds)
#     n = length(inds)
#     terms = []
#     iters = inds
#     for iters in Iterators.product([1:4 for i in 1:n]...)
#         assignments = Expr(:block,[Expr(:(=), esc(inds[i]), iters[i]) for i in 1:n]...)
#         body = esc(expr)
#         expr′ = Expr(:let, assignments, body)
#         push!(terms, expr′)
#     end
#     # terms2 = []
#     # for part in Iterators.partition(terms,4)
#     #     push!(terms2,Expr(:call, :+, part...))
#     # end
#     # res = Expr(:call, :+, terms2...)
#     if n <= 2
#         res = Expr(:call, :+, terms...)
#     else
#         terms2 = []
#         for part in Iterators.partition(terms,16)
#             push!(terms2,Expr(:call, :+, part...))
#         end
#         res = Expr(:call, :+, terms2...)
#     end
#     return res
# end


#     iters = indices
#     #ranges = 
#     if expr isa Expr
#        @inline f(indices...) = expr
#        +([f(iters...) for iters in Iterators.product([1:4 for i in 1:n]...)]...)
#     end
#     # Abort if we find something unexpected
#     println(typeof(expr))
#     @assert false
# end

 
     # plot(Array(θ[2,1:res:end]), Array(∂ₜU.x[var][1:res:end,1])); 
    # ylims!(-100, 100)
    # frame(anim)
    # println(mean(∂ₜU.x[var][1:res:end,1]))
 
 function RK4!(Un,Un1,H,∂H,r,θ,ns,dt,_ds)

    rxy = r[x,y]; θxy = θ[x,y];

    Hxy = H[x,y]
    ∂Hxy = ∂H[x,y]

    Unxy = Un[x,y] # Explicit read from system memory

    #Un1xy = Un1[x,y]

    #########################################
    # Begin 4th order Runge-Kutta algorithm

    Un1xy = Unxy

    ∂tUxy = rhs(Unxy,Hxy,∂Hxy,rxy,θxy,ns,_ds)
    
    #BC_r!(∂ₜU,Un,gauge,t,rmax,θ[1,:],dr)

    Un1xy = Un1xy + (dt/6)*∂tUxy

    kxy = Unxy + (dt/2)*∂tUxy

    ∂tUxy = rhs(kxy,Hxy,∂Hxy,rxy,θxy,ns,_ds)

    #BC_r!(∂ₜU,k,gauge,t,rmax,θ[1,:],dr)

    Un1xy = Un1xy + (dt/3)*∂tUxy

    kxy = Unxy + (dt/2)*∂tUxy

    ∂tUxy = rhs(kxy,Hxy,∂Hxy,rxy,θxy,ns,_ds)

    # BC_r!(∂ₜU,k,gauge,t,rmax,θ[1,:],dr)

    Un1xy = Un1xy + (dt/3)*∂tUxy
    
    kxy = Unxy + dt*∂tUxy

    ∂tUxy = rhs(kxy,Hxy,∂Hxy,rxy,θxy,ns,_ds)

    # BC_r!(∂ₜU,k,gauge,t,rmax,θ[1,:],dr)

    Un1xy = Un1xy + (dt/6)*∂tUxy

    # End 4th order Runge-Kutta algorithm
    #######################################

    Un1[x,y] = Un1xy # Explicit save to system memory

    return

end
 # @inline function Base.zero(::Type{StateStorage{T}}) where T
#     g  = NTuple{7,T}((0.0,0.0,0.0,0.0,0.0,0.0,0.0))
#     dr = NTuple{7,T}((0.0,0.0,0.0,0.0,0.0,0.0,0.0))
#     dθ = NTuple{7,T}((0.0,0.0,0.0,0.0,0.0,0.0,0.0))
#     P  = NTuple{7,T}((0.0,0.0,0.0,0.0,0.0,0.0,0.0))
#     return StateStorage(g,dr,dθ,P)
# end

# @inline function add!(A::StateVector{T},B::StateVector{T},c::T)
#     A.g.x  .+= c*B.g.x
#     A.dr.x .+= c*B.dr.x
#     A.dθ.x .+= c*B.dθ.x
#     A.P.x  .+= c*B.P.x
# end

# @inline function lin_comb!(A::StateVector{T},B::StateVector{T},C::StateVector{T},c::T)
#     A.g.x  .= B.g.x  .+ c*C.g.x
#     A.dr.x .= B.dr.x .+ c*C.dr.x
#     A.dθ.x .= B.dθ.x .+ c*C.dθ.x
#     A.P.x  .= B.P.x  .+ c*C.P.x
# end

# Sample analytic functions to the grid
# function sample!(f, fun, ns, r, θ, μ...)

#     f .= Data.Array([fun(r[i,j],θ[i,j],μ...) for i in 1:ns[1], j in 1:ns[2]])

# end
 # @inline function index2linear(μ::Int,ν::Int)
#     (μ,ν) == (1,1) && return 1
#     (μ,ν) == (1,2) && return 2
#     (μ,ν) == (2,1) && return 2
#     (μ,ν) == (1,3) && return 3
#     (μ,ν) == (3,1) && return 3
#     (μ,ν) == (2,2) && return 4 
#     (μ,ν) == (2,3) && return 5
#     (μ,ν) == (3,2) && return 5
#     (μ,ν) == (3,3) && return 6
#     (μ,ν) == (4,4) && return 7  
#     @assert false
# end

# Define how to index this custom tensor type so that it acts like a matrix
# @inline function Base.getindex(A::Axisymmetric2Tensor{T},μ::Int,ν::Int)
#     nonzeroQ(μ,ν) && return A.x[index2linear(μ,ν)]
#     return zero(T)
# end

# @inline function Base.setindex(A::Axisymmetric2Tensor,val,μ::Int,ν::Int)
#     nonzeroQ(μ,ν) && return Axisymmetric2Tensor(setindex(A.x,val,index2linear(μ,ν)))
#     @assert false
# end

# @inline function Base.getindex(A::Axisymmetric3Tensor{T},σ::Int,μ::Int,ν::Int)
#     σ == 1 && return A.t[μ,ν]
#     σ == 2 && return A.r[μ,ν]
#     σ == 3 && return A.θ[μ,ν]
#     return zero(T)
# end

# Define how to allocate these types, either mutable or immutable
# @inline function Base.zeros(::Type{Axisymmetric2Tensor},::Type{T},mutable=true) where T
#     mutable && return Axisymmetric2Tensor{T}(@MVector [0.,0.,0.,0.,0.,0.,0.])
#     return Axisymmetric2Tensor{T}(@SVector [0.,0.,0.,0.,0.,0.,0.])
# end

# @inline function Base.zeros(::Type{StateVector},::Type{T},mutable=true) where T
#     return StateVector{T}([Base.zeros(Axisymmetric2Tensor,T,mutable) for i in 1:4]...)
# end
  
  
  
  # Sample initial state vector onto the grid
    # sample!(gtt, g_init, ns, r, θ, 1, 1)
    # sample!(gtr, g_init, ns, r, θ, 1, 2)
    # sample!(gtθ, g_init, ns, r, θ, 1, 3)
    # sample!(grr, g_init, ns, r, θ, 2, 2)
    # sample!(grθ, g_init, ns, r, θ, 2, 3)
    # sample!(gθθ, g_init, ns, r, θ, 3, 3)
    # sample!(gϕϕ, g_init, ns, r, θ, 4, 4)

    # sample!(Ptt, P_init, ns, r, θ, 1, 1)
    # sample!(Ptr, P_init, ns, r, θ, 1, 2)
    # sample!(Ptθ, P_init, ns, r, θ, 1, 3)
    # sample!(Prr, P_init, ns, r, θ, 2, 2)
    # sample!(Prθ, P_init, ns, r, θ, 2, 3)
    # sample!(Pθθ, P_init, ns, r, θ, 3, 3)
    # sample!(Pϕϕ, P_init, ns, r, θ, 4, 4)

    # sample!(drtt, d_init, ns, r, θ, 2, 1, 1)
    # sample!(drtr, d_init, ns, r, θ, 2, 1, 2)
    # sample!(drtθ, d_init, ns, r, θ, 2, 1, 3)
    # sample!(drrr, d_init, ns, r, θ, 2, 2, 2)
    # sample!(drrθ, d_init, ns, r, θ, 2, 2, 3)
    # sample!(drθθ, d_init, ns, r, θ, 2, 3, 3)
    # sample!(drϕϕ, d_init, ns, r, θ, 2, 4, 4)

    # sample!(dθtt, d_init, ns, r, θ, 3, 1, 1)
    # sample!(dθtr, d_init, ns, r, θ, 3, 1, 2)
    # sample!(dθtθ, d_init, ns, r, θ, 3, 1, 3)
    # sample!(dθrr, d_init, ns, r, θ, 3, 2, 2)
    # sample!(dθrθ, d_init, ns, r, θ, 3, 2, 3)
    # sample!(dθθθ, d_init, ns, r, θ, 3, 3, 3)
    # sample!(dθϕϕ, d_init, ns, r, θ, 3, 4, 4)

    # sample!(Ht, fH, ns, r, θ, 1)
    # sample!(Hr, fH, ns, r, θ, 2)
    # sample!(Hθ, fH, ns, r, θ, 3)

    # sample!(∂rHt, ∂H, ns, r, θ, 2, 1)
    # sample!(∂rHr, ∂H, ns, r, θ, 2, 2)
    # sample!(∂rHθ, ∂H, ns, r, θ, 2, 3)
    # sample!(∂θHt, ∂H, ns, r, θ, 3, 1)
    # sample!(∂θHr, ∂H, ns, r, θ, 3, 2)
    # sample!(∂θHθ, ∂H, ns, r, θ, 3, 3)


# Define custom struct that stores a StaticVector, but indexes like a proper matrix,
# taking advantage of the symmetric nature of the tensors, as well as the simplifications
# that come from axisymmetry in space
# struct Axisymmetric2Tensor{T <: Data.Number}
#     x::NTuple{7,T}
# end

# struct Axisymmetric2Tensor{T <: Data.Number}
#     x::StaticArray{Tuple{7}, T, 1}
# end

# Define a 3rd rank version of the above
# struct Axisymmetric3Tensor{T <: Data.Number}
#     t::Axisymmetric2Tensor{T}
#     r::Axisymmetric2Tensor{T}
#     θ::Axisymmetric2Tensor{T}
# end

# Define a container that holds the state vector in our system of PDEs
# struct StateVector{T <: Data.Number}
#     # φ::T
#     # ψr::T
#     # ψθ::T
#     # Π::T
#     g::Axisymmetric2Tensor{T}
#     dr::Axisymmetric2Tensor{T}
#     dθ::Axisymmetric2Tensor{T}
#     P::Axisymmetric2Tensor{T}
# end
 
 
 # Unsliced version of lapse and shift for finite differencing
    # βr(x,y) = @part (x,y) (gtr*gθθ-gtθ*grθ)/(grr*gθθ-grθ^2)
    # βθ(x,y) = @part (x,y) (gtθ*grr-gtr*grθ)/(grr*gθθ-grθ^2)
    # α(x,y)  = sqrt(-gtt[x,y] + grr[x,y]*βr(x,y)^2 + 2*grθ[x,y]*βr(x,y)*βθ(x,y) + gθθ[x,y]*βθ(x,y)^2)

    #@inline β(i,x,y) = (βr(x,y),βθ(x,y),0.)[i-1]

    # Define an unsliced version of the state vector for use in finite differencing
    # @inline g(μ,ν,x,y)   = ((gtt,gtr,gtθ,0.),(gtr,grr,grθ,0.),(gtθ,grθ,gθθ,0.),(0.,0.,0.,gϕϕ))[μ][ν][x,y]
    # @inline P(μ,ν,x,y)   = ((Ptt,Ptr,Ptθ,0.),(Ptr,Prr,Prθ,0.),(Ptθ,Prθ,Pθθ,0.),(0.,0.,0.,Pϕϕ))[μ][ν][x,y]
    # @inline d(i,μ,ν,x,y) = if i==4 0. else 
    #                     (((drtt,drtr,drtθ,0.),(drtr,drrr,drrθ,0.),(drtθ,drrθ,drθθ,0.),(0.,0.,0.,drϕϕ)),
    #                      ((dθtt,dθtr,dθtθ,0.),(dθtr,dθrr,dθrθ,0.),(dθtθ,dθrθ,dθθθ,0.),(0.,0.,0.,dθϕϕ)))[i-1][μ][ν][x,y]
    # end                    
    # sqrt of the metric determinant
    #rootγ(x,y) = @part (x,y) sqrt((grr*gθθ-grθ^2)*gϕϕ)
 
 
 
 #=
    @inbounds for μ in 1:4, ν in 1:4

        if zeroQ(μ,ν) && lowerQ(μ,ν) continue end
        
        #temporaries
        #∂tgμν = 0.; ∂tPμν = 0.; ∂tdrμν = 0.; ∂tdθμν = 0.; 

        # Calculate the metric time derivative

        ∂tgμν += ∂g[1,μ,ν]
        
        # Calculate the actual right hand side of Einstein's equations
        # starting with the source-like terms
        
        #∂tPμν += 8*pi*g[μ,ν]*Tt - 16*pi*T(μ,ν) + ∂H[μ,ν] + ∂H[ν,μ]

        #675μs

        #∂tPμν = @sum (ϵ,σ) gi[ϵ,σ]*g[ϵ,σ]

        #@sum (ϵ,σ,λ,ρ) ∂tPμν += gi[ϵ,σ]*gi[λ,ρ]*g[ϵ,ρ]*g[λ,σ]

        #fast
        # @sum (ϵ,σ,λ,ρ) ∂tPμν += gi[ϵ,σ]*gi[λ,ρ]*∂g[λ,ϵ,μ]*∂g[ρ,σ,ν]

        # @sum (ϵ,σ,λ,ρ) ∂tPμν -= gi[ϵ,σ]*gi[λ,ρ]*Γ[μ,ϵ,λ]*Γ[ν,σ,ρ]
        
        # #slow
        # @sum (ϵ,σ) ∂tPμν += gi(ϵ,σ)*∂g(ν,μ,σ)*H(ϵ)
        # @sum (ϵ,σ) ∂tPμν += gi(ϵ,σ)*∂g(μ,ν,σ)*H(ϵ)

        #∂tP(μ,ν)[x,y] += αxy*temp

        # for ϵ in 1:4, σ in 1:4, λ in 1:4, ρ in 1:4
        #         ∂tPμν += gi[ϵ,σ]*gi[λ,ρ]*(∂g[λ,ϵ,μ]*∂g[ρ,σ,ν] - Γ[μ,ϵ,λ]*Γ[ν,σ,ρ])
        # end

        for ϵ in 1:4, σ in 1:4
            if zeroQ(ϵ,σ) continue end
            for λ in 1:4, ρ in 1:4
                if zeroQ(λ,ρ) continue end
                ∂tPμν += gi[ϵ,σ]*gi[λ,ρ]*(∂g[λ,ϵ,μ]*∂g[ρ,σ,ν] - Γ[μ,ϵ,λ]*Γ[ν,σ,ρ])
            end
        end

        # ∂tP(μ,ν)[x,y] += αxy*temp

        # Multiply by lapse at the end for efficency
        # ∂tP(μ,ν)[x,y] *= αxy

        # Finish by calculating all of the principle parts

        # ∂tP(μ,ν)[x,y]   += -Div(v,rootγ,ns,_ds,x,y,μ,ν)

        # ∂td(2,μ,ν)[x,y] = -Grad(u,ns,x,y,2,μ,ν)*_ds[1]

        # ∂td(3,μ,ν)[x,y] = -Grad(u,ns,x,y,3,μ,ν)*_ds[2]

        # ∂tg = setindex(∂tg,∂tgμν,μ,ν)
        # ∂tP = setindex(∂tP,∂tPμν,μ,ν)

    end 
    =#
    # gi = Axisymmetric2Tensor(SA[gitt,gitr,gitθ,girr,girθ,giθθ,giϕϕ])

    #Calculate the Christoffel symbols (all lower indices)
    # Γttt = 0.5*∂tgttxy; Γttr = 0.5*drttxy; Γttθ = 0.5*dθttxy; Γtrr = drtrxy-0.5*∂tgrrxy; 
    # Γtrθ = 0.5*(drtθxy+dθtrxy-∂tgrθxy); Γtθθ = dθtθxy-0.5*∂tgθθxy; Γtϕϕ = -0.5*∂tgϕϕxy;

    # Γrtt = ∂tgtrxy-0.5*drttxy; Γrtr = 0.5*∂tgrrxy; Γrtθ = 0.5*(∂tgrθxy+dθtrxy-drtθxy);
    # Γrrr = 0.5*drrrxy; Γrrθ = 0.5*drrθxy; Γrθθ = dθrθxy-0.5*drθθxy; Γrϕϕ = -0.5*drϕϕxy;

    # Γθtt = ∂tgtθxy-0.5*dθttxy; Γθtr = 0.5*(drtθxy+∂tgtθxy-dθtrxy); Γθtθ = 0.5*∂tgθθxy;
    # Γθrr = drrθxy-0.5*dθrrxy; Γθrθ = 0.5*drθθxy; Γθθθ = 0.5*dθθθxy; Γθϕϕ = -0.5*dθϕϕxy;

    # Γt  = Axisymmetric2Tensor((Γttt,Γttr,Γttθ,Γtrr,Γtrθ,Γtθθ,Γtϕϕ))
    # Γr  = Axisymmetric2Tensor((Γrtt,Γrtr,Γrtθ,Γrrr,Γrrθ,Γrθθ,Γrϕϕ))
    # Γθ  = Axisymmetric2Tensor((Γθtt,Γθtr,Γθtθ,Γθrr,Γθrθ,Γθθθ,Γθϕϕ))

    #Γ = Axisymmetric3Tensor(Γt,Γr,Γθ)
    
    #gi = Axisymmetric2Tensor((gitt,gitr,gitθ,girr,girθ,giθθ,giϕϕ))


@parallel_indices (y) function BC_r!(∂ₜU, U,gauge,t,r,θ, dr)
        

    for l in 1:length(∂ₜU.x)
        ∂ₜU.x[l][1,y] = 0.
        ∂ₜU.x[l][end,y] = 0.
    end

    # φ,ψr,ψθ,Π = U.x
    # ∂ₜφ,∂ₜψr,∂ₜψθ,∂ₜΠ = ∂ₜU.x
    # γrr,γθθ,rootγ,α,βr = metric.x


    # # Upper indices
    # nr = -1/sqrt(γrr[1,j])
    # nθ = 0

    # qr =  nθ*sqrt(γθθ[1,j])/sqrt(γrr[1,j])
    # qθ = -nr*sqrt(γrr[1,j])/sqrt(γθθ[1,j])

    # kr = -cos(θ[j])/sqrt(γrr[1,j])
    # kθ =  sin(θ[j])/sqrt(γθθ[1,j])

    # s = abs(kr)/(2*dr)

    # # ∂ₜΠ[1,j]  +=  s*(Upk/2 - Π[1,j])
    # # ∂ₜψr[1,j] +=  s*(γrr[1,j]*kr*Upk/2 - ψr[1,j])
    # # ∂ₜψθ[1,j] +=  s*(γθθ[1,j]*kθ*Upk/2 - ψθ[1,j])

    # Upn = ∂ₜΠ[1,j] + nr*∂ₜψr[1,j] + nθ*∂ₜψθ[1,j] 
    # #Umn = ∂ₜΠ[1,j] - nr*∂ₜψr[1,j] - nθ*∂ₜψθ[1,j] 

    # U0n = qr*∂ₜψr[1,j] + qθ*∂ₜψθ[1,j] 

    # # Boundary Condition:
    # if abs(Upn)>10*eps() && ((Upn^2-U0n^2)/(Upn^2+U0n^2)) > 0
    #     Umn = U0n^2/Upn
    # else
    #     Umn = Upn
    # end

    # # ∂ₜΠ[1,j]  +=  s*((Upn + Umn)/2 - Π[1,j])
    # # ∂ₜψr[1,j] +=  s*((Upn - Umn)*γrr[1,j]*nr/2 + U0n*γrr[1,j]*qr - ψr[1,j])
    # # ∂ₜψθ[1,j] +=  s*((Upn - Umn)*γθθ[1,j]*nθ/2 + U0n*γθθ[1,j]*qθ - ψθ[1,j])

    #  ∂ₜΠ[1,j] =  (Upn + Umn)/2
    # ∂ₜψr[1,j] =  (Upn - Umn)*γrr[1,j]*nr/2 + U0n*γrr[1,j]*qr
    # ∂ₜψθ[1,j] =  (Upn - Umn)*γθθ[1,j]*nθ/2 + U0n*γθθ[1,j]*qθ


    # # ∂ₜΠ[1,j]  +=  s*(Upk/2 - Π[1,j])
    # # ∂ₜψr[1,j] +=  s*(γrr[1,j]*kr*Upk/2 - ψr[1,j])
    # # ∂ₜψθ[1,j] +=  s*(γθθ[1,j]*kθ*Upk/2 - ψθ[1,j])

    # # if kr <= 0
    # #     ∂ₜΠ[1,j]  +=  s*(Upk/2 - Π[1,j])
    # #     ∂ₜψr[1,j] +=  s*(γrr[1,j]*kr*Upk/2 - ψr[1,j])
    # #     ∂ₜψθ[1,j] +=  s*(γθθ[1,j]*kθ*Upk/2 - ψθ[1,j])
    # # else
    # #     ∂ₜΠ[1,j]  +=  s*((Upn + Umn)/2 - Π[1,j])
    # #     ∂ₜψr[1,j] +=  s*((Upn - Umn)*γrr[1,j]*nr/2 + U0n*γrr[1,j]*qr - ψr[1,j])
    # #     ∂ₜψθ[1,j] +=  s*((Upn - Umn)*γθθ[1,j]*nθ/2 + U0n*γθθ[1,j]*qθ - ψθ[1,j])
    # # end

    # # Upper indices
    # nr = 1/sqrt(γrr[end,j])
    # nθ = 0

    # qr =  nθ*sqrt(γθθ[end,j])/sqrt(γrr[end,j])
    # qθ =  nr*sqrt(γrr[end,j])/sqrt(γθθ[end,j])

    # Upn = ∂ₜΠ[end,j] + nr*∂ₜψr[end,j] + nθ*∂ₜψθ[end,j] 
    # #Umn = ∂ₜΠ[1,j] - nr*∂ₜψr[1,j] - nθ*∂ₜψθ[1,j] 

    # U0n = qr*∂ₜψr[end,j] + qθ*∂ₜψθ[end,j] 

    # # Boundary Condition:
    # # if abs(Upn)>10*eps() && ((Upn^2-U0n^2)/(Upn^2+U0n^2)) > 0
    # #     Umn = U0n^2/Upn
    # # else
    # #     Umn = 0
    # # end

    # μ=16.5
    # σ=1
    # A=1
    # kr = -cos(θ[j])/sqrt(γrr[end,j])
    # kθ =  sin(θ[j])/sqrt(γθθ[end,j])

    # c = α[end,j] + sqrt(γrr[end,j])*(-cos(θ[j]))*βr[end,j]
    # x = r*cos(θ[j]) + c*t
    # if (μ-σ)<x<(μ+σ) && kr<=0 
    #     Upk = -16*A*((x-μ)^2-σ^2)^2*(7*(x-μ)^2-σ^2)/σ^8
    #     ∂ₜΠ[end,j] = Upk/2
    #     ∂ₜψr[end,j] = γrr[end,j]*kr*(Upk/2)
    #     ∂ₜψθ[end,j] = γθθ[end,j]*kθ*(Upk/2)
    # elseif abs(Upn)>10*eps() && ((Upn^2-U0n^2)/(Upn^2+U0n^2)) > 0
    #     Umn = U0n^2/Upn
    #     ∂ₜΠ[end,j] =  (Upn + Umn)/2
    #     ∂ₜψr[end,j] =  (Upn - Umn)*γrr[end,j]*nr/2 + U0n*γrr[end,j]*qr
    #     ∂ₜψθ[end,j] =  (Upn - Umn)*γθθ[end,j]*nθ/2 + U0n*γθθ[end,j]*qθ
    # else
    #     Umn = -Upn
    #     ∂ₜΠ[end,j] =  (Upn + Umn)/2
    #     ∂ₜψr[end,j] =  (Upn - Umn)*γrr[end,j]*nr/2 + U0n*γrr[end,j]*qr
    #     ∂ₜψθ[end,j] =  (Upn - Umn)*γθθ[end,j]*nθ/2 + U0n*γθθ[end,j]*qθ
    # end



    return
end
 
 
 # Define a sliced version of the state vector in index notation for use in the right hand side calculation
    # @inline g(μ,ν)   = (@SMatrix [gttxy gtrxy gtθxy 0.; gtrxy grrxy grθxy 0.; gtθxy grθxy gθθxy 0.; 0. 0. 0. gϕϕxy])[μ,ν]
    # @inline P(μ,ν)   = (@SMatrix [Pttxy Ptrxy Ptθxy 0.; Ptrxy Prrxy Prθxy 0.; Ptθxy Prθxy Pθθxy 0.; 0. 0. 0. Pϕϕxy])[μ,ν]
    # @inline d(i,μ,ν) = if i==4 0. elseif i==2
    #                    (@SMatrix [drttxy drtrxy drtθxy 0.; drtrxy drrrxy drrθxy 0.; drtθxy drrθxy drθθxy 0.; 0. 0. 0. drϕϕxy])[μ,ν]
    #                    elseif i==3
    #                    (@SMatrix [dθttxy dθtrxy dθtθxy 0.; dθtrxy dθrrxy dθrθxy 0.; dθtθxy dθrθxy dθθθxy 0.; 0. 0. 0. dθϕϕxy])[μ,ν]
    #                    end
function rhs!(∂ₜU,U,Hm,∂Hm,rm,θm,ns,_ds,x,y)

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
    ( ∂ₜgtt,  ∂ₜgtr,  ∂ₜgtθ,  ∂ₜgrr,  ∂ₜgrθ,  ∂ₜgθθ,  ∂ₜgϕϕ,
      ∂ₜPtt,  ∂ₜPtr,  ∂ₜPtθ,  ∂ₜPrr,  ∂ₜPrθ,  ∂ₜPθθ,  ∂ₜPϕϕ,
      ∂ₜdrtt, ∂ₜdrtr, ∂ₜdrtθ, ∂ₜdrrr, ∂ₜdrrθ, ∂ₜdrθθ, ∂ₜdrϕϕ,
      ∂ₜdθtt, ∂ₜdθtr, ∂ₜdθtθ, ∂ₜdθrr, ∂ₜdθrθ, ∂ₜdθθθ, ∂ₜdθϕϕ  ) = ∂ₜU.x

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

    # Metric determinant and inverse components
    γirr = gθθxy/det; γirθ = -grθxy/det; γiθθ = grrxy/det; γiϕϕ = 1.0/gϕϕxy;
    nt = 1.0/αxy; nx = -βrxy/αxy; ny = -βθxy/αxy; 

    # Put gauge functions into index notation
    H(μ) = (Htxy,Hrxy,Hθxy,0.)[μ]

    # Same with the derivatives
    ∂H(μ,ν) = ((∂tHtxy,∂tHrxy,∂tHθxy,0.),(∂rHtxy,∂rHrxy,∂rHθxy,0.),(∂θHtxy,∂θHrxy,∂θHθxy,0.),(0.,0.,0.,0.))[μ][ν]

    # Unsliced version of lapse and shift for finite differencing
    # βr(x,y) = @part (x,y) (gtr*gθθ-gtθ*grθ)/(grr*gθθ-grθ^2)
    # βθ(x,y) = @part (x,y) (gtθ*grr-gtr*grθ)/(grr*gθθ-grθ^2)
    # α(x,y)  = sqrt(-gtt[x,y] + grr[x,y]*βr(x,y)^2 + 2*grθ[x,y]*βr(x,y)*βθ(x,y) + gθθ[x,y]*βθ(x,y)^2)

    # Put shift into index notation
    @inline β(i) = (βrxy,βθxy,0.)[i-1]
    #@inline β(i,x,y) = (βr(x,y),βθ(x,y),0.)[i-1]

    # Define inverse metric in index notation
    @inline γi(μ,ν)   = ((0.,0.,0.,0.),(0.,γirr,γirθ,0.),(0.,γirθ,γiθθ,0.),(0.,0.,0.,γiϕϕ))[μ][ν]
    @inline n(μ)      = (nt,nx,ny,0.)[μ]
    @inline gi(μ,ν)   = γi(μ,ν) - n(μ)*n(ν)

    # Define a sliced version of the state vector in index notation for use in the right hand side calculation
    #@inline g(μ,ν)   = ((gttxy,gtrxy,gtθxy,0.)[ν],(gtrxy,grrxy,grθxy,0.)[ν],(gtθxy,grθxy,gθθxy,0.)[ν],(0.,0.,0.,gϕϕxy)[ν])[μ]

    @inline g(μ,ν)    = μ==1 ? (ν==1 ? gttxy : ν==2 ? gtrxy : ν==3 ? gtθxy : ν==4 ? 0.    : @assert false) :
                        μ==2 ? (ν==1 ? gtrxy : ν==2 ? grrxy : ν==3 ? grθxy : ν==4 ? 0.    : @assert false) :
                        μ==3 ? (ν==1 ? gtθxy : ν==2 ? grθxy : ν==3 ? gθθxy : ν==4 ? 0.    : @assert false) :
                        μ==4 ? (ν==1 ? 0.    : ν==2 ? 0.    : ν==3 ? 0.    : ν==4 ? gϕϕxy : @assert false) : @assert false

    @inline P(μ,ν)    = μ==1 ? (ν==1 ? Pttxy : ν==2 ? Ptrxy : ν==3 ? Ptθxy : ν==4 ? 0.    : @assert false) :
                        μ==2 ? (ν==1 ? Ptrxy : ν==2 ? Prrxy : ν==3 ? Prθxy : ν==4 ? 0.    : @assert false) :
                        μ==3 ? (ν==1 ? Ptθxy : ν==2 ? Prθxy : ν==3 ? Pθθxy : ν==4 ? 0.    : @assert false) :
                        μ==4 ? (ν==1 ? 0.    : ν==2 ? 0.    : ν==3 ? 0.    : ν==4 ? Pϕϕxy : @assert false) : @assert false

    #@inline P(μ,ν)   = ((Pttxy,Ptrxy,Ptθxy,0.)[ν],(Ptrxy,Prrxy,Prθxy,0.)[ν],(Ptθxy,Prθxy,Pθθxy,0.)[ν],(0.,0.,0.,Pϕϕxy)[ν])[μ]
    @inline d(i,μ,ν) = if i==4 0. else 
                        (((drttxy,drtrxy,drtθxy,0.),(drtrxy,drrrxy,drrθxy,0.),(drtθxy,drrθxy,drθθxy,0.),(0.,0.,0.,drϕϕxy)),
                         ((dθttxy,dθtrxy,dθtθxy,0.),(dθtrxy,dθrrxy,dθrθxy,0.),(dθtθxy,dθrθxy,dθθθxy,0.),(0.,0.,0.,dθϕϕxy)))[i-1][μ][ν]
    end

    # Define an unsliced version of the state vector for use in finite differencing
    # @inline g(μ,ν,x,y)   = ((gtt,gtr,gtθ,0.),(gtr,grr,grθ,0.),(gtθ,grθ,gθθ,0.),(0.,0.,0.,gϕϕ))[μ][ν][x,y]
    # @inline P(μ,ν,x,y)   = ((Ptt,Ptr,Ptθ,0.),(Ptr,Prr,Prθ,0.),(Ptθ,Prθ,Pθθ,0.),(0.,0.,0.,Pϕϕ))[μ][ν][x,y]
    # @inline d(i,μ,ν,x,y) = if i==4 0. else 
    #                     (((drtt,drtr,drtθ,0.),(drtr,drrr,drrθ,0.),(drtθ,drrθ,drθθ,0.),(0.,0.,0.,drϕϕ)),
    #                      ((dθtt,dθtr,dθtθ,0.),(dθtr,dθrr,dθrθ,0.),(dθtθ,dθrθ,dθθθ,0.),(0.,0.,0.,dθϕϕ)))[i-1][μ][ν][x,y]
    # end

    # Define unsliced time derivative of the state vector in index notation
    #@inline ∂ₜg(μ,ν)   = ((∂ₜgtt,∂ₜgtr,∂ₜgtθ,0.),(∂ₜgtr,∂ₜgrr,∂ₜgrθ,0.),(∂ₜgtθ,∂ₜgrθ,∂ₜgθθ,0.),(0.,0.,0.,∂ₜgϕϕ))[μ][ν]
    # @inline ∂ₜg(μ,ν)   = (@SMatrix [∂ₜgtt ∂ₜgtr ∂ₜgtθ 0.; ∂ₜgtr ∂ₜgrr ∂ₜgrθ 0.; ∂ₜgtθ ∂ₜgrθ ∂ₜgθθ 0.; 0. 0. 0. ∂ₜgϕϕ])[μ,ν]
    
    # @inline ∂ₜP(μ,ν)   = ((∂ₜPtt,∂ₜPtr,∂ₜPtθ,0.),(∂ₜPtr,∂ₜPrr,∂ₜPrθ,0.),(∂ₜPtθ,∂ₜPrθ,∂ₜPθθ,0.),(0.,0.,0.,∂ₜPϕϕ))[μ][ν]
    # @inline ∂ₜd(i,μ,ν) = if i==4 0. else 
    #                     (((∂ₜdrtt,∂ₜdrtr,∂ₜdrtθ,0.),(∂ₜdrtr,∂ₜdrrr,∂ₜdrrθ,0.),(∂ₜdrtθ,∂ₜdrrθ,∂ₜdrθθ,0.),(0.,0.,0.,∂ₜdrϕϕ)),
    #                      ((∂ₜdθtt,∂ₜdθtr,∂ₜdθtθ,0.),(∂ₜdθtr,∂ₜdθrr,∂ₜdθrθ,0.),(∂ₜdθtθ,∂ₜdθrθ,∂ₜdθθθ,0.),(0.,0.,0.,∂ₜdθϕϕ)))[i-1][μ][ν]
    # end

    # sqrt of the metric determinant
    #rootγ(x,y) = @part (x,y) sqrt((grr*gθθ-grθ^2)*gϕϕ)

    # Define "fluxes" to be directly acted on by the finite differencing operators
    #@inline u(x,y,μ,ν)   = β(2,x,y)*d(2,μ,ν,x,y) + β(3,x,y)*d(3,μ,ν,x,y) - α(x,y)*P(μ,ν,x,y)
    # Here, i is a contravariant spatial index, μ and ν are spacetime covariant indices
    #@inline v(x,y,i,μ,ν) = α(x,y)*(γi(i,2)*d(2,μ,ν,x,y)+γi(i,3)*d(3,μ,ν,x,y)) + β(i,x,y)*P(μ,ν,x,y)

    # Define all metric derivatives in terms of the state vector in index notation
    @inline ∂g(σ,μ,ν) = if σ==1 β(2)*d(2,μ,ν) + β(3)*d(3,μ,ν) - αxy*P(μ,ν) else d(σ,μ,ν) end

    # Define completely covariant Christoffel symbols
    @inline Γ(σ,μ,ν) = (∂g(ν,μ,σ) + ∂g(μ,ν,σ) - ∂g(σ,μ,ν))/2.

    # Define Stress energy tensor and trace 
    @inline T(μ,ν) = 0.
    T = 0.

    # 680μs

    # Begin calculation of the right hand side of the evolution equations
    # Only set μ and ν to the values they need to calculate. Due to axisymmetry and
    # the symmetric nature of the metric, there are only 7 (μ,ν) pairs here.
    for k in 1:1
        
        (μ,ν) = ((1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(4,4))[k]

        # Calculate the metric time derivative

        #∂ₜgtt[x,y] = αxy*g(μ,ν)
        ∂ₜU.x[k][x,y] = -αxy*g(μ,ν)
        #∂ₜU.x[k][x,y] = -αxy*P(μ,ν)
        #∂ₜU.x[k][x,y] = -αxy*P(1,4)

        #println(P(μ,ν))

        #∂ₜg(μ,ν)[x,y] = -αxy*g(μ,ν)

        # 300ms
        #=

        for i in 2:3
            ∂ₜg(μ,ν)[x,y] += β(i)*d(i,μ,ν)
        end
        #1.15s



        # Calculate the actual right hand side of Einstein's equations
        # starting with the source-like terms

        ∂ₜP(μ,ν)[x,y] = 8*pi*g(μ,ν)*T #- 16*pi*T(μ,ν) + ∂H(μ,ν) + ∂H(ν,μ)

        for ϵ in 1:4, σ in 1:4
            
            ∂ₜP(μ,ν)[x,y] += gi(ϵ,σ)*(∂g(μ,ν,σ)*H(ϵ) + ∂g(ν,μ,σ)*H(ϵ)) 

            for λ in 1:4, ρ in 1:4
                ∂ₜP(μ,ν)[x,y] += 2*gi(ϵ,σ)*gi(λ,ρ)*(∂g(λ,ϵ,μ)*∂g(ρ,σ,ν) - Γ(μ,ϵ,λ)*Γ(ν,σ,ρ))
            end

        end

        # Multiply by lapse at the end for efficency
        ∂ₜP(μ,ν)[x,y] *= αxy

        for i in 2:4, j in 2:4
            ∂ₜP(μ,ν)[x,y] += -0.5*γi(i,j)*∂g(1,i,j)*P(μ,ν)
        end

        # Finish by calculating all of the principle parts

        # ∂ₜP(μ,ν)[x,y]   += -Div(v,rootγ,ns,_ds,x,y,μ,ν)

        # ∂ₜd(2,μ,ν)[x,y] = -Grad(u,ns,x,y,2,μ,ν)*_ds[1]

        # ∂ₜd(3,μ,ν)[x,y] = -Grad(u,ns,x,y,3,μ,ν)*_ds[2]

        =#
        
    end 
    
    # 210ms
    
    return
    
end
 
 
 # @inline function Dr2(f::AbstractArray,i,j,n) 
#     if i in 2:n-1
#         0.5*(-f[i-1,j]+f[i+1,j])
#     elseif  i==1
#         -f[1,j] + f[2,j]
#     elseif i==n 
#         -f[n-1,j] + f[n,j]
#     end
# end

# @inline function Dθ2(f::AbstractArray,i,j,n,parity=1.0)
#     if j in 2:n-1
#         0.5*(-f[i,j-1]+f[i,j+1])
#     elseif  j==1
#         -0.5*parity*f[i,1]+0.5*f[i,2]
#         # 0.5*f[i,1]+0.5*f[i,2]
#         # -f[i,1] + f[i,2]
#     elseif j==n 
#         #-f[i,end-1]
#         -0.5*f[i,n-1]+0.5*parity*f[i,n] 
#     end
# end


macro within(macroname::String, A::Symbol)
    if     macroname == "@all"    esc( :($ix<=size($A,1)     && $iy<=size($A,2)     ) )
    elseif macroname == "@inn"    esc( :($ix<=size($A,1)-2*b && $iy<=size($A,2)-2*b ) )
    elseif macroname == "@inn_x"  esc( :($ix<=size($A,1)-2*b && $iy<=size($A,2)     ) )
    elseif macroname == "@inn_y"  esc( :($ix<=size($A,1)     && $iy<=size($A,2)-2*b ) )
    elseif macroname == "@D_r"    esc( :($ix<=size($A,1)     && $iy<=size($A,2)     ) )
    elseif macroname == "@D_θ"    esc( :($ix<=size($A,1)     && $iy<=size($A,2)     ) )
    else error("unkown macroname: $macroname. If you want to add your own assignement macros, 
        overwrite the macro 'within(macroname::String, A::Symbol)'; 
        to still use the exising macro within as well call 
        ParallelStencil.FiniteDifferences{1|2|3}D.@within(macroname, A) at the end.")
    end
end
@inline function Grad2(Vr::Function,Vθ::Function,γ::Function,dr,dθ,i,j,nr,nθ)  

    if abs(γ(i,j)) < 10*eps()
        Dr2(Vr,i,j,nr)/dr 
    else
        γVr(i,j) = γ(i,j)*Vr(i,j)
        γVθ(i,j) = γ(i,j)*Vθ(i,j)
        (Dr2(γVr,i,j,nr)/dr + Dθ2(γVθ,i,j,nθ)/dθ)/γ(i,j) 
    end

end

# @inline fgtt(r,θ) = -(1 - 2*M/r)
# @inline fgtr(r,θ) = sign*2*M/r
# @inline fgtθ(r,θ) = 0.
# @inline fgrr(r,θ) = 1 + 2*M/r
# @inline fgrθ(r,θ) = 0.
# @inline fgθθ(r,θ) = r^2
# @inline fgϕϕ(r,θ) = (r^2)*sin(θ)^2

# @inline fdrtr(r,θ) = ForwardDiff.derivative(r -> fgtr(r,θ), r)
# @inline fdrtθ(r,θ) = ForwardDiff.derivative(r -> fgtθ(r,θ), r)
# @inline fdrrr(r,θ) = ForwardDiff.derivative(r -> fgrr(r,θ), r)
# @inline fdrrθ(r,θ) = ForwardDiff.derivative(r -> fgrθ(r,θ), r)
# @inline fdrθθ(r,θ) = ForwardDiff.derivative(r -> fgθθ(r,θ), r)
# @inline fdrϕϕ(r,θ) = ForwardDiff.derivative(r -> fgϕϕ(r,θ), r)

# @inline fdθtt(r,θ) = ForwardDiff.derivative(θ -> fgtt(r,θ), θ)
# @inline fdθtr(r,θ) = ForwardDiff.derivative(θ -> fgtr(r,θ), θ)
# @inline fdθtθ(r,θ) = ForwardDiff.derivative(θ -> fgtθ(r,θ), θ)
# @inline fdθrr(r,θ) = ForwardDiff.derivative(θ -> fgrr(r,θ), θ)
# @inline fdθrθ(r,θ) = ForwardDiff.derivative(θ -> fgrθ(r,θ), θ)
# @inline fdθθθ(r,θ) = ForwardDiff.derivative(θ -> fgθθ(r,θ), θ)
# @inline fdθϕϕ(r,θ) = ForwardDiff.derivative(θ -> fgϕϕ(r,θ), θ)

# @inline f∂tgtt(r,θ) = 0.
# @inline f∂tgtr(r,θ) = 0.

# @inline f∂tgtt(r,θ) = 0.
# @inline f∂tgtr(r,θ) = 0.
# @inline f∂tgtθ(r,θ) = 0.
# @inline f∂tgrr(r,θ) = 0.
# @inline f∂tgrθ(r,θ) = 0.
# @inline f∂tgθθ(r,θ) = 0.
# @inline f∂tgϕϕ(r,θ) = 0.

# @inline fPtt(r,θ) = 0.
# @inline f∂tgtr(r,θ) = 0.
# @inline f∂tgtθ(r,θ) = 0.
# @inline f∂tgrr(r,θ) = 0.
# @inline f∂tgrθ(r,θ) = 0.
# @inline f∂tgθθ(r,θ) = 0.
# @inline f∂tgϕϕ(r,θ) = 0.
@parallel_indices (x,y) function rhs!(∂ₜU,U,gauge,rm,θm,nr,nθ,_dr,_dθ)

r = rm[x,y]; θ = θm[x,y];

γrr  = U.x[1][x,y];   γθθ = U.x[2][x,y];   γϕϕ = U.x[3][x,y];
Krr  = U.x[4][x,y];   Kθθ = U.x[5][x,y];   Kϕϕ = U.x[6][x,y];
frrr = U.x[7][x,y];  frθθ = U.x[8][x,y];  frϕϕ = U.x[9][x,y];
fθrr = U.x[10][x,y];  fθT = U.x[11][x,y]; fθϕϕ = U.x[12][x,y];
#γrr,γθθ,γϕϕ,Krr,Kθθ,Kϕϕ,frrr,frθθ,frϕϕ,fθrr,fθθθ,fθϕϕ = U.x
∂ₜγrr,∂ₜγθθ,∂ₜγϕϕ,∂ₜKrr,∂ₜKθθ,∂ₜKϕϕ,∂ₜfrrr,∂ₜfrθθ,∂ₜfrϕϕ,∂ₜfθrr,∂ₜfθθθ,∂ₜfθϕϕ = ∂ₜU.x

# Regularization
fθθθ = fθT*cos(θ)/sin(θ);

αt     = gauge.x[1][x,y];  ∂rlnαt = gauge.x[2][x,y];  ∂2ᵣlnαt = gauge.x[3][x,y];
∂θlnαt = gauge.x[4][x,y]; ∂2θlnαt = gauge.x[5][x,y];  ∂rθlnαt = gauge.x[6][x,y];
βr     = gauge.x[7][x,y];    ∂rβr = gauge.x[8][x,y];    ∂2ᵣβr = gauge.x[9][x,y];

γuprr = 1.0/γrr; γupθθ = 1.0/γθθ; γupϕϕ = 1.0/γϕϕ;

#αt,∂rlnαt,∂2ᵣlnαt,∂θlnαt,∂2θlnαt,fθT,βr,∂rβr,∂2ᵣβr = gauge.x

@inline γ(i,j) = if i==j if i==1 γrr elseif i==2 γθθ elseif i==3 γϕϕ else 0. end else 0. end
@inline K(i,j) = if i==j if i==1 Krr elseif i==2 Kθθ elseif i==3 Kϕϕ else 0. end else 0. end
#@inline K(i,j) = if i==j if i==1 Krr*fKrr(f∂ₜγrr,r,θ) elseif i==2 Kθθ*fKθθ(f∂ₜγθθ,r,θ) elseif i==3 Kϕϕ*fKϕϕ(f∂ₜγϕϕ,r,θ) else 0. end else 0. end
#@inline K(i,j) = if i==j if i==1 fγrr(r,θ) elseif i==2 fγθθ(r,θ) elseif i==3 fγϕϕ(r,θ) else 0. end else 0. end
@inline f(k,i,j) = (if k==1 
                    if i==j 
                        if i==1 frrr elseif i==2 frθθ  elseif i==3 frϕϕ  end 
                    elseif (i==1 && j==2) || (i==2 && j==1)
                        fθrr  + γrr*fθϕϕ/(γϕϕ)
                    else 0. end
                   elseif k==2
                    if i==j
                        if i==1 fθrr elseif i==2 fθθθ elseif i==3 fθϕϕ end 
                    elseif (i==1 && j==2) || (i==2 && j==1)
                        frθθ + γθθ*frϕϕ/(γϕϕ)
                    else 0. end
                   elseif k==3
                    if (i==3 && j==1) || (i==1 && j==3)
                        frϕϕ + γϕϕ*frθθ/(γθθ)
                    elseif (i==3 && j==2) || (i==2 && j==3)
                        fθϕϕ + γϕϕ*fθrr/(γrr)
                    else 0. end
                   else 0. end)

@inline ∂γ(k,i,j) = (if k==1
                        if (i==j) 
                            if     i==1 Dr2(U.x[1],x,y,nr)*_dr
                            elseif i==2 Dr2(U.x[2],x,y,nr)*_dr
                            elseif i==3 Dr2(U.x[3],x,y,nr)*_dr
                            else 0. end 
                        else 0. end
                    elseif k==2 
                        if (i==j) 
                                if i==1 Dθ2(U.x[1],x,y,nθ)*_dθ 
                            elseif i==2 Dθ2(U.x[2],x,y,nθ)*_dθ
                            elseif i==3 Dθ2(U.x[3],x,y,nθ)*_dθ
                            else 0. end 
                        else 0. end
                    else 0. end)

@inline ∂K(k,i,j) = (if k==1
                        if (i==j) 
                        if     i==1 Dr2(U.x[4],x,y,nr)*_dr
                        elseif i==2 Dr2(U.x[5],x,y,nr)*_dr
                        elseif i==3 Dr2(U.x[6],x,y,nr)*_dr
                        else 0. end 
                    else 0. end
                    elseif k==2 
                        if (i==j) 
                            if i==1 Dθ2(U.x[4],x,y,nθ)*_dθ
                        elseif i==2 Dθ2(U.x[5],x,y,nθ)*_dθ
                        elseif i==3 Dθ2(U.x[6],x,y,nθ)*_dθ
                        else 0. end 
                    else 0. end
                    else 0. end)

@inline function ∂f(l,k,i,j)
    if l==1
        if k==1 
            if i==j 
                    if i==1 Dr2(U.x[7],x,y,nr)*_dr#*ffrrr(r,θ) + frrr*f∂rfrrr(r,θ)
                elseif i==2 Dr2(U.x[8],x,y,nr)*_dr#*ffrθθ(r,θ) + frθθ*f∂rfrθθ(r,θ)
                elseif i==3 Dr2(U.x[9],x,y,nr)*_dr#*ffrϕϕ(r,θ) + frϕϕ*f∂rfrϕϕ(r,θ) 
                else 0. end 
            else 0. end
        elseif k==2
            if i==j 
                    if i==1 Dr2(U.x[10],x,y,nr)*_dr#*ffθrr(r,θ) + fθrr*f∂rfθrr(r,θ)
                elseif i==2 Dr2(U.x[11],x,y,nr)*_dr*cos(θ)/sin(θ)
                elseif i==3 Dr2(U.x[12],x,y,nr)*_dr#*ffθϕϕ(r,θ) + fθϕϕ*f∂rfθϕϕ(r,θ)
                else 0. end 
            else 0. end
        else 0. end
    elseif l==2
        if k==1 
            if i==j 
                    if i==1 Dθ2(U.x[7],x,y,nθ)*_dθ#*ffrrr(r,θ) + frrr*f∂θfrrr(r,θ)
                elseif i==2 Dθ2(U.x[8],x,y,nθ)*_dθ#*ffrθθ(r,θ) + frθθ*f∂θfrθθ(r,θ)
                elseif i==3 Dθ2(U.x[9],x,y,nθ)*_dθ#*ffrϕϕ(r,θ) + frϕϕ*f∂θfrϕϕ(r,θ)
                else 0. end 
            else 0. end
        elseif k==2
            if i==j 
                    if i==1 Dθ2(U.x[10],x,y,nθ,-1.0)*_dθ#*ffθrr(r,θ) + fθrr*f∂θfθrr(r,θ)
                elseif i==2 Dθ2(U.x[11],x,y,nθ,-1.0)*_dθ*cos(θ)/sin(θ) - fθT/sin(θ)^2
                elseif i==3 Dθ2(U.x[12],x,y,nθ,-1.0)*_dθ#*ffθϕϕ(r,θ) + fθϕϕ*f∂θfθϕϕ(r,θ)
                else 0. end 
            else 0. end
        else 0. end
    else 0. end
end

# @inline function ∂f(l,k,i,j)
#     if l==1
#         if k==1 
#             if i==j if i==1 Dr2(frrr,x,y,nr)*_dr elseif i==2 Dr2(frθθ,x,y,nr)*_dr elseif i==3 Dr2(frϕϕ,x,y,nr)*_dr else 0. end else 0. end
#         elseif k==2
#             if i==j if i==1 Dr2(fθrr,x,y,nr)*_dr elseif i==2 Dr2(fθθθ,x,y,nr)*_dr elseif i==3 Dr2(fθϕϕ,x,y,nr)*_dr else 0. end else 0. end
#         else 0. end
#     elseif l==2
#         if k==1 
#             if i==j if i==1 Dθ2(frrr,x,y,nθ)*_dθ elseif i==2 Dθ2(frθθ,x,y,nθ)*_dθ elseif i==3 Dθ2(frϕϕ,x,y,nθ)*_dθ else 0. end else 0. end
#         elseif k==2
#             if i==j if i==1 Dθ2(fθrr,x,y,nθ)*_dθ elseif i==2 Dθ2(fθθθ,x,y,nθ)*_dθ elseif i==3 Dθ2(fθϕϕ,x,y,nθ)*_dθ else 0. end else 0. end
#         else 0. end
#     else 0. end
# end
#Dθ2(fθθθ,x,y,nθ)*_dθ
# cos(θ)/sin(θ)*Dθ2(fθT,x,y,nθ)*_dθ + fθT/sin(θ)^2
#∂ₜγrr,∂ₜγθθ,∂ₜγϕϕ,∂ₜKrr,∂ₜKθθ,∂ₜKϕϕ,∂ₜfrrr,∂ₜfrθθ,∂ₜfrϕϕ,∂ₜfθrr,∂ₜfθθθ,∂ₜfθϕϕ = ∂ₜU.x

@inline ∂ₜγ(i,j) = (i==j) ? ( if i==1 ∂ₜγrr elseif i==2 ∂ₜγθθ elseif i==3 ∂ₜγϕϕ end) : @assert false
@inline ∂ₜK(i,j) = (i==j) ? ( if i==1 ∂ₜKrr elseif i==2 ∂ₜKθθ elseif i==3 ∂ₜKϕϕ end) : @assert false
@inline ∂ₜf(k,i,j) = (if k==1 
                    if i==j if i==1 ∂ₜfrrr elseif i==2 ∂ₜfrθθ elseif i==3 ∂ₜfrϕϕ end else @assert false end
                   elseif k==2
                    if i==j if i==1 ∂ₜfθrr elseif i==2 ∂ₜfθθθ elseif i==3 ∂ₜfθϕϕ end else @assert false end
                   else @assert false end )  
                
# Gauge and misc functions

@inline γup(i,j) = if i==j if i==1 γuprr elseif i==2 γupθθ elseif i==3 γupϕϕ else 0. end else 0. end

if γrr*γθθ*γϕϕ < 0 
    println(x," ",y," ",γrr," ",γθθ," ",γϕϕ," ")
end

α = αt*sqrt(γrr*γθθ*γϕϕ)

@inline ∂lnα(i) = if i==1 ∂rlnαt elseif i==2 ∂θlnαt else 0. end
@inline ∂2lnα(i,j) = if (i==1 && j==1) ∂2ᵣlnαt elseif (i==2 && j==2) ∂2θlnαt else 0. end

@inline β(i) = if i==1 βr else 0. end
@inline ∂β(i,j) = if (i==1 && j==1) ∂rβr else 0. end
@inline ∂2β(k,i,j) = if (i==1 && j==1 && k==1) ∂2ᵣβr else 0. end

@inline S(i) = 0.
@inline S(i,j) = 0.
T = 0.

for i in 1:3
    j = i
    ∂ₜγ(i,j)[x,y] = -(2*α)*K(i,j)
    ∂ₜK(i,j)[x,y] = α*(4*pi*(γ(i,j)*T) - 8*pi*S(i,j) - ∂2lnα(i,j) - ∂lnα(i)*∂lnα(j))
    for k in 1:3
        ∂ₜγ(i,j)[x,y] += β(k)*∂γ(k,i,j) + γ(i,k)*∂β(j,k) + γ(j,k)*∂β(i,k)
        ∂ₜK(i,j)[x,y] += β(k)*∂K(k,i,j) + K(i,k)*∂β(j,k) + K(j,k)*∂β(i,k)
        l = k
        #for l in 1:3
        ∂ₜK(i,j)[x,y] += (α*γup(k,l)*( -∂f(l,k,i,j) + K(k,l)*K(i,j) - 2*K(k,i)*K(l,j)
            + (f(i,j,k)+f(j,i,k)-f(k,i,j))*∂lnα(l) + 2*f(k,l,i)*∂lnα(j)
            + 2*f(k,l,j)*∂lnα(i) - 3*(f(i,k,l)*∂lnα(j) + f(j,k,l)*∂lnα(i))))
        for m in 1:3 #, n in 1:3
            n = m
            ∂ₜK(i,j)[x,y] += (α*γup(k,l)*γup(m,n)*(2*f(k,m,i)*(f(l,n,j)-f(n,l,j))+2*f(k,m,n)*f(l,i,j)
                - 2*f(k,m,l)*f(n,i,j) - f(i,k,m)*f(j,l,n) + 2*(f(i,j,k)+f(j,i,k))*(f(l,n,m)-f(n,l,m)) 
                + 2*(f(k,m,i)*f(j,l,n)+f(k,m,j)*f(i,l,n)) - 8*f(k,l,i)*f(m,n,j)
                + 10*(f(k,l,i)*f(j,m,n)+f(k,l,j)*f(i,m,n)) - 13*f(i,k,l)*f(j,m,n)
                + 2*γ(i,j)*f(k,m,n)*∂lnα(l) - 2*γ(i,j)*f(k,m,l)*∂lnα(n)))
        end
        #end
    end

    for k in 1:2
        ∂ₜf(k,i,j)[x,y] = α*(8*pi*(γ(k,i)*S(j)+γ(k,j)*S(i)) - ∂K(k,i,j) - K(i,j)*∂lnα(k))
        for m in 1:3
            n=m
            ∂ₜf(k,i,j)[x,y] += grad(- ∂K(k,i,j) β(m)*∂f(m,k,i,j) )
            ∂ₜf(k,i,j)[x,y] += (f(m,i,j)*∂β(k,m) + f(k,m,j)*∂β(i,m) 
                + f(k,i,m)*∂β(j,m)  + 0.5*(γ(m,i)*∂2β(k,j,m) + γ(m,j)*∂2β(k,i,m)) )
            ∂ₜf(k,i,j)[x,y] += α*γup(m,n)*( 2*K(k,i)*f(j,m,n) + 2*K(k,j)*f(i,m,n)
                - 2*f(m,n,i)*K(j,k) - 2*f(m,n,j)*K(i,k) + K(i,j)*(2*f(m,n,k)-3*f(k,m,n))
                + (K(m,i)*γ(j,k)+K(m,j)*γ(i,k))*∂lnα(n) - K(m,n)*(γ(k,i)*∂lnα(j)+γ(k,j)*∂lnα(i)))
            for p in 1:3
                q=p
                ∂ₜf(k,i,j)[x,y] += α*γup(m,n)*γup(p,q)*(K(m,p)*(γ(k,i)*f(j,q,n)+γ(k,j)*f(i,q,n)
                    - 2*f(q,n,i)*γ(j,k) - 2*f(q,n,j)*γ(i,k))
                    + (γ(k,i)*K(j,m) + γ(k,j)*K(i,m))*(8*f(n,p,q)-6*f(p,q,n)) 
                    + K(m,n)*(4*(f(p,q,i)*γ(j,k)+f(p,q,j)*γ(i,k)) 
                    - 5*(γ(k,i)*f(j,p,q)+γ(k,j)*f(i,p,q))))
            end
        end
    end
    
end 

#Regularization

∂ₜfθθθ[x,y] *= sin(θ)/cos(θ) 

# 210ms

return
end




    #   for i in 1:3
    #     j = i
    #     ∂ₜγ(i,j)[x,y] = -(2*α)*K(i,j)
    #     ∂ₜK(i,j)[x,y] = α*(4*pi*(γ(i,j)*T) - 8*pi*S(i,j) - ∂2lnα(i,j) - ∂lnα(i)*∂lnα(j))
    #     for k in 1:3
    #         ∂ₜγ(i,j)[x,y] += β(k)*∂γ(k,i,j) + γ(i,k)*∂β(j,k) + γ(j,k)*∂β(i,k)
    #         ∂ₜK(i,j)[x,y] += β(k)*∂K(k,i,j) + K(i,k)*∂β(j,k) + K(j,k)*∂β(i,k)
    #         l = k
    #         #for l in 1:3
    #         ∂ₜK(i,j)[x,y] += (α*γup(k,l)*( -∂f(l,k,i,j) + K(k,l)*K(i,j) - 2*K(k,i)*K(l,j)
    #             + (f(i,j,k)+f(j,i,k)-f(k,i,j))*∂lnα(l) + 2*f(k,l,i)*∂lnα(j)
    #             + 2*f(k,l,j)*∂lnα(i) - 3*(f(i,k,l)*∂lnα(j) + f(j,k,l)*∂lnα(i))))
    #         for m in 1:3 #, n in 1:3
    #             n = m
    #             ∂ₜK(i,j)[x,y] += (α*γup(k,l)*γup(m,n)*(2*f(k,m,i)*(f(l,n,j)-f(n,l,j))+2*f(k,m,n)*f(l,i,j)
    #                 - 2*f(k,m,l)*f(n,i,j) - f(i,k,m)*f(j,l,n) + 2*(f(i,j,k)+f(j,i,k))*(f(l,n,m)-f(n,l,m)) 
    #                 + 2*(f(k,m,i)*f(j,l,n)+f(k,m,j)*f(i,l,n)) - 8*f(k,l,i)*f(m,n,j)
    #                 + 10*(f(k,l,i)*f(j,m,n)+f(k,l,j)*f(i,m,n)) - 13*f(i,k,l)*f(j,m,n)
    #                 + 2*γ(i,j)*f(k,m,n)*∂lnα(l) - 2*γ(i,j)*f(k,m,l)*∂lnα(n)))
    #         end
    #         #end
    #     end

    #     for k in 1:2
    #         ∂ₜf(k,i,j)[x,y] = α*(8*pi*(γ(k,i)*S(j)+γ(k,j)*S(i)) - ∂K(k,i,j) - K(i,j)*∂lnα(k))
    #         for m in 1:3
    #             n=m
    #             #∂ₜf(k,i,j)[x,y] +=
    #             ∂ₜf(k,i,j)[x,y] += (β(m)*∂f(m,k,i,j) + f(m,i,j)*∂β(k,m) + f(k,m,j)*∂β(i,m) 
    #                 + f(k,i,m)*∂β(j,m)  + 0.5*(γ(m,i)*∂2β(k,j,m) + γ(m,j)*∂2β(k,i,m)) )
    #             ∂ₜf(k,i,j)[x,y] += α*γup(m,n)*( 2*K(k,i)*f(j,m,n) + 2*K(k,j)*f(i,m,n)
    #                 - 2*f(m,n,i)*K(j,k) - 2*f(m,n,j)*K(i,k) + K(i,j)*(2*f(m,n,k)-3*f(k,m,n))
    #                 + (K(m,i)*γ(j,k)+K(m,j)*γ(i,k))*∂lnα(n) - K(m,n)*(γ(k,i)*∂lnα(j)+γ(k,j)*∂lnα(i)))
    #             for p in 1:3
    #                 q=p
    #                 ∂ₜf(k,i,j)[x,y] += α*γup(m,n)*γup(p,q)*(K(m,p)*(γ(k,i)*f(j,q,n)+γ(k,j)*f(i,q,n)
    #                     - 2*f(q,n,i)*γ(j,k) - 2*f(q,n,j)*γ(i,k))
    #                     + (γ(k,i)*K(j,m) + γ(k,j)*K(i,m))*(8*f(n,p,q)-6*f(p,q,n)) 
    #                     + K(m,n)*(4*(f(p,q,i)*γ(j,k)+f(p,q,j)*γ(i,k)) 
    #                     - 5*(γ(k,i)*f(j,p,q)+γ(k,j)*f(i,p,q))))
    #             end
    #         end
    #     end
        
    # end 
  
  
  
  
  
  # Package components into arrays

    # 6.3μs, 46 allocations 

    # γ[1,1] = γrr[x,y]; γ[2,2] = γθθ[x,y]; γ[3,3] = γϕϕ[x,y];
    # K[1,1] = Krr[x,y]; K[2,2] = Kθθ[x,y]; K[3,3] = Kϕϕ[x,y];

    # # 18ms 49 allocations

    # f[1,1,1] = frrr[x,y]; f[1,2,2] = frθθ[x,y]; f[1,3,3] = frϕϕ[x,y];
    # f[1,1,2] = fθrr[x,y] + γrr[x,y]*fθϕϕ[x,y]/γϕϕ[x,y]; f[1,2,1] = f[1,1,2];
    # f[2,1,1] = fθrr[x,y]; f[2,2,2] = fθθθ[x,y]; f[2,3,3] = fθϕϕ[x,y];
    # f[2,1,2] = frθθ[x,y] + γrr[x,y]*frϕϕ[x,y]/γϕϕ[x,y]; f[2,2,1] = f[2,1,2];
    # f[3,3,1] = frϕϕ[x,y] + γϕϕ[x,y]*frθθ[x,y]/γθθ[x,y]; f[3,1,3] = f[3,3,1];
    # f[3,3,2] = fθϕϕ[x,y] + γϕϕ[x,y]*fθrr[x,y]/γrr[x,y]; f[3,2,3] = f[3,3,2];

    # # 60ms, 49 allocations

    # # Define gauge variables and other misc

    # αt = αtt[x,y]
    # ∂lnα[1] = ∂ᵣlnαt[x,y]
    # ∂lnα[1,1] = ∂2ᵣlnαt[x,y]
    # β[1] = βr[x,y]
    # ∂β[1,1] = ∂ᵣβr[x,y]
    # ∂2β[1,1,1] = ∂2ᵣβr[x,y]       
  
  
  
  
  # Calculate space derivatives

    # ∂γ[1,1,1] = Dr2(γrr,x,y,nr); ∂γ[1,2,2] = Dr2(γθθ,x,y,nr); ∂γ[1,3,3] = Dr2(γϕϕ,x,y,nr);
    # ∂γ[2,1,1] = Dθ2(γrr,x,y,nr); ∂γ[2,2,2] = Dθ2(γθθ,x,y,nr); ∂γ[2,3,3] = Dθ2(γϕϕ,x,y,nr);

    # ∂K[1,1,1] = Dr2(Krr,x,y,nr); ∂K[1,2,2] = Dr2(Kθθ,x,y,nr); ∂K[1,3,3] = Dr2(Kϕϕ,x,y,nr);
    # ∂K[2,1,1] = Dθ2(Krr,x,y,nθ); ∂K[2,2,2] = Dθ2(Kθθ,x,y,nθ); ∂K[2,3,3] = Dθ2(Kϕϕ,x,y,nθ);

    # ∂f[1,1,1,1] = Dr2(frrr,x,y,nr); ∂f[1,1,2,2] = Dr2(frθθ,x,y,nr); ∂f[1,1,3,3] = Dr2(frϕϕ,x,y,nr);
    # ∂f[1,2,1,1] = Dr2(fθrr,x,y,nr); ∂f[1,2,2,2] = Dr2(fθθθ,x,y,nr); ∂f[1,2,3,3] = Dr2(fθϕϕ,x,y,nr);
    # ∂f[2,1,1,1] = Dθ2(frrr,x,y,nθ); ∂f[2,1,2,2] = Dθ2(frθθ,x,y,nθ); ∂f[2,1,3,3] = Dθ2(frϕϕ,x,y,nθ);
    # ∂f[2,2,1,1] = Dθ2(fθrr,x,y,nθ); ∂f[2,2,2,2] = Dθ2(fθθθ,x,y,nθ); ∂f[2,2,3,3] = Dθ2(fθϕϕ,x,y,nθ);
    
         
         
         
         
         # @einsum ∂ₜK[i,j] = ( β[k]*∂K[k,i,j] - α*γup[l,k]*∂f[l,k,i,j]  #Principle Part
        #     + α*( γup[k,l]*(K[k,l]*K[i,j]-2*K[k,i]*K[l,j])
        #     + γup[k,l]*γup[m,n]*(2*f[k,m,i]*(f[l,n,j]-f[n,l,j])+2*f[k,m,n]*f[l,i,j]
        #     - 2*f[k,m,l]*f[n,i,j] - f[i,k,m]*f[j,l,n] + 2*(f[i,j,k]+f[j,i,k])*(f[l,n,m]-f[n,l,m]) 
        #     + 2*(f[k,m,i]*f[j,l,n]+f[k,m,j]*f[i,l,n]) - 8*f[k,l,i]*f[m,n,j]
        #     + 20*(f[k,l,i]*f[j,m,n]+f[k,l,j]*f[i,m,n]) - 13*f[i,k,l]*f[j,m,n])
        #     - ∂2lnα[i,j] - ∂lnα[i]*∂lnα[j]
        #     + 2*γ[i,j]*γup[k,l]*γup[m,n]*(f[k,m,n]*∂lnα[l] - f[k,m,l]*∂lnα[n]) 
        #     + γup[k,l]*((f[i,j,k]+f[j,i,k]-f[k,i,j])*∂lnα[l] + 2*f[k,l,i]*∂lnα[j] 
        #     + 2*f[k,l,j]*∂lnα[i] - 3*(f[i,k,l]*∂lnα[j] + f[j,k,l]*∂lnα[i]))
        #     - (8*pi)*S[i,j] + (4*pi)*(γ[i,j]*T) ) )

        # 1.179s 
   
        # for k in 1:2

        #     @einsum ∂ₜf[k,i,j] = ( β[l]*∂f[l,k,i,j] - α*∂K[k,i,j] #Principle Part
        #         + α*( γup[m,n]*( 2*K[k,i]*f[j,m,n] + 2*K[k,j]*f[i,m,n]
        #         - 2*f[m,n,i]*K[j,k] - 2*f[m,n,j]*K[i,k] + K[i,j]*(2*f[m,n,k]-3*f[k,m,n]))
        #         + γup[m,n]*γup[p,q]*(K[m,p]*(γ[k,i]*f[j,q,n]+γ[k,j]*f[i,q,n]-2*f[q,n,i]*γ[j,k]-2*f[q,n,j]*γ[i,j])
        #         + (γ[k,i]*K[j,m] + γ[k,j]*K[i,m])*(8*f[n,p,q]-6*f[p,q,n]) 
        #         + K[m,n]*(2*(f[p,q,i]*γ[j,k]+f[p,q,j]*γ[i,k]) - 5*(γ[k,i]*f[j,p,q]+γ[k,j]*f[i,p,q])/2))
        #         - K[i,j]*∂lnα[k] + γup[m,n]*((K[m,i]*γ[j,k]+K[m,j]*γ[i,k])*∂lnα[n]
        #         - K[m,n]*(γ[k,i]*∂lnα[j]+γ[k,j]*∂lnα[i])) + 8*pi*(γ[k,i]*Sv[j]+γ[k,j]*Sv[i]) ) )
        
        # end

        # 8.28s   
   
   
   # @inline ∂γ(k,i,j) = if k==1 
    #                         if i==1 Dr2(γrr,x,y,nr) elseif i==2 Dr2(γθθ,x,y,nr) elseif i==2 Dr2(γϕϕ,x,y,nr) end
    #                       elseif k==2 
    #                         if i==1 Dθ2(γrr,x,y,nθ) elseif i==2 Dθ2(γθθ,x,y,nθ) elseif i==2 Dθ2(γϕϕ,x,y,nθ) end
    #                       else @assert false
    #                     end
    # @inline ∂K(k,i,j) = if k==1 
    #                         if i==1 Dr2(Krr,x,y,nr) elseif i==2 Dr2(Kθθ,x,y,nr) elseif i==2 Dr2(Kϕϕ,x,y,nr) end
    #                       elseif k==2 
    #                         if i==1 Dθ2(Krr,x,y,nθ) elseif i==2 Dθ2(Kθθ,x,y,nθ) elseif i==2 Dθ2(Kϕϕ,x,y,nθ) end
    #                       else @assert false
    #                     end
    # @inline ∂f(l,k,i,j) = if l==1 
    #                         if k==1
    #                          if i==1 Dr2(frrr,x,y,nr) elseif i==2 Dr2(frθθ,x,y,nr) elseif i==2 Dr2(frϕϕ,x,y,nr) end
    #                         elseif k==2
    #                          if i==1 Dr2(fθrr,x,y,nr) elseif i==2 Dr2(fθθθ,x,y,nr) elseif i==2 Dr2(fθϕϕ,x,y,nr) end
    #                         end
    #                       elseif l==2 
    #                         if k==1
    #                             if i==1 Dθ2(frrr,x,y,nθ) elseif i==2 Dθ2(frθθ,x,y,nθ) elseif i==2 Dθ2(frϕϕ,x,y,nθ) end
    #                         elseif k==2
    #                             if i==1 Dθ2(fθrr,x,y,nθ) elseif i==2 Dθ2(fθθθ,x,y,nθ) elseif i==2 Dθ2(fθϕϕ,x,y,nθ) end
    #                         end
    #                       else @assert false end


    # φ,ψr,ψθ,Π = U.x
    # ∂ₜφ,∂ₜψr,∂ₜψθ,∂ₜΠ = ∂ₜU.x
    # γrr,γθθ,rootγ,α,βr = metric.x

    # @inline γ2()  = rootγ[x,y]
    # @inline f∂ₜφ(x,y) = @part (x,y) βr*ψr - α*Π
    # @inline fVr(x,y) = @part (x,y) βr*Π - α*ψr/γrr 
    # @inline fVθ(x,y) = @part (x,y) -α*ψθ/γθθ 


    # # Define the metric, extrinsic curvature, and functions f_{kij}, sometimes these are zero.

    # @inline fγ(i,j) = i==j ? (if i==1 γrr[x,y] elseif i==2 γθθ[x,y] elseif i==3 γϕϕ[x,y] end) : 0 
    # @inline fK(i,j) = i==j ? (if i==1 Krr[x,y] elseif i==2 Kθθ[x,y] elseif i==3 Kϕϕ[x,y] end) : 0 
    # @inline function ff(k,i,j) 
    #     if     k==1 
    #         if i==j 
    #             if i==1 frrr[x,y] elseif i==2 frθθ[x,y] elseif i==3 frϕϕ[x,y] end
    #         elseif (i==1 && j==2) || (i==2 && j==1)
    #             @part (x,y) fθrr + (γrr/γθθ)*fθϕϕ
    #         else 0. end
    #     elseif k==2
    #         if i==j 
    #             if i==1 fθrr[x,y] elseif i==2 fθθθ[x,y] elseif i==3 fθϕϕ[x,y] end
    #         elseif (i==1 && j==2) || (i==2 && j==1)
    #             @part (x,y) frθθ + (γrr/γϕϕ)*frϕϕ
    #         else 0. end
    #     elseif k==3
    #         if     (i==1 && j==3) || (i==3 && j==1)
    #             @part (x,y) frϕϕ + (γϕϕ/γθθ)*frθθ
    #         elseif (i==2 && j==3) || (i==3 && j==2)
    #             @part (x,y) fθϕϕ + (γϕϕ/γθθ)*fθrr
    #         else 0. end
    #     else 0. end
    # end

    # # Define gauge variables and other misc

    # @inline fγup(i,j) = i==j ? (if i==1 1/γrr[x,y] elseif i==2 1/γθθ[x,y] elseif i==3 1/γϕϕ[x,y] end) : 0 
    # @inline fdlnα(l) = 0.
    # @inline fαt() = 
    # @inline fα() = αt()*γ2()
    # @inline fβ(i) = (if i==1 βr[x,y] elseif i==2  0. #= βθ[x,y] =# elseif i==3 0. #= βϕ[x,y] =# end)

    # # Define space derivatives

    # @inline f∂γ(k,i,j)   = k==1 ? Dr2(γ(i,j),x,y,nr)   : k==2 ? Dθ2(γ(i,j),x,y,nθ)   : 0.
    # @inline f∂K(k,i,j)   = k==1 ? Dr2(K(i,j),x,y,nr)   : k==2 ? Dθ2(K(i,j),x,y,nθ)   : 0.
    # @inline f∂f(l,k,i,j) = l==1 ? Dr2(f(k,i,j),x,y,nr) : l==2 ? Dθ2(f(k,i,j),x,y,nθ) : 0. 

    # # Define time derivatives (can't set these to zero, they are filled with values)
    # # This assumes the metric, extrinsic curvature, and functions f_{kij} are diagonal in (i,j)

    # @inline f∂ₜγ(i,j) = i==j ? (if i==1  ∂ₜγrr[x,y] elseif i==2  ∂ₜγθθ[x,y] elseif i==3  ∂ₜγϕϕ[x,y] end) : @assert false
    # @inline f∂ₜK(i,j) = i==j ? (if i==1  ∂ₜKrr[x,y] elseif i==2  ∂ₜKθθ[x,y] elseif i==3  ∂ₜKϕϕ[x,y] end) : @assert false
    # @inline function f∂ₜf(k,i,j) 
    #     if     k==1 
    #         if i==1 ∂ₜfrrr[x,y] elseif i==2 ∂ₜfrθθ[x,y] elseif i==3 ∂ₜfrϕϕ[x,y] end
    #     elseif k==2
    #         if i==1 fθrr[x,y] elseif i==2 fθθθ[x,y] elseif i==3 fθϕϕ[x,y] end
    #     end
    # end

    # # Sources

    # #@inline ρ() = 0.
    # @inline fT() = 0.
    # @inline fS(i) = 0.
    # @inline fS(i,j) = 0.





    #∂ᵣUmθ = @part n ( ∂ᵣKθθ - ∂ᵣfrθθ/sqrt(γrr) + ∂ᵣγrr*frθθ/sqrt(γrr)^3/2 )

    # Umrb = @part n (-Upr - Umθ*γrr/γθθ - 2*∂ᵣUmθ*sqrt(γrr)/Umθ - γrr/Umθ
    #      + 8*pi*γrr*γθθ*(ρ - Sr/sqrt(γrr))/Umθ )

    #Transmitting Conditions?
    #Umrb = Umrin     
     
     # frrr = frrri
        #Uprb = -(Umr - Umri) + Upri

        # ∂ᵣUmθ = @part 1 ∂ᵣKθθ - ∂ᵣfrθθ/sqrt(γrr) + frθθ*(2*frrr - 8*frθθ*γrr/γθθ)/(2*sqrt(γrr)^3)
        #Uprb = @part 1 (-Umr - γrr*Umθ/γθθ - (2*∂ᵣUmθ*sqrt(γrr) + γrr)/Umθ )

        #∂ᵣUpθ = @part 1 ( ∂ᵣKθθ + ∂ᵣfrθθ/sqrt(γrr) - ∂ᵣγrr*frθθ/sqrt(γrr)^3/2 )
        #∂ᵣUmθ = @part n ( ∂ᵣKθθ - ∂ᵣfrθθ/sqrt(γrr) + ∂ᵣγrr*frθθ/sqrt(γrr)^3/2 )

        # Uprb = @part 1 (-Umr - Upθ*γrr/γθθ + 2*∂ᵣUpθ*sqrt(γrr)/Upθ - γrr/Upθ
        #      + 8*pi*γrr*γθθ*(ρ + Sr/sqrt(γrr))/Upθ )

        # Uprb = @part 1 (-Umr - Umθ*γrr/γθθ - 2*∂ᵣUmθ*sqrt(γrr)/Umθ - γrr/Umθ
        #   + 8*pi*γrr*γθθ*(ρ - Sr/sqrt(γrr))/Umθ )

        #Dirichlet on r-mode
        #Uprb = @part 1 (cm/cp)*(Umr-(Krri - frrri/sqrt(γrri))) + Krri + frrri/sqrt(γrri)

    
    # Copy the state into the parameters so that it can be changed

    #######################
    # Attention!
    #
    # Do not do the following:
    # state .= regstate
    #
    # This results in an intense slowdown
    # Do instead:
    # for i in 1:numvar
    #     state.x[i] .= regstate.x[i]
    # end


    # Calculated lapse and derivatives of densitized lapse

    # @. α = ᾶ*γθθ*sqrt(γrr)
    # @. ∂ᵣlnᾶ = ∂ᵣᾶ/ᾶ
    # @. ∂ᵣ2lnᾶ = (∂ᵣ2ᾶ*ᾶ - ∂ᵣᾶ^2)/ᾶ^2


# function deriv!(df,f)
    
#     df[1:6] .= ql*f[1:9]

#     @turbo for i in 7:n-6
#         df[i] = (-f[i-3] + 9*f[i-2] - 45*f[i-1] + 45*f[i+1] - 9*f[i+2] + f[i+3])/60/dr
#     end

#     df[n:n-5] .= qr*f[n:n-8]

# end

# # Sample the 'regular' values and derivatives,
    # # which are used in the regularization process
    # sample!(γrri,   grid, r -> fγrr(M0,r)               )
    # sample!(γθθi,   grid, r -> fγθθ(M0,r)               )
    # sample!(Krri,   grid, r -> fKrr(M0,f∂ₜγrri,r)        )
    # sample!(Kθθi,   grid, r -> fKθθ(M0,f∂ₜγθθi,r)        )
    # sample!(frrri,  grid, r -> ffrrr(M0,r)              )
    # sample!(frθθi,  grid, r -> ffrθθ(M0,r)              )
    # sample!(𝜙i,     grid, r -> f𝜙(M0,r)                 )
    # sample!(ψri,    grid, r -> fψr(M0,r)                )
    # sample!(Πi,     grid, r -> fΠ(M0,r)                 )

# function continuous_print(integrator)

#     ###############################################
#     # Outputs status numbers while the program runs
#     ###############################################

#     dtstate = integrator.p.dtstate

#     ∂ₜγrr,∂ₜγθθ,∂ₜKrr,∂ₜKθθ,∂ₜfrrr,∂ₜfrθθ,∂ₜ𝜙,∂ₜψr,∂ₜΠ = dtstate.x

#     println("| ",
#     rpad(string(round(integrator.t,digits=1)),5," "),"|   ",
#     rpad(string(round(maximum(abs.(∂ₜγrr)), digits=3)),8," "),"|   ",
#     rpad(string(round(maximum(abs.(∂ₜγθθ)), digits=3)),8," "),"|   ",
#     rpad(string(round(maximum(abs.(∂ₜKrr)), digits=3)),8," "),"|   ",
#     rpad(string(round(maximum(abs.(∂ₜKθθ)), digits=3)),8," "),"|   ",
#     rpad(string(round(maximum(abs.(∂ₜfrrr)),digits=3)),9," "),"|   ",
#     rpad(string(round(maximum(abs.(∂ₜfrθθ)),digits=3)),9," "),"|"
#     )

#     return

# end

# γrrrhs = ∂ₜγrr[n]; γθθrhs = ∂ₜγθθ[n];
# Krrrhs = ∂ₜKrr[n]; frrrrhs = ∂ₜfrrr[n];
# Kθθrhs = ∂ₜKθθ[n]; frθθrhs = ∂ₜfrθθ[n];
#
# ∂ₜU0r = ∂ₜγrr[n]
# ∂ₜUmr = 0.
#
# @part n ∂ₜKrr  = ∂ₜUmr/2 + Krrrhs/2 + frrrrhs/sqrt(γrr)/2 - frrr*γrrrhs/4/sqrt(γrr)^3
# @part n ∂ₜfrrr = (frrrrhs/2 - ∂ₜUmr*sqrt(γrr)/2 + Krrrhs*sqrt(γrr)/2
#  - frrr*γrrrhs/4/γrr + frrr*∂ₜU0r/2/γrr)
#
# ∂ᵣUmθ = @part n ( -(Umr + Upr)*Umθ/2/sqrt(γrr) - (1. + Umθ^2/γθθ)*sqrt(γrr)/2
#     + 4*pi*sqrt(γrr)*γθθ*(ρ - Sr/sqrt(γrr)) )
#
# ∂ₜUmθ = @part n ( α - (-βʳ - α/sqrt(γrr))*∂ᵣUmθ + Upr*Umθ*α/γrr
#     - (Upθ - Umθ)*Umθ*α/γθθ + α*∂ᵣlnᾶ*Umθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )
#
# @part n ∂ₜKθθ  = ∂ₜUmθ/2 + Kθθrhs/2 + frθθrhs/sqrt(γrr)/2 - frθθ*γrrrhs/4/sqrt(γrr)^3
# @part n ∂ₜfrθθ = (frθθrhs/2 - ∂ₜUmθ*sqrt(γrr)/2 + Kθθrhs*sqrt(γrr)/2
#  - frθθ*γrrrhs/γrr/4 + frθθ*∂ₜU0r/γrr/2)

# @. ∂ₜψr =   βʳ*∂ᵣψr - α*∂ᵣΠ - α*(frrr/γrr - 2*frθθ/γθθ + ∂ᵣlnᾶ)*Π + ψr*∂ᵣβʳ
#
# @. ∂ₜΠ = ( βʳ*∂ᵣΠ - α*∂ᵣψr/γrr + α*(Krr/γrr + 2*Kθθ/γθθ)*Π
#  - α*(4*frθθ/γθθ + ∂ᵣlnᾶ)*ψr/γrr + m^2*α*𝜙 )

# @. ∂ₜγrr = βʳ*∂ᵣγrr + 2*∂ᵣβʳ*γrr - 2*α*Krr
#
# @. ∂ₜγθθ = βʳ*∂ᵣγθθ - 2*α*Kθθ
#
# @. ∂ₜKrr  = ( βʳ*∂ᵣKrr - α*∂ᵣfrrr/γrr + 2*α*frrr^2/γrr^2 - 6*α*frθθ^2/γθθ^2
#  - α*Krr^2/γrr + 2*α*Krr*Kθθ/γθθ - 8*α*frrr*frθθ/(γrr*γθθ)
#  - α*frrr*∂ᵣlnᾶ/γrr - α*∂ᵣlnᾶ^2 - α*∂ᵣ2lnᾶ + 2*∂ᵣβʳ*Krr)
#
# @. ∂ₜKθθ  = ( βʳ*∂ᵣKθθ - α*∂ᵣfrθθ/γrr + α + α*Krr*Kθθ/γrr
#  - 2*α*frθθ^2/(γrr*γθθ) - α*frθθ*∂ᵣlnᾶ/γrr)
#
# @. ∂ₜfrrr = ( βʳ*∂ᵣfrrr - α*∂ᵣKrr - α*frrr*Krr/γrr
#  + 12*α*frθθ*Kθθ*γrr/γθθ^2 - 10*α*frθθ*Krr/γθθ - 4*α*frrr*Kθθ/γθθ
#  - α*Krr*∂ᵣlnᾶ - 4*α*Kθθ*γrr*∂ᵣlnᾶ/γθθ + 3*∂ᵣβʳ*frrr + γrr*∂ᵣ2βʳ )
#
# @. ∂ₜfrθθ = ( βʳ*∂ᵣfrθθ - α*∂ᵣKθθ - α*frrr*Kθθ/γrr + 2*α*frθθ*Kθθ/γθθ
#  - α*Kθθ*∂ᵣlnᾶ + ∂ᵣβʳ*frθθ )

# Neumann
# ∂ₜUp𝜙 = @part 1 -∂ₜUm𝜙*cm/cp
#∂ₜUp𝜙 = 0.

#∂ₜUp𝜙 = 0.

# ∂ₜψ[1]  += s*sqrt(γrr[1])*(Up𝜙b - Up𝜙)/(dr̃*σ00)/2.
# ∂ₜΠ[1]  += s*(Up𝜙b - Up𝜙)/(dr̃*σ00)/2.
#∂ₜ𝜙[1]  += s*(0. - 𝜙[1])/(dr̃*σ00)

#∂ₜ𝜙[1] = 0.
# ∂ₜψ[1] += Πrhs/cp
# ∂ₜΠ[1] = 0.

# γrrrhs = ∂ₜγrr[1]; γθθrhs = ∂ₜγθθ[1];
# Krrrhs = ∂ₜKrr[1]; frrrrhs = ∂ₜfrrr[1];
# Kθθrhs = ∂ₜKθθ[1]; frθθrhs = ∂ₜfrθθ[1];
# Πrhs = ∂ₜΠ[1]; ψrhs = ∂ₜψ[1];

# @part 1 ∂ₜΠ = ∂ₜUp𝜙/2 + Πrhs/2 - ψrhs/sqrt(γrr)/2 + ψ*γrrrhs/4/sqrt(γrr)^3
# @part 1 ∂ₜψ = ψrhs/2 + ∂ₜUp𝜙*sqrt(γrr)/2 - Πrhs*sqrt(γrr)/2 + ψ*γrrrhs/4/γrr
#Define boundary condition
# Dirichlet condition keeps areal radius constant.
# Upθ = @part 1 Umθ*cm/cp
#
# ∂ₜUmr = @part 1 ∂ₜKrr - ∂ₜfrrr/sqrt(γrr) + frrr*∂ₜγrr/2/sqrt(γrr)^3

#∂ₜUpr = @part 1 ∂ₜUmr*cm/cp + Umr*(2*ᾶ*βʳ*∂ₜγθθ/cp^2)

# ∂ₜUpr = 0
#
# #Dirichlet on r-mode
# #Uprb = @part 1 (cm/cp)*Umr
#
# @part 1 ∂ₜKrr = ∂ₜUpr/2 + Krrrhs/2 - frrrrhs/sqrt(γrr)/2 + frrr*γrrrhs/4/sqrt(γrr)^3
# @part 1 ∂ₜfrrr = frrrrhs/2 + ∂ₜUpr*sqrt(γrr)/2 - Krrrhs*sqrt(γrr)/2 + frrr*γrrrhs/4/γrr
#
# # ∂ₜKrr[1]  += s*(Uprb - Upr[1])/(dr̃*Σ11)/2.
# # ∂ₜfrrr[1] += s*sqrt(γrr[1])*(Uprb - Upr[1])/(dr̃*Σ11)/2.
#
# ∂ᵣUpθ = @part 1 ( (Umr + Upr)*Upθ/2/sqrt(γrr) + (1. + Upθ^2/γθθ)*sqrt(γrr)/2
#     - 4*pi*sqrt(γrr)*γθθ*(ρ + Sr/sqrt(γrr)) )
#
# ∂ₜUpθ = @part 1 ( α - cp*∂ᵣUpθ + Umr*Upθ*α/γrr + (Upθ - Umθ)*Upθ*α/γθθ
#     - α*∂ᵣlnᾶ*Upθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )
#
# @part 1 ∂ₜKθθ  = ∂ₜUpθ/2 + Kθθrhs/2 - frθθrhs/sqrt(γrr)/2 + frθθ*γrrrhs/4/sqrt(γrr)^3
# @part 1 ∂ₜfrθθ = frθθrhs/2 + ∂ₜUpθ*sqrt(γrr)/2 - Kθθrhs*sqrt(γrr)/2 + frθθ*γrrrhs/4/γrr

# Gauge Conditions
# Keep radius areal and keep cp constant

# @. ᾶ  = cp*frθθ/γθθ/(frθθ-Kθθ*sqrt(γrr))
#
# @. βʳ = cp*Kθθ*sqrt(γrr)/(frθθ-Kθθ*sqrt(γrr))
#
# mul!(∂ᵣᾶ,D,ᾶ)
# mul!(∂ᵣ2ᾶ,D,∂ᵣᾶ)
#
# mul!(∂ᵣβʳ,D,βʳ)
# mul!(∂ᵣ2βʳ,D,∂ᵣβʳ)

# Keep both ingoing and outgoing coordinate speeds of light fixed
# @. ᾶ    = (cp-cm)/γθθ/2
# @. ∂ᵣᾶ  = (∂ᵣcp-∂ᵣcm)/γθθ/2 - 4*frθθ*ᾶ/γθθ/2
# @. ∂ᵣ2ᾶ = ( (∂ᵣ2cp-∂ᵣ2cm)/γθθ/2 + 7*ᾶ*frθθ^2/γθθ^2 - ᾶ*Kθθ^2*γrr/γθθ^2
#  - 4*∂ᵣᾶ*frθθ/γθθ - 2*ᾶ*Krr*Kθθ/γθθ - 2*ᾶ*frrr*frθθ/γrr/γθθ - ᾶ*γrr/γθθ
#  + 8*pi*ᾶ*γrr*ρ )
#
# @. βʳ    = -(cp+cm)/2
# @. ∂ᵣβʳ  = -(∂ᵣcp+∂ᵣcm)/2
# @. ∂ᵣ2βʳ = -(∂ᵣ2cp+∂ᵣ2cm)/2

# ρ = temp.x[5]
# @. ρ = ( Π^2 + ψ^2/γrr + (m^2)*𝜙^2)/2
#
# @. βʳ    = -cp + ᾶ*γθθ
# @. ∂ᵣβʳ  = -∂ᵣcp + ∂ᵣᾶ*γθθ + 2*ᾶ*frθθ
# @. ∂ᵣ2βʳ = ( -∂ᵣ2cp + ∂ᵣ2ᾶ*γθθ + 4*∂ᵣᾶ*frθθ + 2*ᾶ*Krr*Kθθ + 2*ᾶ*frrr*frθθ/γrr
#     + ᾶ*γrr - 7*ᾶ*frθθ^2/γθθ + ᾶ*Kθθ^2*γrr/γθθ - 8*pi*ᾶ*γrr*γθθ*ρ )

# Gauge condition for preventing apparent horizon formation

# @. βʳ = ᾶ*γθθ*γrr*( 3*frθθ^2 - 2*frθθ*Kθθ*sqrt(γrr) - γrr*Kθθ^2 - 2*γθθ*∂ᵣlnᾶ*frθθ
#  - 2*frrr*frθθ*γθθ/γrr + 2*frrr*Kθθ*γθθ/sqrt(γrr) + 2*∂ᵣlnᾶ*Kθθ*γθθ*sqrt(γrr)
#  + γrr*γθθ - 4*pi*γθθ^2*sqrt(γrr)*Sr + 8*pi*γθθ^2*γrr*ρ )/(
#     frθθ^2*γrr - 2*frθθ*Kθθ*sqrt(γrr)^3 + Kθθ^2*γrr^2 - 2*frθθ*Krr*γθθ*sqrt(γrr)
#  + 2*Krr*Kθθ*γrr*γθθ + γrr^2*γθθ + 4*pi*γθθ^2*sqrt(γrr)^3*Sr - 8*pi*γθθ^2*γrr^2*ρ )
#
# mul!(∂ᵣβʳ,D,βʳ)
# mul!(∂ᵣ2βʳ,D,∂ᵣβʳ)

#Gauge condition for preventing apparent horizon formation

# @. βʳ = ( 2*rh*vh + ᾶ*γθθ*sqrt(γrr)*Kθθ )/frθθ
#
# mul!(∂ᵣβʳ,D,βʳ)
# mul!(∂ᵣ2βʳ,D,∂ᵣβʳ)


#Inject boundary into evolution equation for Upr

# ∂tUpr = @part 1 ( -cp*∂rUpr - α*Umr*Upr/γrr - 9*α*Umθ^2*γrr/γθθ^2/2.
#     + 3*α*Umθ*Upθ*γrr/γθθ^2 + 3*α*Upθ^2*γrr/γθθ^2/2. + 3*α*Umr*Umθ/γθθ
#     + 3*α*Upr*Umθ/γθθ - 4*α*Upr*Upθ/γθθ - 2*∂rlnᾶ*(Upθ+Umθ)*α*sqrt(γrr)/γθθ
#     - α*∂rlnᾶ*Upr/sqrt(γrr) + 2*∂rβr*Upr + ∂r2βr*sqrt(γrr) - α*∂rlnᾶ^2
#     - α*∂r2lnᾶ + 16*pi*α*sqrt(γrr)*Sr - 8*pi*α*Srr + 4*pi*α*γrr*Tt )

# Umθ = @part n ( Kθθ - frθθ/sqrt(γrr) )
# Upθ = @part n ( Kθθ + frθθ/sqrt(γrr) )
# Umr = @part n ( Krr - frrr/sqrt(γrr) )
# Upr = @part n ( Krr + frrr/sqrt(γrr) )
#
# γrrrhs = ∂tγrr[n]; γθθrhs  = ∂tγθθ[n];
# Krrrhs = ∂tKrr[n]; frrrrhs = ∂tfrrr[n];
# Kθθrhs = ∂tKθθ[n]; frθθrhs = ∂tfrθθ[n];
#
# ∂tUmrb = @part n ( ∂tKrr - ∂tfrrr/sqrt(γrr) + frrr*∂tγrr/sqrt(γrr)^3/2 )
#
# # ∂tγrr[n] = s*(γrri[n] - γrr[n])/(dr̃*σ00)
# # ∂tγθθ[n] = s*(γθθi[n] - γθθ[n])/(dr̃*σ00)
# # ∂tγrr[n] = 0.
# # ∂tγθθ[n] = 0.
#
# #println(γrri[n] - γrr[n])
# #println(γθθi[n] - γθθ[n])
#
# # Mode speeds
# cm = @part n ( -βr - α/sqrt(γrr) )
# cp = @part n ( -βr + α/sqrt(γrr) )
#
# Umθb = Upθ*cp/cm
# #Umθb = Kθθi[n] - frθθi[n]/sqrt(γrri[n])
# #Umθb = Umθ
#
# ∂tKθθ[n]  += s*(Umθb - Umθ)/(dr̃*σ00)/2
# ∂tfrθθ[n] += s*sqrt(γrr[n])*(Umθb - Umθ)/(dr̃*σ00)/2# + frθθ[n]*(∂tγrr[n]-γrrrhs)/γrr[n]/2
#
# ∂tUmθ = @part n ( ∂tKθθ - ∂tfrθθ/sqrt(γrr) + frθθ*∂tγrr/sqrt(γrr)^3/2 )
#
# Umθ = Umθb
#
# # Define derivative of incoming characteristic based on evolution equations
# ######### Problem starts here
# ∂rUmθ = @part n (-∂tUmθ + α + Upr*Umθ*α/γrr - (Upθ - Umθ)*Umθ*α/γθθ
#  + α*∂rlnᾶ*Umθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )/cm
#
# # Calculate radial incoming characteristic based on constraints
# Umrb = @part n (-Upr - Umθ*γrr/γθθ - 2*∂rUmθ*sqrt(γrr)/Umθ - γrr/Umθ
#     + 8*pi*γrr*γθθ*(ρ - Sr/sqrt(γrr))/Umθ )
#
# ∂tKrr[n]  += s*(Umrb - Umr)/(dr̃*σ00)/2
# ∂tfrrr[n] += s*sqrt(γrr[n])*(Umrb - Umr)/(dr̃*σ00)/2# + frrr[n]*(∂tγrr[n]-γrrrhs)/γrr[n]/2
#
# Umr = Umrb
#
# #∂tUmr = @part n ( ∂tKrr - ∂tfrrr/sqrt(γrr) + frrr*∂tγrr/sqrt(γrr)^3/2 )
#
# @part n ∂tγrr = -(Umr+Upr)*α - (Umr-Upr)*βr*sqrt(γrr) + 2*∂rβr*γrr + 4*(Umθ-Upθ)*βr*sqrt(γrr)^3/γθθ
# @part n ∂tγθθ = -(Umθ+Upθ)*α - (Umθ-Upθ)*βr*sqrt(γrr)
#
# @part n ∂tγrr = 0.
# @part n ∂tγθθ = 0.

#println(∂tUmrb-∂tUmr)

# println(Umθb - Umθ)



# cp = -βr[n] + α[n]/sqrt(γrr[n])
# cm = -βr[n] - α[n]/sqrt(γrr[n])

# ∂tψ[n] += s*(  Π[n]/cm  )/(dr̃*σ00)
# ∂tΠ[n] += s*( -Π[n] )/(dr̃*σ00)

# ∂tψ[n] += s*(     (Π[n]+cp*ψ[n])/(cm-cp) )/(dr̃*σ00)
# ∂tΠ[n] += s*( -cm*(Π[n]+cp*ψ[n])/(cm-cp) )/(dr̃*σ00)

# cp = -βr[1] + α[1]/sqrt(γrr[1])
# ψrhs = ∂tψ[1]; Πrhs = ∂tΠ[1];
#
# #∂t𝜙[1] = 0.
# ∂tψ[1] += Πrhs/cp
# ∂tΠ[1] = 0.

# ∂tψ[n] += Πrhs/cm
# ∂tΠ[n] = 0.

# Γt = temp.x[5]; Γr = temp.x[6];
#
# @. Γt = (βr*∂rlnᾶ - ∂rβr)/α^2
# @. Γr = 2*βr*∂rβr/α^2 - (1/γrr + (βr/α)^2)*∂rlnᾶ - 4*frθθ/(γrr*γθθ)
#
# @. ∂t𝜙 = Π
# @. ∂tψ = ∂rΠ
# @. ∂tΠ = (α^2)*((1/γrr-(βr/α)^2)*∂rψ + 2*(βr/α^2)*∂rΠ - Γr*ψ - Γt*Π - m^2*𝜙)

# Specify the inner and outer temporal boundary conditions
# for metric variables

# Umθ = @part 1 ( Kθθ - frθθ/sqrt(γrr) )
# Upθ = @part 1 ( Kθθ + frθθ/sqrt(γrr) )
#
# Umr = @part 1 ( Krr - frrr/sqrt(γrr) )
# Upr = @part 1 ( Krr + frrr/sqrt(γrr) )
#
# γrrrhs = ∂tγrr[1]; γθθrhs = ∂tγθθ[1];
# Krrrhs = ∂tKrr[1]; frrrrhs = ∂tfrrr[1];
# Kθθrhs = ∂tKθθ[1]; frθθrhs = ∂tfrθθ[1];

# dtU0r = (2*frrr[1] - 8*frθθ[1]*γrr[1]/γθθ[1])*βr[1] + 2*∂rβr[1]*γrr[1] - 2*α[1]*Krr[1]
# dtU0θ = 2*frθθ[1]*βr[1] - 2*α[1]*Kθθ[1]

# ∂tγrr[1] = dtU0r
# ∂tγθθ[1] = dtU0θ

#∂tUpr = ∂tKrr[1] + ∂tfrrr[1]/sqrt(γrr[1]) - frrr[1]*∂tγrr[1]/2/sqrt(γrr[1])^3

#∂tUpr = 0. + 4*pi*α[1]*(γrr[1]*Tt[1] - 2*Srr[1]) + 16*pi*α[1]*sqrt(γrr[1])*Sr[1]

# ∂tUpr = 0.
#
# @part 1 ∂tKrr = ∂tUpr/2 + Krrrhs/2 - frrrrhs/sqrt(γrr)/2 + frrr*γrrrhs/4/sqrt(γrr)^3
# @part 1 ∂tfrrr = frrrrhs/2 + ∂tUpr*sqrt(γrr)/2 - Krrrhs*sqrt(γrr)/2 + frrr*γrrrhs/4/γrr
#
# ∂rUpθ = @part 1 ( (Umr + Upr)*Upθ/2/sqrt(γrr) + (1. + Upθ^2/γθθ)*sqrt(γrr)/2
#     - 4*pi*sqrt(γrr)*γθθ*(ρ + Sr/sqrt(γrr)) )
#
# ∂tUpθ = @part 1 (α - (-βr + α/sqrt(γrr))*∂rUpθ + Umr*Upθ*α/γrr
#     + (Upθ - Umθ)*Upθ*α/γθθ - α*∂rlnᾶ*Upθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )
#
# @part 1 ∂tKθθ  = ∂tUpθ/2 + Kθθrhs/2 - frθθrhs/sqrt(γrr)/2 + frθθ*γrrrhs/4/sqrt(γrr)^3
# @part 1 ∂tfrθθ = frθθrhs/2 + ∂tUpθ*sqrt(γrr)/2 - Kθθrhs*sqrt(γrr)/2 + frθθ*γrrrhs/4/γrr

# Outer boundary

# Umθ = @part n ( Kθθ - frθθ/sqrt(γrr) )
# Upθ = @part n ( Kθθ + frθθ/sqrt(γrr) )
#
# Umr = @part n ( Krr - frrr/sqrt(γrr) )
# Upr = @part n ( Krr + frrr/sqrt(γrr) )
#
# γrrrhs = ∂tγrr[n]; γθθrhs = ∂tγθθ[n];
# Krrrhs = ∂tKrr[n]; frrrrhs = ∂tfrrr[n];
# Kθθrhs = ∂tKθθ[n]; frθθrhs = ∂tfrθθ[n];
#
# dtU0r = @part n ( (2*frrr - 8*frθθ*γrr/γθθ)*βr + 2*∂rβr*γrr - 2*α*Krr )
# dtU0θ = @part n ( 2*frθθ*βr - 2*α*Kθθ )
#
# ∂tγrr[n] = dtU0r
# ∂tγθθ[n] = dtU0θ
#
# #∂tUmr = ∂tKrr[n] - ∂tfrrr[n]/sqrt(γrr[n]) + frrr[n]*∂tγrr[n]/2/sqrt(γrr[n])^3
# #∂tUmr = 0. + 4*pi*α[n]*(γrr[n]*Tt[n] - 2*Srr[n]) - 16*pi*α[n]*sqrt(γrr[n])*Sr[n]
#
# ∂tUmr = 0.
#
# @part n ∂tKrr  = ∂tUmr/2 + Krrrhs/2 + frrrrhs/sqrt(γrr)/2 - frrr*γrrrhs/4/sqrt(γrr)^3
# @part n ∂tfrrr = (frrrrhs/2 - ∂tUmr*sqrt(γrr)/2 + Krrrhs*sqrt(γrr)/2
#  - frrr*γrrrhs/4/γrr + frrr*dtU0r/2/γrr)
#
# ∂rUmθ = @part n ( -(Umr + Upr)*Umθ/2/sqrt(γrr) - (1. + Umθ^2/γθθ)*sqrt(γrr)/2
#     + 4*pi*sqrt(γrr)*γθθ*(ρ - Sr/sqrt(γrr)) )
#
# ∂tUmθ = @part n ( α - (-βr - α/sqrt(γrr))*∂rUmθ + Upr*Umθ*α/γrr
#     - (Upθ - Umθ)*Umθ*α/γθθ + α*∂rlnᾶ*Umθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )
#
# @part n ∂tKθθ  = ∂tUmθ/2 + Kθθrhs/2 + frθθrhs/sqrt(γrr)/2 - frθθ*γrrrhs/4/sqrt(γrr)^3
# @part n ∂tfrθθ = (frθθrhs/2 - ∂tUmθ*sqrt(γrr)/2 + Kθθrhs*sqrt(γrr)/2
#  - frθθ*γrrrhs/γrr/4 + frθθ*dtU0r/γrr/2)



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

# @inline function deriv2!(df::Vector{T}, f::Vector{T}, n::Int64, dx::T) where T
#
#     #@inbounds
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

# Upθi = Kθθi[1] + frθθi[1]/sqrt(γrri[1])
# Upθb = Upθi

# Apply boundary condition to incoming characteristic using SAT
# ∂tKθθ[1]  += s*(Upθb - Upθ[1])/(dr̃*σ00)/2.
# ∂tfrθθ[1] += s*sqrt(γrr[1])*(Upθb - Upθ[1])/(dr̃*σ00)/2.

# ∂tUpθ = @part 1 ∂tKθθ + ∂tfrθθ/sqrt(γrr) - frθθ*∂tγrr/sqrt(γrr)^3/2
#
# ∂rUpθ = @part 1 ∂rKθθ + ∂rfrθθ/sqrt(γrr) - frθθ*∂rγrr/sqrt(γrr)^3/2
# ∂rUmθ = @part 1 ∂rKθθ - ∂rfrθθ/sqrt(γrr) + frθθ*∂rγrr/sqrt(γrr)^3/2

# ∂rUpθ = @part 1 ∂rKθθ + ∂rfrθθ/sqrt(γrr) - frθθ*(2*frrr - 8*frθθ*γrr/γθθ)/sqrt(γrr)^3/2
# ∂rUmθ = @part 1 ∂rKθθ - ∂rfrθθ/sqrt(γrr) + frθθ*(2*frrr - 8*frθθ*γrr/γθθ)/sqrt(γrr)^3/2

# @. Upθv /= Kθθi + frθθi/sqrt(γrri)
# ∂rUpθ = (-25*Upθv[1] + 48*Upθv[2] - 36*Upθv[3] + 16*Upθv[4] - 3*Upθv[5])/(12*dr̃)/drdr̃[1]
# ∂rUpθ = @part 1 ∂rUpθ*(Kθθi + frθθi/sqrt(γrri)) + Upθv*(∂rKθθi + ∂rfrθθi/sqrt(γrri) - frθθi*∂rγrri/sqrt(γrri)^3/2)

#Calculate radial incoming characteristic based on constraints
# Uprb = @part 1 (-Umr - Upθ*γrr/γθθ + 2*∂rUpθ*sqrt(γrr)/Upθ - γrr/Upθ
#     + 8*pi*γrr*γθθ*(ρ + Sr/sqrt(γrr))/Upθ )

#Define derivative of incoming characteristic based on evolution equations
# ∂rUpθ = @part 1 (-∂tUpθ + α + Umr*Upθ*α/γrr + (Upθ - Umθ)*Upθ*α/γθθ
# - α*∂rlnᾶ*Upθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )/cp
#
# ∂rUmθ = @part 1 (-∂tUmθ + α + Upr*Umθ*α/γrr - (Upθ - Umθ)*Umθ*α/γθθ
#  + α*∂rlnᾶ*Umθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )/cm

# a = 1 pure constraint Dirichlet
# a =-1 pure constraint Neumann
# a = 0 pure constraint freezing

# a = 0.
#
# Uprb = @part 1 ( -Umr + (-2*(cp*∂rUpθ+a*cm*∂rUmθ)*sqrt(γrr) - (a*cm*Umθ^2-cp*Upθ^2)*γrr/γθθ
#   - (a*cm-cp)*γrr - 8*pi*(a*cm+cp)*sqrt(γrr)*γθθ*Sr + 8*pi*(a*cm-cp)*γrr*γθθ*ρ )/(a*cm*Umθ-cp*Upθ) )

# a = cm/cp

# Uprb = @part 1 ( -Umr + (2*(a*∂rUpθ-∂rUmθ)*sqrt(γrr) - (a*Upθ^2+Umθ^2)*γrr/γθθ
# - (1+a)*γrr + 8*pi*(a-1)*sqrt(γrr)*γθθ*Sr + 8*pi*(1+a)*γrr*γθθ*ρ )/(a*Upθ+Umθ) )

# ∂rcm = @part 1 ( -∂rβr - ∂rα/sqrt(γrr) + α*∂rγrr/sqrt(γrr)^3/2  )
# ∂rcp = @part 1 ( -∂rβr + ∂rα/sqrt(γrr) - α*∂rγrr/sqrt(γrr)^3/2  )

# ∂rcm = @part 1 ( -∂rβr - ∂rᾶ*γθθ - ᾶ*sqrt(γrr)*Umθ*(cm/cp-1)  )
# ∂rcp = @part 1 ( -∂rβr + ∂rᾶ*γθθ + ᾶ*sqrt(γrr)*Umθ*(cm/cp-1)  )
#
# # Uprb = @part 1 ( -Umr + (∂rcm/cm - ∂rcp/cp)*sqrt(γrr) - (1+cp/cm)*γrr/Umθ/2
# #  - (1+cm/cp)*Umθ*γrr/γθθ/2 + 4*pi*γrr*γθθ*((1+cp/cm)*ρ - (1-cp/cm)*Sr/sqrt(γrr))/Umθ )
#
# Uprb = @part 1 ( -Umr + (∂rcm/cm - ∂rcp/cp)*sqrt(γrr) - (1+cp/cm)*γrr/Umθ/2
#  - (1+cm/cp)*Umθ*γrr/γθθ/2 + 2*pi*γrr*γθθ*(1+cm/cp)*Um𝜙^2/Umθ )

# Uprb = @part 1 Krri + frrri/sqrt(γrri)

# ∂tKrr[1]  += s*(Uprb - Upr)/(dr̃*σ00)/2.
# ∂tfrrr[1] += s*sqrt(γrr[1])*(Uprb - Upr)/(dr̃*σ00)/2.

# ∂tUmr = ∂tKrr[1] - ∂tfrrr[1]/sqrt(γrr[1]) + frrr[1]*∂tγrr[1]/2/sqrt(γrr[1])^3
#
# #∂tUpr = 0. + 4*pi*α[1]*(γrr[1]*Tt[1] - 2*Srr[1]) + 16*pi*α[1]*sqrt(γrr[1])*Sr[1]
#
# # ∂tUpθ = @part 1 (α - (-βr + α/sqrt(γrr))*∂rUpθ + Umr*Upθ*α/γrr
# #     + (Upθ - Umθ)*Upθ*α/γθθ - α*∂rlnᾶ*Upθ/sqrt(γrr) + 4*pi*α*(γθθ*Tt - 2*Sθθ) )
#
# #Uprb = @part 1 ( -∂tγrr/α/2 + βr*frrr/α + frrr/sqrt(γrr) + ∂rβr*γrr/α - 4*frθθ*βr*γrr/γθθ/α )
#
# ∂tUpr = 0. #+ s*(Uprb - Upr)/(dr̃*σ00)/2
#


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
