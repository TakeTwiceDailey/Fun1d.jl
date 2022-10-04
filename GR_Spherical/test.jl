using DifferentialEquations, Plots

# Linear ODE which starts at 0.5 and solves from t=0.0 to t=1.0
prob = ODEProblem((u,p,t)->1.01u,0.5,(0.0,1.0))

integrator = init(prob,Tsit5();dt=1//2^(4),tstops=[0.5])
#pyplot()
plot(integrator)
for i in integrator
  display(plot!(integrator,vars=(0,1),legend=false))
end
step!(integrator); plot!(integrator,vars=(0,1),legend=false)