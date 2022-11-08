
using DoubleFloats
using SparseArrays

# Type to use for all calculations
const T = Float64;

macro T_str(str::AbstractString)
    :(parse(T,$str))
end

# Initial mass of Black Hole
# M0 = 0. for flat Spherical Coordinates
# M0 > 0. for Schwarzschild black hole
const M0 = T"1.";

# Number of grid points
const n = 32001;

# Spatial coordinate domain span in units of M0

const rspan = T[T"2.01",T"6.0"];

# const C =  M0*(rspan[2]+2*M0)/(rspan[2]-2*M0)

# fr̃(r) = r - M0*(r+2*M0)/(r-2*M0) + C
# fr(r̃) = (3*M0-C+r̃+sqrt(17*M0^2+2*C*M0+C^2-2*(M0+C)*r̃+r̃^2))/2
# fdrdr̃(r̃) = ForwardDiff.derivative(x -> fr(x), r̃)
fr̃(r) = r
fr(r̃) = r̃
fdrdr̃(r̃) = ForwardDiff.derivative(x -> fr(x), r̃)

const r̃span = T[fr̃(rspan[1]),fr̃(rspan[2])];

# Inverse Courant factor (4 works fine, CFL condition demands >~ 1)
const CFL::T = 4;

# Spatial interval size
const dr̃ = T((r̃span[2]-r̃span[1])/(n-1));

# Temporal interval size
const dt = dr̃/(CFL)::T;

# Temporal coordinate span in units of M0
const tspan = T[T"0.", T"20."];

# Interval between prints to the screen in units of M0
const print_interval = T"1.0";

# Interval to save the state in units of M0
const save_interval = T"0.1";

# Mass of scalar field
const m = T"0.";

# Initial conditions on the scalar field
# Here is a pulse with amplitude A, total width 2*σr, and location r0
const r0 = T"4.";
const σr = T"0.5";
const Amp  = T"0.001";
#const Amp  = T"0.0";
const p = 4;

f𝜙(M,r) = (r0-σr)<r<(r0+σr) ? (Amp/r)*(r-(r0-σr))^p*(r-(r0+σr))^p/σr^(2*p) : 0
#f𝜙(M,r) = (Amp/r)*exp(-((r-r0)/(σr/4))^2)

#k=4.948926441009052
#f𝜙(M,r) = (Amp/r)*sin(k*(pi/2)*(r-rspan[1])/(rspan[2]-rspan[1]))
#f𝜙(M,r) = (Amp/r)*sin((pi/2)*(r-rspan[1])/(rspan[2]-rspan[1]))

# Initial conditions on the time derivative
# This asserts the pulse is initially moving at speed cm (defined in main program)
f∂ₜ𝜙(M,r) = (r0-σr)<r<(r0+σr) ? -(8*Amp*fcm(M,r)/r)*((r-r0)^2-σr^2)^3*(r-r0)/σr^8 : 0

#f∂ₜ𝜙(M,r) = fβʳ(M,r)*f∂ᵣ𝜙(M,r)

# Magnitude of dissipation
# Must be of order 1.
const ε = T"1.0"/(2^6);

# Which variables to perform regularization
# State vector is ordered as:
#γrr,γθθ,Krr,Kθθ,frrr,frθθ,... = state.x
#const reg_list = Int64[1,2,3,4,5,6];
const reg_list = Int64[];

include("SBP_coeffs_6-3.jl")
