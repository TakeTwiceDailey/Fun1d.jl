
using DoubleFloats
using SparseArrays

# Type to use for all calculations
const T = Float64;

macro T_str(str::AbstractString)
    :(parse(T,$str))
end

# Number of grid points
const n = 251;

# Spatial coordinate domain span in units of M0
const rspan = T[T"3.0",T"8.0"];

# Inverse Courant factor (4 works fine, CFL condition demands >~ 1)
const CFL::T = 4;

# Spatial interval size
const dr = T((rspan[2]-rspan[1])/(n-1));

# Temporal interval size
const dt = dr/CFL::T;

# Temporal coordinate span in units of M0
const tspan = T[T"0.", T"100."];

# Interval between prints to the screen in units of M0
const print_interval = T"10.0";

# Interval to save the state in units of M0
const save_interval = T"0.2";

# Initial mass of Black Hole
# M0 = 0. for flat Spherical Coordinates
# M0 > 0. for Schwarzschild black hole
const M0 = T"0.";

# Mass of scalar field
const m = T"0.";

# Initial conditions on the scalar field
# Here is a pulse with amplitude A, total width 2*σr, and location r0
const r0 = T"5.";
const σr = T"0.5";
const Amp  = T"0.1";
const p = 4;

f𝜙(M,r) = (r0-σr)<r<(r0+σr) ? (Amp/r)*(r-(r0-σr))^p*(r-(r0+σr))^p/σr^(2*p) : 0
#f𝜙(M,r) = (Amp/r)*exp(-((r-r0)/(σr/4))^2)

#k=4.948926441009052
#f𝜙(M,r) = (Amp/r)*sin(k*(pi/2)*(r-rspan[1])/(rspan[2]-rspan[1]))
#f𝜙(M,r) = (Amp/r)*sin((pi/2)*(r-rspan[1])/(rspan[2]-rspan[1]))

# Initial conditions on the time derivative
# This asserts the pulse is initially moving at speed cm (defined in main program)
#f∂ₜ𝜙(M,r,r̃) = (r0-σr)<r(r̃)<(r0+σr) ? -(8*Amp*fcp(M,r,r̃)/r(r̃))*((r(r̃)-r0)^2-σr^2)^3*(r(r̃)-r0)/σr^8 : 0

f∂ₜ𝜙(M,r) = fβʳ(M,r)*f∂ᵣ𝜙(M,r)

# Magnitude of dissipation
# Must be of order 1.
const ε = T"1.0"/(2^6);

# Which variables to perform regularization
# State vector is ordered as:
#γrr,γθθ,Krr,Kθθ,frrr,frθθ,... = state.x
#const reg_list = Int64[1,2,3,4,5,6];
const reg_list = Int64[];

include("SBP_coeffs_6-3.jl")
