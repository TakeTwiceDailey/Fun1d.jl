
#using DoubleFloats
using SparseArrays
using BandedMatrices

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
const n = 5001;

# Spatial coordinate domain span in units of M0

const rspan = T[T"2.1",T"203.0"];

# Inverse Courant factor (4 works fine, CFL condition demands >~ 1)
const CFL::T = 4;

# Spatial interval size
const dr = T((rspan[2]-rspan[1])/(n-1));

# Temporal interval size
const dt = dr/(CFL)::T;

# Temporal coordinate span in units of M0
const tspan = T[T"0.", T"250."];

# Interval between prints to the screen in units of M0
const print_interval = T"10.";

# Interval to save the state in units of M0
const save_interval = T"1.";

# Mass of scalar field (not properly implemented on boundaries)
const m = T"0.0";

# Initial conditions on the scalar field
# Here is a pulse with amplitude A, total width 2*σr, and location r0
# const σr = T"20.0"; const Amp  = T"0.1649";
# const σr = T"19.0"; const Amp  = T"0.1508";
# const σr = T"18.0"; const Amp  = T"0.1565";
# const σr = T"17.0"; const Amp  = T"0.1522";
# const σr = T"16.0"; const Amp  = T"0.1477";

# const σr = T"20.0"; const Amp  = T"0.18060"; Mtot = 1.06
# const σr = T"22.0"; const Amp  = T"0.20436"; Mtot = 1.07
# const σr = T"25.0"; const Amp  = T"0.23317"; Mtot = 1.08
# const σr = T"30.0"; const Amp  = T"0.28550"; Mtot = 1.1
# const σr = T"100.0"; const Amp  = T"3.0"; Mtot = 4.29
# const σr = T"90.0"; const Amp  = T"2.6";

const r0 = T"103.0";
const σr = T"100.0"; const Amp  = T"1.3";

f𝜙(M,r) = (r0-σr)<r<(r0+σr) ? (Amp/r)*(r-(r0-σr))^4*(r-(r0+σr))^4/σr^8 : 0
#f𝜙(M,r) = (Amp/r)*exp(-((r-r0)/(σr/4))^2)
#f𝜙(M,r) = 0.

#k=4.948926441009052
#f𝜙(M,r) = (Amp/r)*sin(k*(pi/2)*(r-rspan[1])/(rspan[2]-rspan[1]))
#f𝜙(M,r) = (Amp/r)*sin((pi/2)*(r-rspan[1])/(rspan[2]-rspan[1]))

# Initial conditions on the time derivative
# This asserts the pulse is initially moving at speed cm (defined in main program)
f∂ₜ𝜙(M,r) = (r0-σr)<r<(r0+σr) ? -(8*Amp*fcm(M,r)/r)*((r-r0)^2-σr^2)^3*(r-r0)/σr^8 : 0

# f𝜙(M,r)   = (r0-σr)<r<(r0+σr) ? (Amp/r)*((r-c*t)-(r0-σr))^4*((r-c*t)-(r0+σr))^4/σr^8 : 0
# f∂ₜ𝜙(M,r)  = (r0-σr)<r<(r0+σr) ? -(8*Amp*c/r)*(((r-c*t)-r0)^2-σr^2)^3*((r-c*t)-r0)/σr^8 : 0

# Inner Boundary Condition
const ka = 0

# Outer Boundary Condition
const kb = 0

# Magnitude of dissipation
# Must be of order 1.
const ε = T"2.0"/(2^6);

# Which variables to perform regularization
# State vector is ordered as:
#γrr,γθθ,Krr,Kθθ,frrr,frθθ,... = state.x
#const reg_list = Int64[1,2,3,4,5,6];
#const reg_list = Int64[];

include("SBP_coeffs_6-3.jl")
