
# Type to use for all calculations
const T = Float64;

# Number of grid points
const n = 2000;

# Spatial coordinate domain span in units of M0
const r̃span = T[10.0,15.0];

#rspan = T[r̃min,r̃max*10.]
# f(x) = x*tan((r̃max-r̃min)/x) + r̃min - rspan[2]
#
# rs = find_zero(f, 0.64*r̃max)
#
# r(r̃) = rs*tan((r̃-r̃min)/rs) + r̃min
# drdr̃(r̃) = sec((r̃-r̃min)/rs)^2
# d2rdr̃(r̃) = (2/rs)*(sec((r̃-r̃min)/rs)^2)*tan((r̃-r̃min)/rs)

r(r̃) = r̃
drdr̃(r̃) = 1.
d2rdr̃(r̃) = 0.

# Inverse Courant factor (4 works fine, CFL condition demands >~ 1)
const CFL = T(4.);

# Spatial interval size
const dr̃ = T((r̃span[2]-r̃span[1])/(n-1));

# Temporal interval size
const dt = dr̃/CFL::T;

# Temporal coordinate span in units of M0
const tspan = T[0., 10.];

# Interval between prints to the screen in units of M0
const printtimes = T(1.);

# Interval to save the state in units of M0
const savetimes = T(0.1);

# Initial mass of Black Hole
# M0 = 0. for flat Spherical Coordinates
# M0 > 0. for Schwarzschild black hole
const M0 = T(0.);

# Mass of scalar field
const m = T(0.);

# Total number of variables in the state vector
const numvar = 9

# Initial conditions on the scalar field
# Here is a pulse with amplitude A, total width 2*σr, and location r0
const r0 = T(12.5);
const σr = T(0.5);
const Amp  = T(0.2);

f𝜙(M,r,r̃) = (r0-σr)<r(r̃)<(r0+σr) ? (Amp/r(r̃))*(r(r̃)-(r0-σr))^4*(r(r̃)-(r0+σr))^4/σr^8 : 0.
#f𝜙(M,r,r̃) = (r0-σr)<r(r̃)<(r0+σr) ? Amp*(r(r̃)-(r0-σr))^4*(r(r̃)-(r0+σr))^4/σr^8 : 0.

# Initial conditions on the time derivative
# This asserts the pulse is initially moving at speed cm (defined in main program)
#f∂ₜ𝜙(M,r,r̃) = (r0-σr)<r(r̃)<(r0+σr) ? -(8*Amp*fcm(M,r,r̃)/r(r̃))*((r(r̃)-r0)^2-σr^2)^3*(r(r̃)-r0)/σr^8 : 0.

f∂ₜ𝜙(M,r,r̃) = fβʳ(M,r,r̃)*fψ(M,r,r̃)

# Magnitude of dissipation
# Must be of order 1.
const ε = T(0.5 /(2^6));

# Which variables to perform regularization
# State vector is ordered as:
#γrr,γθθ,Krr,Kθθ,frrr,frθθ,... = state.x
#const reg_list = Int64[1,2,3,4,5,6];
const reg_list = Int64[];
