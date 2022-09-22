###### Parameters for the 4-3 SBP Operator ######

# Obtained from arXiv:gr-qc/0512001

# I have chosen the D_{4-3} operator from this paper.
# In order to find the explicit numerical values of the coefficients
# for this operator and others along with the norms, follow the
# link and under Download on the arXiv page, click 'Other formats',
# and download the 'Source' format. Use a text editor to view the file,
# and you can read off the coefficients as directed
# in the appendix of the paper.

# Coefficents of the inverse norm σi_{mn}

const σi11 =  48/17.
const σi22 =  48/59.
const σi33 =  48/43.
const σi44 =  48/49.

# The norm is block diagonal and symmetric, with the interior
# like the identity matrix.
# You can also see the apparent 'restricted-full' form
# of the norm, as it is diagonal on the boundary.
# Form the 'left' and 'right' blocks of the matrix:

const σil = T[ σi11  0.   0.   0.  ;
                0.  σi22  0.   0.  ;
                0.   0.  σi33  0.  ;
                0.   0.   0.  σi44 ]

const σir = σil[end:-1:1,end:-1:1];

# First derivative coefficents
# Defines the coefficents of the D_{4-3} operator q_{ij}

const q11 = -24/17.; const q21 = 59/34. ;
const q31 = -4/17. ; const q41 = -3/34. ;
const q51 = 0.     ; const q61 =  0.    ;

const q12 = -1/2.  ; const q22 = 0.     ;
const q32 = 1/2.   ; const q42 = 0.     ;
const q52 = 0.     ; const q62 = 0.     ;

const q13 =  4/43. ; const q23 = -59/86.;
const q33 =  0.    ; const q43 = 59/86. ;
const q53 =  -4/43.; const q63 = 0.     ;

const q14 = 3/98.  ; const q24 = 0.     ;
const q34 = -59/98.; const q44 = 0.     ;
const q54 = 32/49. ; const q64 = -4/49. ;

# The D_{4-3} operator is only different from a 4th order
# centered finite differencing operator at the left and right boundaries
# Form these 'left' and 'right' blocks:

const ql =  T[ q11 q21 q31 q41 q51 q61 ;
               q12 q22 q32 q42 q52 q62 ;
               q13 q23 q33 q43 q53 q63 ;
               q14 q24 q34 q44 q54 q64 ]

# Important to note that there is a minus sign difference on the right block

const qr = -ql[end:-1:1,end:-1:1];

# Now put all of these numbers into sparse arrays

# Define a sparse 5-diagonal 4th order centered finite differencing operator
Dc = spdiagm(-2=>ones(T,n-2),-1=>-8*ones(T,n-1),1=>8*ones(T,n-1),2=>-ones(T,n-2))/12.;
# Overwrite the left and right blocks
Dc[1:4,1:6] .= ql; Dc[n-3:n,n-5:n] .= qr;

# Define SBP differencing operator
const D = Dc/dr̃;

# Next form the dissipation operator

# Boundary operator (Identity matrix except near the boundary)
# tr dictates the transition region size, where it increase linearly
# from dr̃ to the interior value of 1. Here I chose 5% of the domain size.
Bvec = ones(T,n);
# tr = round(Int64, n/20);
# for i in 1:tr Bvec[i] = T(((i-1) + (tr-(i-1))*dr̃)/tr) end
# Bvec[n:-1:n-(tr-1)] .= Bvec[1:tr];
B2 = spdiagm(0=>Bvec);

# Second Derivative Operator
# Coefficents as defined in the paper
D2 = spdiagm(-1=>ones(T,n-1),0=>-2*ones(T,n),1=>ones(T,n-1));
D2[1,1:3] .= D2[2,1:3]; D2[n,n-2:n] .= D2[n-1,n-2:n];

# Inverse norm
# Form the full sparse inverse norm matrix
diag = spdiagm(0=>ones(T,n));
diag[1:4,1:4] .= σil; diag[n-3:n,n-3:n] .= σir;

const Σi = copy(diag);

# Norm
# Form the full sparse norm matrix
# by taking the inverse of the left and right blocks.
diag[1:4,1:4] .= inv(σil); diag[n-3:n,n-3:n] .= inv(σir);

const Σ = copy(diag);

# Complete construction of the dissipation operator
const A4 = -ε*Σi*(D2')*B2*D2;
