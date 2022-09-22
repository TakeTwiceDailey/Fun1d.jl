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

const σi11 =  4.18659537039222689736221685976984622636900

const σi22 =  0.67251919212256207318887148369831164208710
const σi32 =  0.36134181811349492593705029667363069843670
const σi42 = -0.20213161172938997914816745396318796627070
const σi52 =  0.03455320708729270824077678274955265350304

const σi33 =  0.72061337116301470577204420986238473629500
const σi43 =  0.13764723405465693683216163897649587925910
const σi53 = -0.04136405531324488624637892257286207044784

const σi44 =  0.95786536079310268220741334414499096895090
const σi54 =  0.02069353627247161734563597102894256809696

const σi55 =  0.99082727033708614730077989259069683806540

# The norm is block diagonal and symmetric, with the interior
# like the identity matrix.
# You can also see the apparent 'restricted-full' form
# of the norm, as it is diagonal on the boundary.
# Form the 'left' and 'right' blocks of the matrix:

const σil = T[ σi11  0.   0.   0.   0.  ;
                0.  σi22 σi32 σi42 σi52 ;
                0.  σi32 σi33 σi43 σi53 ;
                0.  σi42 σi43 σi44 σi54 ;
                0.  σi52 σi53 σi54 σi55 ]

const σir = σil[end:-1:1,end:-1:1];

# First derivative coefficents
# Defines the coefficents of the D_{4-3} operator q_{ij}

const q11 = -2.0932976346634987158873300; const q21 =  4.0398572053206615302160000;
const q31 = -3.0597858079809922953240000; const q41 =  1.3731905386539948635493300;
const q51 = -0.2599643013301653825540000; const q61 =  0.; const  q71 = 0.;

const q12 = -0.3164158528594044527229700; const q22 = -0.5393078897398042232738800;
const q32 =  0.9851773202864434338329700; const q42 = -0.0526466598929757814670900;
const q52 = -0.1138072517506242350132580; const q62 =  0.0398797678898499118031030;
const q72 = -0.0028794339334846531588787;

const q13 =  0.1302691618502116452445200; const q23 = -0.8796685899505924925689000;
const q33 =  0.3860964096110007000013400; const q43 =  0.3135836907243558874598800;
const q53 =  0.0853189419136783846335110; const q63 = -0.0390466157927346402746410;
const q73 =  0.0034470016440805155042908;

const q14 = -0.0172451219382464791217200; const q24 =  0.1627228822712750438113400;
const q34 = -0.8134981024864881302921700; const q44 =  0.1383326926647983321564500;
const q54 =  0.5974385432854805339961600; const q64 = -0.0660264343462998876193240;
const q74 = -0.0017244594505194129307249;

const q15 = -0.0088356946855219296506100; const q25 =  0.0305607475920320385728400;
const q35 =  0.0502116827453085423227800; const q45 = -0.6630736465244492953406800;
const q55 =  0.0148787874640051911160880; const q65 =  0.6588270638170747195382000;
const q75 = -0.0825689404084492665586150;

# The D_{4-3} operator is only different from a 4th order
# centered finite differencing operator at the left and right boundaries
# Form these 'left' and 'right' blocks:

const ql =  T[ q11 q21 q31 q41 q51 q61 q71 ;
               q12 q22 q32 q42 q52 q62 q72 ;
               q13 q23 q33 q43 q53 q63 q73 ;
               q14 q24 q34 q44 q54 q64 q74 ;
               q15 q25 q35 q45 q55 q65 q75 ]

# Important to note that there is a minus sign difference on the right block

const qr = -ql[end:-1:1,end:-1:1];

# Now put all of these numbers into sparse arrays

# Define a sparse 5-diagonal 4th order centered finite differencing operator
Dc = spdiagm(-2=>ones(T,n-2),-1=>-8*ones(T,n-1),1=>8*ones(T,n-1),2=>-ones(T,n-2))/12.;
# Overwrite the left and right blocks
Dc[1:5,1:7] .= ql; Dc[n-4:n,n-6:n] .= qr;

# Define SBP differencing operator
const D = Dc/dr̃;

# Next form the dissipation operator

# Boundary operator (Identity matrix except near the boundary)
# tr dictates the transition region size, where it increase linearly
# from dr̃ to the interior value of 1. Here I chose 5% of the domain size.
Bvec = ones(T,n); tr = round(Int64, n/20);
for i in 1:tr Bvec[i] = T(((i-1) + (tr-(i-1))*dr̃)/tr) end
Bvec[n:-1:n-(tr-1)] .= Bvec[1:tr]; B2 = spdiagm(0=>Bvec);

# Second Derivative Operator
# Coefficents as defined in the paper
D2 = spdiagm(-1=>ones(T,n-1),0=>-2*ones(T,n),1=>ones(T,n-1));
D2[1,1:3] .= D2[2,1:3]; D2[n,n-2:n] .= D2[n-1,n-2:n];

# Inverse norm
# Form the full sparse inverse norm matrix
diag = spdiagm(0=>ones(T,n));
diag[1:5,1:5] .= σil; diag[n-4:n,n-4:n] .= σir;

const Σi = dr̃*diag;

# Norm
# Form the full sparse norm matrix
# by taking the inverse of the left and right blocks.
diag[1:5,1:5] .= inv(σil); diag[n-4:n,n-4:n] .= inv(σir);

const Σ = dr̃*diag;

# Complete construction of the dissipation operator
const A = -ε*Σi*(D2')*B2*D2;
