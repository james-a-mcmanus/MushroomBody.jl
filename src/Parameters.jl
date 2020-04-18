#=module Parameters

export nn, c, d, C, noisestd, vr, cap, a, b, k, vt, dvoltage, synt, quantile, t₋, A₋, tconst=#

using StaticArrays

nn = SA[20, 100, 1]
c = SA[-65 -65 -65];
d = SA[8 8 8];
C = SA[100 4 100]
noisestd = SA[0.05 0.05 0.05];    
#input_activity = 40 * num_neurons(1);

vr = SA[-60 -60 -60]
cap = SA[100 4 100];
a = SA[0.3 0.01 0.3];
b = SA[-0.2 -0.3 -0.2];
k = SA[2 0.035 2];
vt = SA[-40 -25 -40];
dvoltage = SA[-60 -85 -60];
synt = SA[3, 8, undef];
quantile = SA[0.93, 8, undef];
t₋ = SA[15, 15, 15]
A₋ = SA[-1, -1, -1]
tconst = SA[40, 40, 40]
rev = SA[0, 0, 0]
da = [0 0 0]
Φ = [0.5, 0.5, 0.5]
