using JLD2, FileIO

function saveparams()

#PerLayers
nn = SA[20, 100, 1]
C = SA[100 4 100]
cap = SA[100 4 100];
a = SA[0.3 0.01 0.3];
b = SA[-0.2 -0.3 -0.2];
k = SA[2 0.035 2];
vt = SA[-40 -25 -40];
dvoltage = SA[-60 -85 -60];
synt = SA[3, 8, undef];
quantile = SA[0.93, 8, undef];

# same over layers
c = SA[-65];
d = SA[8];
noisestd = SA[0.05];
vr = SA[-60]
t₋ = SA[15]
A₋ = SA[-1]
tconst = SA[40]
rev = SA[0]


@save "Parameters.jld2" nn C cap a b k vt dvoltage synt quantile c d noisestd vr t₋ A₋ tconst rev

end