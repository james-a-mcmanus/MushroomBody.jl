module MushroomBody

export run_model, create_synapses!, update_weights!, sim_spikes, test_weight, test_transmission, test_ach, test_da, sparsedensemult, fillcells!, fillentries!, calc_input!,update_activation!, update_γ!


using SparseArrays, Debugger

include("UpdateWeights.jl")
include("initialisers.jl")
include("Parameters.jl")
include("Neurotransmitters.jl")
include("UpdateActivation.jl")
include("helpers.jl")
include("Tests.jl")
#include("plotters.jl")

function run_model()

	# preallocating all the parameters
	nn = SA[10, 100, 1]

	# neuron types
	activation = [Vector{Float64}(undef,i) for i in nn]
	rec = [Vector{Float64}(undef,i) for i in nn]
	spiked = [Vector{Bool}(undef,i) for i in nn]
	spt = [Vector{Int}(undef,i) for i in nn]
	I = [Vector{Float64}(undef,i) for i in nn]
	ACh = [Vector{Float64}(undef,i) for i in nn]
	input = [Vector{Float64}(undef,i) for i in nn]

	#synapse types
	synapses = [spzeros(nn[i], nn[i+1]) for i = 1:length(nn)-1]
	γ = [spzeros(nn[i], nn[i+1]) for i = 1:length(nn)-1]
	#input = [spzeros(nn[i], nn[i+1]) for i = 1:length(nn)-1]

	# create layers
	fillcells!(activation,-60.0)
	fillcells!(rec,0.0)
	fillcells!(spiked,false)
	fillcells!(spt,-1)
	fillcells!(I,250)
	fillcells!(ACh, 0.0)
	fillcells!(input, 0.0)
	#I[1] .= I[1] .+ generate_noise(50,nn[1])

	create_synapses!(synapses, 1)
	weights = 2 .* synapses

	for t = 1:10

		for l = 1:length(nn)

			update_activation!(nn[l], activation[l], vr[l], spiked[l], spt[l], t, vt[l], rec[l], input[l], C[l], a[l], b[l], c[l], d[l], k[l])
			
			#da = update_da!(da, BA, τ)
			da = 0

			if l != length(nn)

				# output functions etc.
				update_ACh!(ACh[l], synt[l], Φ[l], t, spt[l]) #ACh should be a neuron type...
				calc_input!(input[l+1], weights[l], ACh[l], rev[l], activation[l+1])
				update_weights!(weights[l], γ[l], synapses[l], t, spt[l], spt[l+1], da, tconst[l]; δt=1, A₋=-1, t₋=15)

			end


		end
	end
end
end # module
