module MushroomBody

export run_model, create_synapses!, sparsedensemult, fillcells!, fillentries!, calc_output!#, get_parameters, create_synapses!

#using JLD2, FileIO
using SparseArrays

include("UpdateWeights.jl")
include("initialisers.jl")
include("Parameters.jl")
include("Rewards.jl")
include("UpdateActivation.jl")
include("helpers.jl")
#include("Parameters2.jl")



function run_model()

	# preallocating all the parameters
	nn = SA[1000, 10000, 1]
	nl = length(nn)

	# neuron types
	activation = [Vector{Float64}(undef,i) for i in nn]
	rec = [Vector{Float64}(undef,i) for i in nn]
	spiked = [Vector{Bool}(undef,i) for i in nn]
	spt = [Vector{Int}(undef,i) for i in nn]
	I = [Vector{Float64}(undef,i) for i in nn]

	#synapse types
	synapses = [spzeros(nn[i], nn[i+1]) for i = 1:length(nn)-1]
	#weights = [spzeros(nn[i], nn[i+1]) for i = 1:length(nn)-1]
	γ = [spzeros(nn[i], nn[i+1]) for i = 1:length(nn)-1]
	ACh = [spzeros(nn[i], nn[i+1]) for i = 1:length(nn)-1]
	output = [spzeros(nn[i], nn[i+1]) for i = 1:length(nn)-1]
	#actually should output be a neuron-type?

	#parameters = load("Parameters.jld2")

	# create layers
	fillcells!(activation,-60.0)
	fillcells!(rec,0.0)
	fillcells!(spiked,false)
	fillcells!(spt,-1)
	fillcells!(I,250)
	#I[1] .= I[1] .+ generate_noise(50,nn[1])

	create_synapses!(synapses, 1)
	weights = 2 .* synapses


	for t = 1:10

		for l = 1:length(nn)

			update_activation!(nn[l], activation[l], vr[l], spiked[l], spt[l], t, vt[l], rec[l], I[l], C[l], a[l], b[l], c[l], d[l], k[l])
			#da = update_da(da, BA, τ, δt)

			da = 1

			if l != length(nn)


				calc_output!(output[l], weights[l], ACh[l], rev[l], activation[l])

				update_weights!(weights[l], γ[l], synapses[l], t, spt[l], spt[l+1], da, tconst[l]; δt=1, A₋=-1, t₋=15)				

			end

		end
	end

end


function get_parameters(parameters,paramlist)

	out = []

	for item in paramlist

		push!(out, parameters[item])

	end

	return out

end

end # module
