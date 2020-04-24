module MushroomBody

export run_model, NeuronLayer, NeuronLayers, SynapseLayers, SynapseLayer, create_synapses, initialise_matrices, initialise_matrices_old, Trainer, create_synapses!, update_weights!, create_input, sim_spikes, test_weight, test_transmission, test_ach, test_da, sparsedensemult, fillcells!, fillentries!, calc_input!,update_activation!, update_γ!


using SparseArrays

include("UpdateWeights.jl")
include("initialisers.jl")
include("Parameters.jl")
include("Neurotransmitters.jl")
include("UpdateActivation.jl")
include("helpers.jl")
include("Tests.jl")
include("input.jl")
include("plotters.jl")

struct Trainer

	activation
	rec
	spiked
	spt
	I
	ACh
	input
	synapses
	weights
	γ
	da # da might be a problem for immutability, probably modify in-place
end

function run_model()

	nn = SA[10, 100, 1]
	numsteps  = 100
	in1 = create_input(nn[1], 350:450, numsteps, numsteps, [50,50], 0.8, BAstart=10)

	run_model(in1,nn)
end

function run_model(in1,nn)

	run_model(in1, nn)
end

function run_model(t::Trainer)

	numsteps = 100

end

function run_model(in1::RandInput, nn; showplot=false)


	# preallocating all the parameters
	numsteps  = 100

	#something here like initialiseplotfunction that takes 
	#in the function defined at the top

	plt = Dashplot()

	activation, 
	rec, 
	spiked, 
	spt, 
	I, 
	ACh, 
	input, 
	synapses,
	weights,
	γ, 
	da = initialise_matrices(nn)

	for t = 1:numsteps

		BA, da = inputandreward(t, input,in1, da, τ)
		
		for l = 1:length(nn)

			update_activation!(nn[l], activation[l], vr[l], spiked[l], spt[l], t, vt[l], rec[l], input[l], C[l], a[l], b[l], c[l], d[l], k[l])

			if l != length(nn)

				innerlayerupdates!(l, ACh, synt, Φ, t, spt, input, weights, rev, activation, γ, synapses, da, tconst)
				
			end

		end

		sl=1
		dashboard(plt, t, weights[sl], activation[sl], spt[sl], da, rec[sl])

	end

	return(synapses, weights)
end

function test_model(in1, weights)

 5
end

function train_model(in1)

	activation, 
	rec, 
	spiked, 
	spt, 
	I, 
	ACh, 
	input, 
	synapses,
	weights,
	γ, 
	da = initialise_matrices()

	for t = 1:numsteps()

	end
end

function innerlayerupdates!(l, ACh, synt, Φ, t, spt, input, weights, rev, activation, γ, synapses, da, tconst)

	update_ACh!(ACh[l], synt[l], Φ[l], t, spt[l])
	calc_input!(input[l+1], weights[l], ACh[l], rev[l], activation[l+1])
	update_weights!(weights[l], γ[l], synapses[l], t, spt[l], spt[l+1], da, tconst[l]; δt=1, A₋=-1, t₋=15)
end

function inputandreward(t, input, in1, da, τ)

	input[1] .= in1.inarrayseq[t]==1 ? in1.inarray[1] : input[1]
	BA = in1.BAseq[t]
	da = update_da(da, BA, τ)

	return(BA,da)	
end


function initialise_matrices(nn)

	activation = NeuronLayers([NeuronLayer(-60.0, i) for i in nn])
	rec = NeuronLayers([NeuronLayer(0.0, i) for i in nn])
	spiked = NeuronLayers([NeuronLayer(false, i) for i in nn])
	spt = NeuronLayers([NeuronLayer(Int(-1), i) for i in nn])
	I = NeuronLayers([NeuronLayer(250.0, i) for i in nn])
	ACh = NeuronLayers([NeuronLayer(0.0, i) for i in nn])
	input = NeuronLayers([NeuronLayer(0.0, i) for i in nn])

	synapses = create_synapses(SynapseLayers,nn)
	γ = fill_synapses(SynapseLayers, nn, 0.0)

end



function initialise_matrices_old(nn)

	# neuron types
	activation = [Vector{Float64}(undef,i) for i in nn]
	rec = [Vector{Float64}(undef,i) for i in nn]
	spiked = [Vector{Bool}(undef,i) for i in nn]
	spt = [Vector{Int}(undef,i) for i in nn]
	I = [Vector{Float64}(undef,i) for i in nn]
	ACh = [Vector{Float64}(undef,i) for i in nn]
	input = [Vector{Float64}(undef,i) for i in nn]
	#synapse types
#=	synapses = [spzeros(nn[i], nn[i+1]) for i = 1:length(nn)-1]
	γ = [spzeros(nn[i], nn[i+1]) for i = 1:length(nn)-1]
	da = 0.0=#

	# create layers
	fillcells!(activation,-60.0)
	fillcells!(rec,0.0)
	fillcells!(spiked,false)
	fillcells!(spt,-1)
	fillcells!(I,250)
	fillcells!(ACh, 0.0)
	fillcells!(input, 0.0)

#=	create_synapses!(synapses, 1)
	weights = 2 .* synapses	
=#
end

end # module
