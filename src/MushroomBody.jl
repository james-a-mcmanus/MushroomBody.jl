module MushroomBody

export run_model, get_parameters, MatrixTypes, NeuronLayer, clone_synapses, NeuronLayers, ConnectionLayer, SynapseLayers, SynapseLayer, create_synapses, initialise_matrices, initialise_matrices_old, Trainer, create_synapses!, update_weights!, create_input, sim_spikes, test_weight, test_transmission, test_ach, test_da, sparsedensemult, fillcells!, fillentries!, calc_input!,update_activation!, update_γ!


using SparseArrays

include("UpdateWeights.jl")
include("UpdateActivation.jl")
include("initialisers.jl")
include("Parameters.jl")
include("Neurotransmitters.jl")
include("helpers.jl")
include("Tests.jl")
include("input.jl")
include("plotters.jl")

struct NeurotransmitterTypes
	da::Number
	BA::Number
end

function run_model()

	nn = SA[10, 100, 1]
	numsteps  = 100
	in1 = create_input(nn[1], 350:450, numsteps, numsteps, [50,50], 0.8, BAstart=10)

	run_model(in1,nn)
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

	parameters = initialise_parameters()

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

	nn = [1,100,10]

	m = MatrixTypes(initialise_matrices(nn)...)
	p = get_parameters

	for t = 1:numsteps()
		inputandreward()
		for layer = 1:length(nn)
			update_activation!(t, layer, nn, m, p) #these haven't been defined: maybe have another get function for these?
			if layer==length(nn)
				update_ACh!(t, layer, m, p, da)
				calc_input!(layer, m, p)
				update_weights!(t, layer, m, p, da)
			end
		end
	end
end

function inputandreward(t, input, in1, da, τ)

	input[1] .= in1.inarrayseq[t]==1 ? in1.inarray[1] : input[1]
	BA = in1.BAseq[t]
	da = update_da(da, BA, τ)

	return(BA,da)	
end

end
