module MushroomBody

export run_model, networkplot, plotconnections, Dashplot, train_model, get_parameters, MatrixTypes, NeuronLayer, clone_synapses, NeuronLayers, ConnectionLayer, SynapseLayers, SynapseLayer, create_synapses, initialise_matrices, initialise_matrices_old, Trainer, create_synapses!, update_weights!, create_input, sim_spikes, test_weight, test_transmission, test_ach, test_da, sparsedensemult, fillcells!, fillentries!, calc_input!,update_activation!, update_γ!


using SparseArrays

include("UpdateWeights.jl")
include("UpdateActivation.jl")
include("initialisers.jl")
include("Neurotransmitters.jl")
include("helpers.jl")
include("Tests.jl")
include("input.jl")
include("Parameters.jl")
include("plotters.jl")

struct NeurotransmitterTypes
	da::Number
	BA::Number
end

function run_model()

	nn = [100, 1_000, 1]
	numsteps  = 100
	in1 = create_input(nn[1], 350:450, numsteps, numsteps, [50,50], 0.8, BAstart=10)

	weights = train_model(in1,nn,numsteps)
	test_model(in1,nn,numsteps,weights)
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
		dashboard(plt, t, sl, m, da)

	end

	return(synapses, weights)
end

function test_model(in1, nn, numsteps, weights)

	m = MatrixTypes(initialise_matrices(nn, weights)...)
	p = get_parameters()
	da = 0
	
	plt = Dashplot()

	for t = 1:numsteps
	
		BA, da = inputandreward!(t, m.input.layers[1], in1, p.τ[1], da=da)
		for layer = 1:length(nn)
			update_activation!(t, layer, nn, m, p) #these haven't been defined: maybe have another get function for these?
			if layer !== length(nn)
				update_ACh!(t, layer, m, p, da)
				calc_input!(layer, m, p)
			end
			sl=1
			dashboard(plt, t, sl, m, da)
		end
	end
end

function train_model(in1, nn, numsteps)

	m = MatrixTypes(initialise_matrices(nn)...)
	p = get_parameters()
	da = 0

	for t = 1:numsteps
	
		BA, da = inputandreward!(t, m.input.layers[1], in1, p.τ[1], da=da)
		for layer = 1:length(nn)
			update_activation!(t, layer, nn, m, p) #these haven't been defined: maybe have another get function for these?
			if layer !== length(nn)
				update_ACh!(t, layer, m, p, da)
				calc_input!(layer, m, p)
				if layer == 2
					update_weights!(t, layer, m, p, da)
				end
			end
		end
	end
	return m.weights
end

function inputandreward!(t, input, in1, τ; da=0)

	input .= in1.inarrayseq[t]==1 ? in1.inarray[1] : input*1
	BA = in1.BAseq[t]
	da = update_da(da, BA, τ)

	return(BA,da)	
end

end
