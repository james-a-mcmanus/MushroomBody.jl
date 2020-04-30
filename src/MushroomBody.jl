module MushroomBody

export 
run_model, 
gif_model, 
networkplot, 
neuronplot!, 
plotconnections!, 
Dashplot, train_model, 
get_parameters, 
MatrixTypes, 
NeuronLayer, 
clone_synapses, 
NeuronLayers, ConnectionLayer, 
SynapseLayers, 
SynapseLayer, 
create_synapses, 
initialise_matrices, 
initialise_matrices_old, 
Trainer, create_synapses!, 
update_weights!, create_input, 
sim_spikes, 
test_weight, 
test_transmission, 
test_ach, 
test_da, 
sparsedensemult, 
fillcells!, 
fillentries!, 
calc_input!,
update_activation!, 
update_γ!,
SparseInput


using SparseArrays, Infiltrator

include("UpdateWeights.jl")
include("UpdateActivation.jl")
include("initialisers.jl")
include("Neurotransmitters.jl")
include("helpers.jl")
include("Tests.jl")
include("input.jl")
include("Parameters.jl")
include("plotters.jl")

"""
run the model, i.e. put through a training phase and a test phase.
"""
function run_model()

	nn = [100, 1_000, 5]
	numsteps  = 50
	in1 = create_input(nn[1], 350:450, numsteps, numsteps, [50,50], 0.8, BAstart=10)

	weights, synapses = train_model(in1,nn,numsteps, showplot=true)
	test_model(in1,nn,numsteps,weights, synapses)# calcs an input and then calls run_model(::RandInput).
end
function run_model(gf::GifPlot)
	nn = [100, 1_000, 5]
	numsteps  = 50
	in1 = create_input(nn[1], 350:450, numsteps, numsteps, [50,50], 0.8, BAstart=10)

	weights, synapses, gf = train_model(in1,nn,numsteps,gf)
	gf = test_model(in1,nn,numsteps,weights,synapses,gf)
	gif(gf.anim,gf.fname,fps=15)# runs model and saves as gif
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

	return(synapses, weights)## basic form of model.
end

"""
save a run of the model as a gif
"""
function gif_model()
	outfolder = "C:\\Users\\James\\.julia\\dev\\MushroomBody\\src\\output\\"
	outname = "plotest.gif"
	gf = GifPlot(outfolder * outname)
	run_model(gf)
end

"""
train the model, returns weights and synapses
"""
function train_model(in1, nn, numsteps; showplot=false, gifplot=true)

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
		showplot && shownetwork(init_plot(), t, nn, m)
	end

	return (m.weights, m.synapses)
end
function train_model(in1, nn, numsteps, gf::GifPlot)

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
		gf = gifnetwork(gf, init_plot(), t, nn, m)		
	end

	return (m.weights, m.synapses, gf)
end

"""
Test the model, takes weights
"""
function test_model(in1, nn, numsteps, weights, synapses, gf::GifPlot)

	m = MatrixTypes(initialise_matrices(nn, weights, synapses)...)
	p = get_parameters()
	da = 0

	for t = 1:numsteps
	
		BA, da = inputandreward!(t, m.input.layers[1], in1, p.τ[1], da=da)
		for layer = 1:length(nn)
			update_activation!(t, layer, nn, m, p) #these haven't been defined: maybe have another get function for these?
			if layer !== length(nn)
				update_ACh!(t, layer, m, p, da)
				calc_input!(layer, m, p)
			end
		end
		gf = gifnetwork(gf, init_plot(), t, nn, m)
	end	

	return gf
end
function test_model(in1, nn, numsteps, weights, synapses; showplot=false)

	m = MatrixTypes(initialise_matrices(nn, weights, synapses)...)
	p = get_parameters()
	da = 0

	for t = 1:numsteps
	
		BA, da = inputandreward!(t, m.input.layers[1], in1, p.τ[1], da=da)
		for layer = 1:length(nn)
			update_activation!(t, layer, nn, m, p) #these haven't been defined: maybe have another get function for these?
			if layer !== length(nn)
				update_ACh!(t, layer, m, p, da)
				calc_input!(layer, m, p)
			end
		end
		
		showplot && shownetwork(init_plot(), t, nn, m)

	end
end


"""
wrapper for input and reward functions
"""
function inputandreward!(t, input, in1, τ; da=0)

	input .= in1.inarrayseq[t]==1 ? in1.inarray[1] : input*1
	BA = in1.BAseq[t]
	da = update_da(da, BA, τ)

	return(BA,da)	
end

end
