module MushroomBody

export 
run_model, 
gif_model,
response_before_after_learning,
plotmeanlayer,
meanlayer,
plot_before_after


using SparseArrays, Infiltrator

include("UpdateWeights.jl")
include("UpdateActivation.jl")
include("initialisers.jl")
include("Neurotransmitters.jl")
include("helpers.jl")
include("input.jl")
include("Parameters.jl")
include("plotters.jl")
include("Tests.jl")
include("SaveData.jl")

"""
run the model, i.e. put through a training phase and a test phase.
"""
function run_model()
	nn = [1000, 10000, 1]
	sensory = constructinputsequence((1000,), (SparseInput,), stages=[10,100,1], input_bool=Bool[1,1,0], da_bool=Bool[0,1,1])
	numsteps = duration(sensory)
	println("Number of Steps: $numsteps")
	weights, synapses = train_model(sensory,nn,numsteps, showplot=false)
	test_model(sensory,nn,numsteps,weights, synapses)
end
function run_model(gf::GifPlot)

	nn = [100, 1_000, 5]
	sensory = constructinputsequence((100,), (SparseInput,RestInput,SparseInput,RestInput,SparseInput), stages=[3,10,1], input_bool=Bool[1,1,0], da_bool=Bool[0,1,1])
	numsteps = duration(sensory)	
	weights, synapses, gf = train_model(sensory,nn,numsteps,gf)
	gf = test_model(sensory,nn,numsteps,weights,synapses,gf)
	gif(gf.anim,gf.fname,fps=15)# runs model and saves as gif
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

function run_all_steps(nn, numsteps, m, p, sensory, da; update=true, savevars=nothing, reportvar=nothing, showplot=false)
	for t = 1:numsteps
		da = run_step(t, m, p, sensory, da, nn, update=update)
		showplot && shownetwork(init_plot(), t, nn, m)
		!isnothing(savevars) && save_variables(m,savevars)
		!isnothing(reportvar) && returnvar(m,reportvar)
	end
end
function run_step(t, m, p, sensory, da, nn; update=true)
	BA, da = inputandreward!(t, m.input.layers[1], sensory, p.τ[1], p.da_on, da=da)
	for layer = 1:length(nn)
		run_layer(t,layer,nn,m,p,da,update=true)
	end	
	return da
end
function run_layer(t, layer, nn, m, p, da; update=true)
	update_activation!(t, layer, nn, m, p)
	if layer !== length(nn)
		update_pre_layers(t,layer,m,p,da, update=true)
	end	
end
function update_pre_layers(t, layer, m, p, da; update=true)
	update_ACh!(t, layer, m, p, da)
	calc_input!(layer, m, p)
	(layer == 2 && update) && update_weights!(t, layer, m, p, da)	
end

function train_model(sensory, nn, numsteps; showplot=true, gifplot=false, update=true, savevars=nothing, reportvar=nothing)

	m = MatrixTypes(initialise_matrices(nn)...)
	p = get_parameters()
	da = 0

	run_all_steps(nn, numsteps, m, p, sensory, da, savevars=savevars, reportvar=reportvar, update=true)

	return (m.weights, m.synapses)
end
function train_model(sensory, nn, numsteps, gf::GifPlot)

	m = MatrixTypes(initialise_matrices(nn)...)
	p = get_parameters()
	da = 0

	for t = 1:numsteps
	
		BA, da = inputyandreward!(t, m.input.layers[1], sensory, p.τ[1], da=da)
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
function test_model(sensory, nn, numsteps, weights, synapses, gf::GifPlot)

	m = MatrixTypes(initialise_matrices(nn, weights, synapses)...)
	p = get_parameters()
	da = 0

	for t = 1:numsteps
	
		BA, da = inputandreward!(t, m.input.layers[1], sensory, p.τ[1], da=da)
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
function test_model(sensory, nn, numsteps, weights, synapses; showplot=false, savevars=nothing, reportvar=nothing)

	m = MatrixTypes(initialise_matrices(nn, weights, synapses)...)
	p = get_parameters()
	da = 0

	run_all_steps(nn, numsteps, m, p, sensory, da, savevars=savevars, reportvar=reportvar, update=false)
	return (m.weights, m.synapses)
end

end
