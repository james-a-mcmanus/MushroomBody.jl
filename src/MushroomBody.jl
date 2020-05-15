module MushroomBody

export 
run_model, 
gif_model,
response_before_after_learning,
response_before_after_learning2,
plotmeanlayer,
meanlayer,
plot_before_after,
doboth

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
	nn = [100, 1000, 1]
	sensory = constructinputsequence((100,), (SparseRandInput,), stages=[10,100,1], input_bool=Bool[1,1,0], da_bool=Bool[0,1,1])
	numsteps = duration(sensory)
	println(numsteps)
	weights, synapses = train_model(sensory,nn,numsteps, showplot=true)
	test_model(sensory,nn,numsteps,weights, synapses)# calcs an sensory and then calls run_model(::Randsensory).
end

"""
save a run of the model as a gif
"""
function gif_model(nn, sensory; outname="test.gif")
	
	outfolder = "C:\\Users\\James\\.julia\\dev\\MushroomBody\\src\\output\\figures\\"
	gf = GifPlot(outfolder * outname)
	
	numsteps = duration(sensory)

	(weights, synapses, gf) = train_model(sensory, nn, numsteps, gf)
	(weights, synapses, gf) = test_model(sensory, nn, numsteps, weights, synapses, gf)
	gif(gf.anim,gf.fname,fps=15)
end

"""
train the model, returns weights and synapses
"""
function train_model(sensory, nn, numsteps; showplot=false, gifplot=false, update=true, savevars=nothing, reportvar=nothing)
	
	p = get_parameters()
	m = MatrixTypes(initialise_matrices(nn, p)...)
	
	da = 0
	reporter = run_all_steps(nn, numsteps, m, p, sensory, da, savevars=savevars, update=true, reportvar=reportvar, showplot=showplot)
	return (m.weights, m.synapses, reporter)
end
function train_model(sensory, nn, numsteps, gf::GifPlot; update=true, normweights=false)
	
	p = get_parameters()
	m = MatrixTypes(initialise_matrices(nn, p)...)
	da = 0

	gf = run_all_steps(nn, numsteps, m, p, sensory, da, gf, update=update, normweights=normweights)	
	return (m.weights, m.synapses, gf)
end

"""
Test the model, takes weights
"""
function test_model(sensory, nn, numsteps, weights, synapses; showplot=false, update=false, savevars=nothing, reportvar=nothing)
	
	p = get_parameters()
	m = MatrixTypes(initialise_matrices(nn, p, weights, synapses)...)
	
	da = 0
	reporter = run_all_steps(nn, numsteps, m, p, sensory, da, savevars=savevars, update=false, reportvar=reportvar)
	return (m.weights, m.synapses, reporter)
end
function test_model(sensory, nn, numsteps, weights, synapses, gf::GifPlot; update=false, normweights=false)
	
	p = get_parameters()
	m = MatrixTypes(initialise_matrices(nn, p, weights, synapses)...)
	
	da = 0

	gf = run_all_steps(nn, numsteps, m, p, sensory, da, gf, update=update, normweights=normweights)	
	return (m.weights, m.synapses, gf)
end

"""
Run All the Steps
"""
function run_all_steps(nn, numsteps, m, p, sensory, da; update=true, savevars=nothing, showplot=false, reportvar=nothing, normweights=false)

	returnvariable = !isnothing(reportvar) && initialise_return_variable(numsteps,m,reportvar)

	for t = 1:numsteps
		
		da = run_step(t, m, p, sensory, da, nn, update=update, normweights=normweights)
		
		showplot && shownetwork(init_plot(), t, nn, m)
		
		!isnothing(savevars) && save_variables(m,savevars)
		
		if !isnothing(reportvar)
			returnvariable[t] = return_variable(m, reportvar)
		end	
	end
	return returnvariable
end
function run_all_steps(nn, numsteps, m, p, sensory, da, gf::GifPlot; update=true, normweights=false)
	for t = 1:numsteps
		da = run_step(t, m, p, sensory, da, nn, update=update, normweights=normweights)
		gf = gifnetwork(gf, init_plot(), t, nn, m)
	end
	return gf
end

"""
run single step
"""
function run_step(t, m, p, sensory, da, nn; update=true, normweights=false)
	BA, da = inputandreward!(t, m.input.layers[1], sensory, p.τ[1], p.da_on, da=da)
	for layer = 1:length(nn)
		run_layer(t,layer,nn,m,p,da,update=update)
	end 
	normweights && normalise_layer!(m, p, l=(length(nn)-1)) # normalise the last layer.	
	return da
end
"""
run Single layer
"""
function run_layer(t, layer, nn, m, p, da; update=true)
	update_activation!(t, layer, nn, m, p)
	if layer !== length(nn)
		update_pre_layers(t,layer,m,p,da, update=update)
	end	
end
"""
weight-changing and output functions for all but last layers.
"""
function update_pre_layers(t, layer, m, p, da; update=true)
	update_ACh!(t, layer, m, p, da)
	calc_input!(layer, m, p)
	(layer == 2 && update) && update_weights!(t, layer, m, p, da)	
end


end
