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
	nn = [100, 1000, 5]
	sensory = constructinputsequence((100,), (SparseInput,), stages=[10,100,1], input_bool=Bool[1,1,0], da_bool=Bool[0,1,1])
	numsteps = duration(sensory)
	println(numsteps)
	weights, synapses = train_model(sensory,nn,numsteps, showplot=false)
	test_model(sensory,nn,numsteps,weights, synapses)# calcs an sensory and then calls run_model(::Randsensory).
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
Within-Network Functions
"""
function run_all_steps(nn, numsteps, m, p, sensory, da; update=true, savevars=nothing, showplot=false, reportvar=nothing)
	returnvariable = !isnothing(reportvar) && initialise_return_variable(numsteps,m,reportvar)	
	for t = 1:numsteps
		da = run_step(t, m, p, sensory, da, nn, update=update)
		showplot && shownetwork(init_plot(), t, nn, m)
		!isnothing(savevars) && save_variables(m,savevars)
		if !isnothing(reportvar)

			returnvariable[t] = return_variable(m, reportvar)
		end
	end
	return returnvariable
end
function run_step(t, m, p, sensory, da, nn; update=true)
	BA, da = inputandreward!(t, m.input.layers[1], sensory, p.τ[1], p.da_on, da=da)
	for layer = 1:length(nn)
		run_layer(t,layer,nn,m,p,da,update=update)
	end	
	return da
end
function run_layer(t, layer, nn, m, p, da; update=true)
	update_activation!(t, layer, nn, m, p)
	if layer !== length(nn)
		update_pre_layers(t,layer,m,p,da, update=update)
	end	
end
function update_pre_layers(t, layer, m, p, da; update=true)
	update_ACh!(t, layer, m, p, da)
	calc_input!(layer, m, p)
	(layer == 2 && update) && update_weights!(t, layer, m, p, da)	
end

"""
train the model, returns weights and synapses
"""
function train_model(sensory, nn, numsteps; showplot=false, gifplot=false, update=true, savevars=nothing, reportvar=nothing)

	m = MatrixTypes(initialise_matrices(nn)...)
	p = get_parameters()
	da = 0
	reporter = run_all_steps(nn, numsteps, m, p, sensory, da, savevars=savevars, update=true, reportvar=reportvar)
	return (m.weights, m.synapses, reporter)
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
function test_model(sensory, nn, numsteps, weights, synapses; showplot=false, gifplot=false, update=false, savevars=nothing, reportvar=nothing)

	m = MatrixTypes(initialise_matrices(nn, weights, synapses)...)
	p = get_parameters()
	da = 0
	reporter = run_all_steps(nn, numsteps, m, p, sensory, da, savevars=savevars, update=false, reportvar=reportvar)
	return (m.weights, m.synapses, reporter)
end

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









#=function test_model(sensory, nn, numsteps, weights, synapses; showplot=false, savevars=nothing, returnvar=nothing)

	m = MatrixTypes(initialise_matrices(nn, weights, synapses)...)
	p = get_parameters()
	da = 0

	for t = 1:numsteps
	
		BA, da = inputandreward!(t, m.input.layers[1], sensory, p.τ[1], p.da_on, da=da)
		for layer = 1:length(nn)
			update_activation!(t, layer, nn, m, p) #these haven't been defined: maybe have another get function for these?
			if layer !== length(nn)
				update_ACh!(t, layer, m, p, da)
				calc_input!(layer, m, p)
			end
		end
		showplot && shownetwork(init_plot(), t, nn, m)
		!isnothing(savevars) && save_variables(m,savevars)
	end
	return (m.weights, m.synapses)
end=#

end
