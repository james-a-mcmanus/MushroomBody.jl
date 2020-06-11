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
