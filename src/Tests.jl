using Statistics

function test_ach()

	@eval using Plots
	gr()

	nn = 1
	ts = 10
	ACh = fill(0.0, nn)
	synt = 3 # or 8
	Φ = 0.5
	spikesequence = repeat([1, 0, 0, 0, 0], outer=2)
	@show spikesequence
	spt = sim_spikes(nn, ts, spikesequence)
	p = scatter()

	for t = 1:ts

		update_ACh!(ACh, synt, Φ, t, spt[t])
		scatter!(p, [t], [ACh[1]])

	end

	display(p)
end

function sim_spikes(nn, ts, spikesequence)


	lastspike = 1
	out = []

	for i = 1 : ts

		if spikesequence[i] == 1
			push!(out, i)
			lastspike = i
		else
			push!(out, lastspike)
		end

	end
	return out
end

function test_da()

	@eval using Plots
	gr()

	ts = 40
	τ = 20
	da = 0
	BA = repeat([0.5, 0.0, 0.0, 0.0],inner=8, outer=5)
	p = scatter()

	for t = 1:ts
		da = update_da(da, BA[t], τ; δt=1)
		@show da, BA[t]
		scatter!(p, [t], [da], show=true, legend=false)
	end
end

function test_transmission()

	@eval using Plots
	gr()

	# get2 layers, with one synapse between them, and see how activity in the 1st translates to activity in the second.
	nn = [1, 1]
	ts = 60
	ACh = [0.0]
	w = 4.0
	spt1 = sim_spikes(1, 60, repeat([1, 1, 0],inner=20))
	spt2 = 0
	synt = 3 # or 8
	Φ = 0.5
	v = [-60.0]
	rec = [0.0]
	sp = [false]
	C = 100
	a = 0.3
	b = -0.3
	c = -65
	d = 8
	k = 2

	p = scatter()

	for t = 1:ts

		update_ACh!(ACh, synt, Φ, t, spt1[t])
		# calc output.
		input = (-v) .* w .* ACh
		update_activation!(1, v, -60, sp, spt2, t, -40, rec, input, C, a, b, c, d, k;)
		scatter!(p,[t], v, legend=false, color=RGB(0.5, 0.5, 1))

	end

	display(p)
end

function test_weight()

	@eval using Plots
	gr()

	# get2 layers, with one synapse between them, and see how activity in the 1st translates to activity in the second.
	nn = [1, 1]
	ts = 60
	ACh = hcat(0.0)
	w = hcat(4.0)
	γ = hcat(0.0)
	connected = [1]
	spt1 = sim_spikes(1, 60, repeat([1, 1, 0],inner=20))
	spt2 = hcat(0)
	synt = 3 # or 8
	Φ = 0.5
	v = hcat(-60.0)
	rec = hcat(0.0)
	sp = hcat(false)
	C = 100
	a = 0.3
	b = -0.3
	c = -65
	d = 8
	k = 2

	p = scatter()

	for t = 1:ts
		update_ACh!(ACh, synt, Φ, t, spt1[t])
		input = (-v) .* w .* ACh
		update_activation!(1, v, -60, sp, spt2, t, -40, rec, input, C, a, b, c, d, k;)
		update_weights!(w, γ, connected, t, spt1[t], spt2, 1, 40; δt=1, A₋=-1, t₋=15)

		scatter!(p, [t], w, legend=false, color=RGB(0.5, 0.5, 1))
	end
	display(p)
end

function response_before_after_learning2()

	nn = [100, 1000, 5]
	sensory = constructinputsequence((100,), (SparseInput,), stages=[10,100,1], input_bool=Bool[1,1,0], da_bool=Bool[0,1,1])
	numsteps = duration(sensory)
	weights, synapses, trainspikes = train_model(sensory, nn, numsteps, reportvar="spiked")	
	_, _, testspikes = test_model(sensory, nn, numsteps, weights, synapses, reportvar="spiked")
	train_plot = plotmeanlayer(trainspikes)
	test_plot = plotmeanlayer(testspikes)
	plot(train_plot,test_plot, layout=(1,2))
end

function many_stimuli()

	nn = [100, 1000, 5]
	numtrain = 4
	numtest = 4

	facenum = tuple(nn[1])
	stimtype = (SparseInput,)
	sstages = [10, 100 , 10]
	input = Bool[1,1,0]
	dastages = Bool[0,1,0]
	reportvar = "spiked"

	p = get_parameters()
	m = MatrixTypes(initialise_matrices(nn, p)...)
	da = 0	

	sensory = [constructinputsequence(facenum, stimtype, stages=sstages, input_bool=input, da_bool=dastages) for te in 1:(numtrain + numtest)]

	# Training Period	
	for tr in 1:numtrain
		numsteps = duration(sensory[tr])
		reporter = run_all_steps(nn, numsteps, m, p, sensory[tr], da, savevars=nothing, update=true, reportvar=reportvar)
		m = m = reset(nn, p, m.weights, m.synapses)
	end

	# 
	for te in 1:(numtrain + numtest)
		numsteps = duration(sensory[te])
		m = reset(nn, p, m.weights, m.synapses)
		reporter = run_all_steps(nn, numsteps, m, p, sensory[te], da, savevars=nothing, update=false, reportvar=reportvar)
		println(average_layers(reporter,3))
	end
end

function many_observations()
	nn = [100, 100, 1]
	kcn = [100, 1000, 10_000]
	p = []
	for kc in kcn
		nn[2] = kc
		println(nn)
		push!(p,observe_variable("spiked",nn=nn))
	end
	plot(p[1],p[2],p[3])
end

function observe_variable(reportvar::String; nn=[100,1000,1])
	numstim = 1

	facenum = tuple(nn[1])
	stimtype = (SparseInput,)
	sstages = [10, 500 , 10]
	input = Bool[1,1,0]
	dastages = Bool[0,1,0]
	
	p = get_parameters()
	m = MatrixTypes(initialise_matrices(nn,p)...)
	da = 0	

	sensory = [constructinputsequence(facenum, stimtype, stages=sstages, input_bool=input, da_bool=dastages) for te in 1:numstim]
	for i in 1:numstim
		numsteps = duration(sensory[i])
		normalise_layer!(m, p, l=2)		
		reporter = run_all_steps(nn, numsteps, m, p, sensory[i], da, savevars=nothing, update=false, reportvar=reportvar)
		p = plot(average_timestep(reporter))
	end
	return p
end

function observe_variable(reportvar::String; nn=[100,1000,1], layer=1)

	numstim = 1

	facenum = tuple(nn[1])
	stimtype = (SparseInput,)
	sstages = [10, 500 , 10]
	input = Bool[1,1,0]
	dastages = Bool[0,1,0]
	
	p = get_parameters()
	m = MatrixTypes(initialise_matrices(nn,p)...)
	da = 0	

	sensory = [constructinputsequence(facenum, stimtype, stages=sstages, input_bool=input, da_bool=dastages) for te in 1:numstim]
	for i in 1:numstim
		numsteps = duration(sensory[i])
		normalise_layer!(m, p, l=2)		
		reporter = run_all_steps(nn, numsteps, m, p, sensory[i], da, savevars=nothing, update=false, reportvar=reportvar)
		p = plot(average_timestep(reporter,l=layer))
	end
	return p
end

function average_layers(A::Array{<:BrainTypes,1},l)
	holder = Vector{Float64}(undef,length(A))
	for t in 1:length(A)
		holder[t] = mean(A[t].layers[l])
	end
	return mean(holder)
end

function average_timestep(A::Array{<:BrainTypes,1};l=nothing)
	numsteps = length(A)
	numlayers = isnothing(l) ? length(A[1].layers) : l
	out = zeros(numsteps, numlayers)
	for s = 1:numsteps
		for l = 1:numlayers
			out[s,l] = mean(A[s].layers[l])
		end
	end
	return out
end

# this is actually just a less efficient version of initialise_matrices(_,_,weights,synapses)
function reset(nn, p, weights, synapses)

	return MatrixTypes(initialise_matrices(nn, p, weights, synapses)...) 
end

function test_many_kenyon_weights()

	totalweights = range(300,500,length=3)
	p = get_parameters()
	
	numtest=30
	numtrain=30
	nn = [100, 1000, 1]

	for kw in totalweights

		p.weight_target = (kw, kw, kw)
		MBON = test_parameter(p, nn, numtest=numtest, numtrain=numtrain)
		print("\nTrain Average: $(mean(MBON[1:numtest])), Test Average: $(mean(MBON[numtest+1:end]))")

	end
end

function test_parameter(p, nn; numtrain=4, numtest=4, layer_view=3)


	facenum = tuple(nn[1])
	stimtype = (SparseInput,)
	sstages = [10, 100 , 10]
	input = Bool[1,1,0]
	dastages = Bool[0,1,0]
	reportvar = "spiked"
	da=0

	m = MatrixTypes(initialise_matrices(nn, p)...)
	sensory = [constructinputsequence(facenum, stimtype, stages=sstages, input_bool=input, da_bool=dastages) for te in 1:(numtrain + numtest)]

	# Training Period	
	for tr in 1:numtrain		

		m = reset(nn, p, m.weights, m.synapses)
		normalise_layer!(m, p, l=2)
		numsteps = duration(sensory[tr])
		reporter = run_all_steps(nn, numsteps, m, p, sensory[tr], da, savevars=nothing, update=true, normweights=false, reportvar=reportvar)
	end

	avactivation = zeros(numtrain+numtest)
	
	for te in 1:(numtrain + numtest)
		numsteps = duration(sensory[te])
		m = reset(nn, p, m.weights, m.synapses)
		reporter = run_all_steps(nn, numsteps, m, p, sensory[te], da, savevars=nothing, update=false, reportvar=reportvar)
		avactivation[te] = average_layers(reporter,layer_view)
	end
	return avactivation
end

function test_learning()

	p = get_parameters()
	
	numtest=16
	numtrain=16
	nn = [100, 10000, 1]
	numreps = 2
	train_mean = zeros(numreps)
	test_mean = zeros(numreps)
	train_std = zeros(numreps)
	test_std = zeros(numreps)

	for i in 1:numreps
		println(i)
		MBON = test_parameter(p, nn, numtest=numtest, numtrain=numtrain, layer_view=3)
		train_mean[i] = mean(MBON[1:numtrain])
		test_mean[i] = mean(MBON[numtrain+1:end])
		train_std[i] = std(MBON[1:numtrain])
		test_std[i]	= std(MBON[numtrain+1:end])
	end

	bar([1,2], [mean(train_mean), mean(test_mean)], grid=false, yerror=[mean(train_std), mean(test_std)], ylim=[0,1], marker = stroke(2, RGB(0.8,0.1,0.1)))
end

function kcactivation()

	p = get_parameters()
	numtest=4
	nn=[100,1000,1]
	kcnum = [100,1000,4000]
	kc_activation = zeros(length(kcnum))
	for (i, kc) in enumerate(kcnum)
		nn[2] = kc
		kc_activation[i] = mean(test_parameter(p, nn, numtest=numtest, numtrain=0, layer_view=2))
	end
	plot(kcnum, kc_activation)
end

function compare(m1::MatrixTypes, m2::MatrixTypes, nn)
	numlayers = length(m1.spiked.layers)
	for fnm in fieldnames(MatrixTypes)
		println(String(fnm))
		for l = 1:numlayers
			println(all(getfield(m1.layers[l], fnm) .== getfield(m2.layers[l], fnm))) # won't work for 2d.
		end
	end
end