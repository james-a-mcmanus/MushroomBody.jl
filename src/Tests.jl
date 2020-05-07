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

function test_learning()

	nn = SA[10, 100, 1]
	numsteps  = 10
	in1 = create_input(nn[1], 180:220, numsteps, numsteps, [50,50], 0.8, BAstart=10)
	run_model(in1)
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
	numtrain = 2
	numtest = 2

	facenum = tuple(nn[1])
	stimtype = (SparseInput,)
	sstages = [10, 100 , 10]
	input = Bool[1,1,0]
	dastages = Bool[0,1,0]
	reportvar = "spiked"

	m = MatrixTypes(initialise_matrices(nn)...)
	p = get_parameters()
	da = 0	

	trainsensory = [constructinputsequence(facenum, stimtype, stages=sstages, input_bool=input, da_bool=dastages) for tr in 1:numtrain]
	testsensory = [constructinputsequence(facenum, stimtype, stages=sstages, input_bool=input, da_bool=dastages) for te in 1:numtest]
	
	for tr in 1:numtrain
		numsteps = duration(trainsensory[tr])
		reporter = run_all_steps(nn, numsteps, m, p, trainsensory[tr], da, savevars=nothing, update=true, reportvar=reportvar)
		println(average_layers(reporter,3)) # wait we only want the 3rd layer...
	end

	for te in 1:numtest
		numsteps = duration(testsensory[te])
		reporter = run_all_steps(nn, numsteps, m, p, testsensory[te], da, savevars=nothing, update=false, reportvar=reportvar)
		println(average_layers(reporter,3))
	end

end

function average_layers(A::Array{<:BrainTypes,1},l)
	holder = Vector{Float64}(undef,length(A))
	for t in 1:length(A)
		holder[t] = mean(A[t].layers[l])
	end
	return mean(holder)
end
