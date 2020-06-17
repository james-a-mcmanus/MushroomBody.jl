using Statistics, Dates

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
	p = Plots.scatter()

	for t = 1:ts
		da = update_da(da, BA[t], τ; δt=1)
		@show da, BA[t]
		Plots.scatter!(p, [t], [da], show=true, legend=false)
	end
end

function test_transmission()


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

	p = Plots.scatter()

	for t = 1:ts

		update_ACh!(ACh, synt, Φ, t, spt1[t])
		# calc output.
		input = (-v) .* w .* ACh
		update_activation!(1, v, -60, sp, spt2, t, -40, rec, input, C, a, b, c, d, k;)
		Plots.scatter!(p,[t], v, legend=false, color=RGB(0.5, 0.5, 1))

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

const stimtype = (ColorInput,)
const sstages = [20, 5, 50]
const inputstages = Bool[1,1,1]
const dastages = Bool[0,1,0]
const reportvar = "spiked"
const init_da=0

struct normdata
	mean::Float64
	std::Float64
end
normdata(intuple::Tuple{Float64,Float64}) = normdata(intuple[1],intuple[2])
function normdata(norm::Vector{normdata})
	
	mean = 0
	std = 0


	for n in norm
		mean += n.mean
		std += n.std
	end

	return normdata(mean/length(norm),std/length(norm))
end
normdata(a::Array) = normdata(mean(a), std(a))




function setup()

	p = get_parameters()
	m = MatrixTypes(initialise_matrices(p.nn, p)...)
	return (p, m)
end

function setup(p::ParameterTypes)

	m = MatrixTypes(initialise_matrices(p.nn, p)...)
	return (p, m)
end

function initialise_reporter(sensory_input, m)
	[initialise_return_variable(duration(s), m, reportvar) for (i,s) in enumerate(sensory_input)]
end

function train_input!(p, m, sensory_input::AbstractInput)
	numsteps = duration(sensory_input)
	da = Dopamine(init_da)
	run_all_steps(p.nn, numsteps, m, p, sensory_input, da, savevars=nothing, update=true, normweights=true)
	reset!(m,p)
	return
end

InputArrays = Union{InputSequence, Vector{<:AbstractInput}}

function train_input!(p, m, sensory_input::InputArrays)

	for s in sensory_input
		train_input!(p, m, s)
	end
end

function test_input(p, m, sensory_input::AbstractInput)
	reset!(m,p)
	numsteps = duration(sensory_input)
	da = Dopamine(init_da)
	return run_all_steps(p.nn, numsteps, m, p, sensory_input, da, savevars=nothing, update=false, normweights=false, reportvar="spiked")
end

function test_input(p, m, sensory_input::InputArrays)
	report = initialise_reporter(sensory_input, m)
	for (i, s) in enumerate(sensory_input)
		report[i] = test_input(p, m, s)
	end
	return report
end







function spiked_statistics(reporter::Array{NeuronLayers{Bool,1}}; layer=3)

	meanspikes = zeros(length(reporter))
	for (i,timestep) in enumerate(reporter)
		meanspikes[i] = mean(timestep.layers[layer])
	end

	return normdata(mean(meanspikes), std(meanspikes))
end

function spiked_statistics(reporter::Array{Array{NeuronLayers{Bool,1},1},1}; layer=3)
	stats = Vector{normdata}(undef, length(reporter))
	for (i, epoch) in enumerate(reporter)
		stats[i] = normdata(spiked_statistics(epoch, layer=layer))
	end
	return stats
end

function kenyon_cell_representation(p, m, sensory_input::AbstractInput)
	spiked = test_input(p,m,sensory_input)
	representation = spike_representation(spiked, p.nn[2], layer=2)
end

function spike_representation(spiked, neurons_in_layer; layer=2)

	representation = zeros(neurons_in_layer)
	for timestep in spiked
		representation += timestep.layers[layer]
	end
	representation ./ length(spiked)
end

function spike_learning_measure(p,m, train, test)
    
    for sensory_train in train
        train_input!(p,m,sensory_train)
    end
    
    test_spikes = Vector{normdata}(undef, length(test))
    for (i,sensory_test) in enumerate(test)
        test_spikes[i] = spiked_statistics(test_input(p,m,sensory_test))
    end
    
    train_spikes = similar(test_spikes)
    for (i,sensory_train) in enumerate(train)
        train_spikes[i] = spiked_statistics(test_input(p,m,sensory_train))
    end    
    
    return (train_spikes, test_spikes)
end

function manhatten_distance(array1, array2)

	sum( abs.( array1 .- array2 ) )
end

function nearest_neighbour(array1, all_other_arrays)

	distances = zeros(length(all_other_arrays))

	for (i,a) in enumerate(all_other_arrays)

		distances[i] = manhatten_distance(array1, a) 
	
	end

	return findmin(distances)
end
