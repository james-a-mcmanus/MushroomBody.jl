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