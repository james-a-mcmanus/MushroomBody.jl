# this is going to hold the main updating functions e.g. upduate activation etc.
function update_activation!(t, layer, nn, matrices, parameters)

	vt, vr, C, a, b, c, d, k, σ, δt = get_parameters(update_activation!,parameters,layer)
	v, sp, spt, rec, I = get_matrices(update_activation!, matrices, layer)

	ξ = generate_noise(σ,nn[layer])	

	update_voltage!(v, vr, vt, rec, I, ξ, C, k, δt)
	
	update_recovery!(v, vr, rec, a, b, δt)
	update_spikes!(v, sp, spt, t, vt, rec, c, d)
	layer==2 && apl_inhibition2!(matrices, parameters)
end	


function generate_noise(σ,nn)

	(rand(nn) .- 0.5) .* σ
end

function update_voltage!(v, vr, vt, rec, I, ξ, C, k, δt)

	δv = (k .* (v .- vr) .* (v .- vt) .- rec .+ I .+ ξ) ./ C

	v .= v .+ δv .* δt
end

function update_recovery!(v, vr, rec, a, b, δt)

	δrec = a .* (b .* (v ./ vr) .- rec)

	rec .= rec + δrec .* δt
end

function update_spikes!(v, sp, spt, t, vt, rec, c, d)

	# check spikes
	sp .= v .> vt
	spt[sp] .= t
	v .= v .* (.! sp) .+ c .* sp # turn all into c if above resting
	rec .= rec .+ (.! sp) .* d
end

function calc_input!(l, m, p; rev=0)

	input, w, ACh, v = get_matrices(calc_input!, m, l)

	input .= dropdims(sum( @.(w * ACh * (rev - v)'), dims=1), dims=1) #dropdims(sum(w .* ACh .* (rev .- v)', dims = 1), dims=1)
end


function apl_inhibition!(m, p)
	# i'm thinking we'll calculate this based on the ACh output of the KC layer.
	# don't exactly know what the function should be i've not modelled any inhibitory input.
	# could base it off some chlorine current (i think it's GABA?)
	# could just subtract it from the input for next timestep
	m.activation.layers[2] .= calc_inhibit(m.activation.layers[2], p.vt[2], 0.001)
end
function calc_inhibit(activations, vt, modif)
	divisor = sum(activations[activations .> vt] .- vt)
	divisor==0 && return activations
	activations .- (modif .* divisor)
end

function apl_inhibition2!(m,p)

	spike_ind = m.activation.layers[2] .> p.vt[2]
	bottom_n = bottomk(m.activation.layers[2][spike_ind], sum(spike_ind) - 5)


	#bottom_spiked = m.activation.layers[2][spike_ind][bottom_n]
#=
	if !isempty(bottom_spiked)
		@infiltrate
	end

	#const_punish.(bottom_spiked, 20)
	#bottom_spiked .-= 40000000
	bottom_spiked .-= 50000000000*sum(bottom_spiked) =#

end


const_punish(a, c) = a = a - c
population_punish(a, c, pop) = a = a - c*pop
sum_activity_punish(a, c, act) = a = a - c*act

topk(a, k) = partialsortperm(a, 1:k, rev=true)
bottomk(a, k) = partialsortperm(a, 1:k)