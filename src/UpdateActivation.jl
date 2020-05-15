# this is going to hold the main updating functions e.g. upduate activation etc.
function update_activation!(t, layer, nn, matrices, parameters)

	vt, vr, C, a, b, c, d, k, σ, δt = get_parameters(update_activation!,parameters,layer)
	v, sp, spt, rec, I = get_matrices(update_activation!, matrices, layer)

	ξ = generate_noise(σ,nn[layer])
	update_voltage!(v, vr, vt, rec, I, ξ, C, k, δt)
	update_recovery!(v, vr, rec, a, b, δt)
	layer==2 && apl_inhibition2!(matrices, parameters)
	update_spikes!(v, sp, spt, t, vt, rec, c, d)
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

	a = m.activation.layers[2]

	to_ceil = bottomk(a, length(a) - 5)
	a[to_ceil] .= ceiling.(a[to_ceil],p.vt[2])

end

function ceiling(num, ce)

	num > ce ? ce : num
end

topk(a, k) = partialsortperm(a, 1:k, rev=true)
bottomk(a, k) = partialsortperm(a, 1:k)