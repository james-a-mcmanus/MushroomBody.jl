# this is going to hold the main updating functions e.g. upduate activation etc.
function update_activation!(t, layer, nn, matrices, parameters)

	vt, vr, C, a, b, c, d, k, σ, δt = get_parameters(update_activation!,parameters,layer)
	v, sp, spt, rec, I = get_matrices(update_activation!, matrices, layer)

	ξ = generate_noise(σ,nn[layer])	

	update_voltage!(v, vr, vt, rec, I, ξ, C, k, δt)
	update_recovery!(v, vr, rec, a, b, δt)
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
	
	l==2 && winner_takes_all(ACh,p.winners[l])
	input .= dropdims(sum( @.(w * ACh * (rev - v)'), dims=1), dims=1) #dropdims(sum(w .* ACh .* (rev .- v)', dims = 1), dims=1)
	
	l==1 && winner_takes_all(input, p.winners[l])
end

function divisive_normalise!(input, inmax)
	if sum(input) == 0
		return
	end
	input .= inmax * input ./ sqrt(sum(input .^ 2))
end

function subtractive_normalise(input, λ)

	input = - λ * sum(input) / length(input)
end

function winner_takes_all(input, numwinners)
	if length(input) == numwinners
		return
	end
	anti_assign(input, maxk(input,numwinners), 0.0)
end

function maxk!(ix, a, k; initialized=false)
    partialsortperm!(ix, a, 1:k, rev=true, initialized=initialized)
    return ix
end

function maxk(a,k)

	partialsortperm(a,1:k,rev=true)
end

function anti_assign(array, indices, assign_otherwise)

	for i in 1:length(array)
		array[i] = i in indices ? array[i] : assign_otherwise
	end
end
