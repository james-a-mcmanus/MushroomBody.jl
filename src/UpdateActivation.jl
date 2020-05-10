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

#=
function calc_output!(output, w, ACh, rev, v)
# this is an (hopefully) optimised version of calc_output! in which everything except the voltage (and maybe the ACh?) is a sparse matrix



end=#

function calc_input!(l, m, p; rev=0)

	input, w, ACh, v = get_matrices(calc_input!, m, l)

	input .= dropdims(sum( @.(w * ACh * (rev - v)'), dims=1), dims=1) #dropdims(sum(w .* ACh .* (rev .- v)', dims = 1), dims=1)

end