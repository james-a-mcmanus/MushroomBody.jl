# this is going to hold the main updating functions e.g. upduate activation etc.
function update_activation!(nn, v, vr, sp, spt, t, vt, rec, I, C, a, b, c, d, k; δt=1, σ=0.05)

	ξ = generate_noise(σ,nn)
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
	#spt[sp] .= t
	v .= v .* (.! sp) .+ c .* sp # turn all into c if above resting
	rec .= rec .+ (.! sp) .* d
	return sp

end

#=
function calc_output!(output, w, ACh, rev, v)
# this is an (hopefully) optimised version of calc_output! in which everything except the voltage (and maybe the ACh?) is a sparse matrix



end=#

function calc_output!(output, w, ACh, rev, v)

	output .= dropdims(sum((rev .- v) .* w .* ACh, dims = 1),dims=1)
	# this v should be the v of the current neuron (?)

end