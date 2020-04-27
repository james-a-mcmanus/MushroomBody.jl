function update_da(da, BA, τ; δt=1)

	δda = (-da ./ τ) .+ BA

	da = da + δda .* δt
end

function update_ACh!(t, l, m, params, da)

	ACh, tpre = get_matrices(update_ACh!, m, l)
	Φ, synt, δt = get_parameters(update_ACh!, params, l)

	δACh = -ACh ./ synt .+ Φ .* Δ(t .- tpre)

	ACh .= ACh .+ δACh .* δt
end