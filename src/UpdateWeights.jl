#=module UpdateWeights

export update_weights!, update_γ!, Δ, stdp=#

function update_weights!(w, γ, connections, t, tpre, tpost, da, tconst; δt=1, A₋=-1, t₋=15)

	@bp
	γ .= update_γ!(γ, connections, t, tpre, tpost, tconst, A₋, t₋, δt) # update tag

	δw = γ .* da # update weights based on da and tag

	w .= w .+ δw .* δt # also need to have a maximum and minimum weight
	w[w .< miniw] .= miniw

end

function update_γ!(γ, connections, t, tpre, tpost, tconst, A₋, t₋, δt)

	latency = (tpre .- tpost') .* connections

	δγ = (-γ ./ tconst) .+ stdp(latency, A₋, t₋) .* Δ( (t .- tpre) .* (t .- tpost') )

	γ .= γ .+ δγ .* δt
end


function stdp(latency, A₋, t₋)

	(latency .!= 0) .* A₋ .* exp.(latency ./ t₋) # punishes neurnos where pre fired after post

end
