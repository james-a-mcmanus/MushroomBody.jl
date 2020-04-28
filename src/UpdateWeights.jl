function update_weights!(t, l, m, p, da)

	tconst, A₋, t₋, miniw, δt = get_parameters(update_weights!, p, l)
	w, γ, connections, tpre, tpost = get_matrices(update_weights!, m, l)

	update_γ!(γ, connections, t, tpre, tpost, tconst, A₋, t₋, δt) # update tag

	w .= @. w + (γ * da) * δt

	w[w .< miniw] .= miniw
end

function update_γ!(γ, connections, t, tpre, tpost, tconst, A₋, t₋, δt)

#=	latency = (tpre .- tpost') .* connections

	δγ = @. (-γ / tconst) + stdp(latency, A₋, t₋) * Δ( (t - tpre) * (t - tpost') )

	γ .= γ .+ δγ .* δt=#

	γ .= @. γ + δt * ( (-γ / tconst) + stdp(((tpre - tpost') * connections), A₋, t₋) * Δ( (t - tpre) * (t - tpost') ) )

end


function stdp(latency, A₋, t₋)

	(latency .!= 0) .* A₋ .* exp.(latency ./ t₋) # punishes neurnos where pre fired after post
end
