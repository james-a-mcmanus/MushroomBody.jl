function update_weights!(t, l, m, p, da)

	tconst, A₋, t₋, min_w, δt = get_parameters(update_weights!, p, l)
	w, γ, connections, tpre, tpost = get_matrices(update_weights!, m, l)

	update_γ!(γ, connections, t, tpre, tpost, tconst, A₋, t₋, δt) # update tag

	update_w(connections, w, min_w, γ, da, δt)
end


function update_w(c, w, min_w, γ, da, δt)
	
	if sum(da) == 0
		return
	end
	
	w .= @. w + (γ * sum(da)) * δt
	w[w .< min_w] .= min_w	
end

function update_w(con::MBONLayer, w, min_w, γ, da, δt)

	for (i,c) in enumerate(con)
		update_w(c, w, min_w, γ .* c, da[i], δt)
	end

end

function update_γ!(γ, connections::MBONLayer, t, tpre, tpost, tconst, A₋, t₋, δt)
	for c in connections
		update_γ!(γ, c, t, tpre, tpost, tconst, A₋, t₋, δt)
	end
end

function update_γ!(γ, connections, t, tpre, tpost, tconst, A₋, t₋, δt)

	γ .= @. γ + δt * ( (-γ / tconst) + stdp(((tpre - tpost') * connections), A₋, t₋) * Δ( (t - tpre) * (t - tpost') ) )
end


function stdp(latency, A₋, t₋)

	(latency .!= 0) .* A₋ .* exp.(latency ./ t₋) # punishes neurnos where pre fired after post
end

function normalise_layer!(m, p; l=2)
	
	m.weights.layers[l] .= m.weights.layers[l] .* (p.weight_target[l] ./ sum(m.weights.layers[l]))
end