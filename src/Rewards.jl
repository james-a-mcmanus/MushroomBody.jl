function update_da(da, BA, τ, δt)

	δda = (-d ./ τ) .+ BA

	da = δda .* δt

end