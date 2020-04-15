module MushroomBody

export run_model, create_synapses


include("UpdateWeights.jl")
include("initialisers.jl")
include("Parameters.jl")
include("Rewards.jl")
include("UpdateActivation.jl")


function run_model()


	# create layers
	activation = create_neurons(nn,-60.0)
	rec = create_neurons(nn,0.0)
	spiked = create_neurons(nn,false)
	spt = create_neurons(nn,-1)
	I = create_neurons(nn,250)
	I[1] = I[1] .+ generate_noise(50,nn[1])
	(synapses, weights) = create_synapses(nn)
	γ, ACh = create_synapses(nn,ns=0, weight=0.0)
	output, = create_synapses(nn, ns=0, weight=0.0)

	for t = 1:10000

		for l = 1:2#length(nn)			

			update_activation!(nn[l], activation[l], vr[l], spiked[l], spt[l], t, vt[l], rec[l], I[l], C[l], a[l], b[l], c[l], d[l], k[l])

			output[l] .= weights[l] .* ACh[l] .* (rev[l] .- activation[l])

			#da = update_da(da, BA, τ, δt)

			da = 1

			if l != length(nn)

				update_weights!(weights[l], γ[l], synapses[l], t, spt[l], spt[l+1], da, tconst[l]; δt=1, A₋=-1, t₋=15)				

			end

		end
	end

end

end # module
