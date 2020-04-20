module MushroomBody

export run_model, cuconnections, cusynapses, cuneurons, create_synapses!, update_weights! ,sparsedensemult, fillcells!, fillentries!, calc_output!,update_activation!, update_γ!, generate_noise, update_voltage!


using SparseArrays, Debugger, CuArrays

include("UpdateWeights.jl")
include("initialisers.jl")
include("Parameters.jl")
include("Neurotransmitters.jl")
include("UpdateActivation.jl")
include("helpers.jl")
#include("plotters.jl")


function run_model()

	# preallocating all the parameters
	nn = (100, 1000, 1)

	# neuron types
#=	activation = [Vector{Float64}(undef,i) for i in nn]
	rec = [Vector{Float64}(undef,i) for i in nn]
	spiked = [Vector{Bool}(undef,i) for i in nn]
	spt = [Vector{Int}(undef,i) for i in nn]
	I = [Vector{Float64}(undef,i) for i in nn]
	output = [Vector{Float64}(undef,i) for i in nn]=#
	activation = cuneurons(nn, -60f0)
	rec = cuneurons(nn, 0f0)
	spiked = cuneurons(nn, false)
	spt = cuneurons(nn, -1)
	I = cuneurons(nn, 250f0)
	output = cuneurons(nn, 0f0)

	synapses = cuconnections(nn, 1.0f0, 10)
	γ = cusynapses(nn, 0f0)
	ACh = cusynapses(nn, 0f0)
	weights = synapses .* 2f0

	#I[1] .= I[1] .+ generate_noise(50,nn[1])

	

	for t = 1:1000

		for l = 1:length(nn)

			update_activation!(nn[l], activation[l], vr[l], spiked[l], spt[l], t, vt[l], rec[l], output[l], C[l], a[l], b[l], c[l], d[l], k[l])
			da = 1
			#da = update_da!(da, BA, τ)

			if l != length(nn)

				update_ACh!(ACh[l], synt[l], Φ[l], t, spt[l])

				#calc_output!(output[l+1], weights[l], ACh[l], rev[l], activation[l])

				#update_weights!(weights[l], γ[l], synapses[l], t, spt[l], spt[l+1], da, tconst[l]; δt=1, A₋=-1, t₋=15)

			end


		end
	end

	return weights

end

end # module
