module MushroomBody

export 
run_model, 
gif_model,
response_before_after_learning,
response_before_after_learning2,
plotmeanlayer,
meanlayer,
plot_before_after,
doboth

using SparseArrays, Infiltrator, ColorCoding, Random, StaticArrays, Plots
import Base: size, sum, show, getindex, setindex!, getproperty, setproperty!, zero, length, one, fill!
using Random, BenchmarkTools


include("Types.jl")
include("UpdateWeights.jl")
include("UpdateActivation.jl")
include("initialisers.jl")
include("Neurotransmitters.jl")
include("helpers.jl")
include("input.jl")
include("Parameters.jl")
include("Tests2.jl")
include("plotters.jl")
include("SaveData.jl")
include("ImageInput.jl")


"""
Run All the Steps
"""
function run_all_steps(nn, numsteps, m, p, sensory, da; update=true, savevars=nothing, showplot=false, reportvar=nothing, normweights=false)

	returnvariable = !isnothing(reportvar) && initialise_return_variable(numsteps,m,reportvar)

	for t = 1:numsteps
		
		da = run_step(t, m, p, sensory, da, nn, update=update, normweights=normweights)
		
		showplot && shownetwork(init_plot(), t, nn, m)
		
		!isnothing(savevars) && save_variables(m,savevars)
		
		if !isnothing(reportvar)
			returnvariable[t] = return_variable(m, reportvar)
		end	
	end
	return returnvariable
end
function run_all_steps(nn, numsteps, m, p, sensory, da, gf::GifPlot; update=true, normweights=false)
	for t = 1:numsteps
		da = run_step(t, m, p, sensory, da, nn, update=update, normweights=normweights)
		gf = gifnetwork(gf, init_plot(), t, nn, m)
	end
	return gf
end

"""
run single step
"""
function run_step(t, m, p, sensory, da, nn; update=true, normweights=false)
	da = inputandreward!(t, m.input.layers[1], sensory, p.τ[1], p.da_on, da)
	for layer = 1:length(nn)
		run_layer(t, layer, nn, m, p, da, update=update)
	end 
	normweights && normalise_layer!(m, p, l=(length(nn)-1)) # normalise the last layer.	
	return da
end
"""
run Single layer
"""
function run_layer(t, layer, nn, m, p, da; update=true)
	update_activation!(t, layer, nn, m, p)
	if layer !== length(nn)
		update_pre_layers(t,layer,m,p, da, update=update)
	end	
end
"""
weight-changing and output functions for all but last layers.
"""
function update_pre_layers(t, layer, m, p, da; update=true)
	update_ACh!(t, layer, m, p)
	calc_input!(layer, m, p)
	(layer == 2 && update) && update_weights!(t, layer, m, p, da)	
end

"""
reset the dynamic aspects of the model, keeping the structur (weights and synapses) intact.
"""
function reset(nn, p, weights, synapses)

	return MatrixTypes(initialise_matrices(nn, p, weights, synapses)...) 
end


function reset!(m, p)

	fill!(m.activation, p.vr)
	fill!(m.rec, 0.0)
	fill!(m.spiked, false)
	fill!(m.spt, Int(-1))
	fill!(m.I, 0.0)
	fill!(m.ACh, 0.0)
	fill!(m.input, 0.0)
	fill!(m.γ, 0.0)

end
	
end
