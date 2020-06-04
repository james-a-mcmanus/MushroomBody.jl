# plotting functions
using Plots, Debugger, Images, Colors
gr()

struct GifPlot
	anim::Animation
	fname::String
end
GifPlot(fname::String) = GifPlot(Animation(), fname)

"""
initialises a plot
"""
function init_plot()
	p = plot(framestyle=:none)
end

"""
Plots the neurons and their static connections
"""
function networkplot(p, nn, connections::SynapseLayers)

	ycoords, xcoords = neuronplot!(p, nn)
	for i in 1:length(nn)-1
		y = (ycoords[i], ycoords[i+1])
		x = (xcoords[i], xcoords[i+1])
		plotconnections!(p, y, x, connections[i])
	end

	return p
end
function networkplot(p, nn, connections::SynapseLayers, weights::SynapseLayers; maxneurons=50)

	nn = min.(maxneurons, nn)
	ycoords, xcoords = neuronplot!(p, nn)

	for i in 1:length(nn)-1
		y = (ycoords[i], ycoords[i+1])
		x = (xcoords[i], xcoords[i+1])

		viewcons = connections.layers[i].data[1:nn[i],1:nn[i+1]]
		viewweights = weights.layers[i].data[1:nn[i],1:nn[i+1]]

		plotconnections!(p, y, x, viewcons, viewweights)
	end

	return p	
end
function networkplot(p, nn, m::MatrixTypes; maxneurons=50)

	nn = min.(maxneurons, nn)
	ycoords, xcoords = neuronplot!(p, nn, m, maxneurons)
	for i in 1:length(nn)-1
		y = (ycoords[i], ycoords[i+1])
		x = (xcoords[i], xcoords[i+1])

		viewweights = m.weights.layers[i].data[1:nn[i],1:nn[i+1]] .> 0
		viewspiked = m.spiked.layers[i].data[1:nn[i]] .* viewweights
		plotconnections!(p, y, x, viewweights, viewspiked)
	end

	return p
end


"""
checks the time and shows the plot if appropriate
"""
function shownetwork(p,t, nn, m::MatrixTypes; framerate=1)
	if t%framerate==0
		display(networkplot(p, nn, m))
	end
end
function gifnetwork(gf::GifPlot, p, t, nn, m::MatrixTypes)
	networkplot(p,nn,m)
	frame(gf.anim)
	return gf
end


"""
This returns a plot of circles marking location of neurons.
"""
function neuronplot!(p, nn)


	numlayers = length(nn)
	maxn = maximum(nn)

	xs = fill.([1:numlayers;], nn)
	xcoords = reduce(vcat, xs)

	f(x) = x==1 ? [round(maxn/2)] : [range(1,stop=maxn, length=x);]

	ys = map(x -> f(x),nn)
	ycoords = reduce(vcat, map(x -> f(x),nn))
	markersizes = reduce(vcat, fill.(neuronmarkersize(nn), nn))
	
	maxmkr = maximum(markersizes)
	maxy = [-maxmkr/5, maxn + maxmkr/5]
	maxx = [1-maxmkr/50, numlayers + maxmkr/50]
	scatter!(p, xcoords, ycoords, markercolor=:white, markersize=markersizes, legend=false, ylims=maxy, xlims=maxx)
	return (ys, xs)
end

function neuronplot!(p, nn, m::MatrixTypes, maxneurons)

	
	
	numlayers = length(nn)
	maxn = maximum(nn)

	xs = fill.([1:numlayers;], nn)
	xcoords = reduce(vcat, xs)
	
	mcolours = RGB.(vcat(m.input.layers[1][1:maxneurons] .> 0, falses(length(xs[2])), m.spiked.layers[3][1:min(length(m.spiked.layers[3]),maxneurons)]))
	
	f(x) = x==1 ? [round(maxn/2)] : [range(1,stop=maxn, length=x);]

	ys = map(x -> f(x),nn)
	ycoords = reduce(vcat, map(x -> f(x),nn))
	markersizes = reduce(vcat, fill.(neuronmarkersize(nn), nn))
	
	maxmkr = maximum(markersizes)
	maxy = [-maxmkr/5, maxn + maxmkr/5]
	maxx = [1-maxmkr/50, numlayers + maxmkr/50]	

	scatter!(p, xcoords, ycoords, markercolor=mcolours, markersize=markersizes, legend=false, ylims=maxy, xlims=maxx)
	return(ys, xs)
	# now we need to get just the inputted ones and colour them in white.
end


"""
Plot a line for each synapse, and flash if spiked
"""
function plotconnections!(p, ycoords, xcoords, connections)

	allconnections = vcat.(ycoords[1], ycoords[2]')
	allxs = vcat.(xcoords[1], xcoords[2]')

	linexs = allxs[connections]
	lineys = allconnections[connections]

	for i in 1:length(linexs)
		plot!(p,linexs[i],lineys[i],legend=false, linecolor=RGB(0.5,0.5,0.5), linealpha=0.9)
	end
end
function plotconnections!(p, ycoords, xcoords, weights::BitArray{<:Any}, spikers::BitArray{<:Any})

	allconnections = vcat.(ycoords[1], ycoords[2]')
	allxs = vcat.(xcoords[1], xcoords[2]')

	spikexs = allxs[spikers]
	spikeys= allconnections[spikers]

	linexs = allxs[weights]
	lineys = allconnections[weights]
	linewidths = weights[weights]

	for i in 1:length(linexs)
		iszero(linewidths[i]) && continue
		plot!(p,linexs[i],lineys[i], legend=false, linecolor=RGB(0.5,0.5,0.5), linealpha=0.9, linewidth=linewidths[i])
	end


	plot!(p, spikexs, spikeys, legend=false, linecolor=RGB(0.9,0.1,0.1))
end
function plotconnections!(p, ycoords, xcoords, connections::BitArray{<:Any}, weights::BitArray{<:Any}, spikers::BitArray{<:Any})

	
	allconnections = vcat.(ycoords[1], ycoords[2]')
	allxs = vcat.(xcoords[1], xcoords[2]')

	spikexs = allxs[spikers]
	spikeys= allconnections[spikers]

	linexs = allxs[connections]
	lineys = allconnections[connections]
	linewidths = weights[connections]

	for i in 1:length(linexs)
		iszero(linewidths[i]) && continue
		plot!(p,linexs[i],lineys[i], legend=false, linecolor=RGB(0.5,0.5,0.5), linealpha=0.9, linewidth=linewidths[i])
	end

	for i in 1:length(spikexs)
		iszero(spikexs[i]) && continue
		plot!(p, spikexs[i], spikeys[i], legend=false, linecolor=RGB(0.9,0.1,0.1))
	end
end
plotconnections!(p, ycoords, xcoords, connections::SynapseLayer, weights::SynapseLayer) = plotconnections!(p, ycoords, xcoords, connections.data .==1, weights.data)
plotconnections!(p, ycoords, xcoords, weights::Array{<:Number,<:Any}, spikers::Array{<:Number,<:Any}) = plotconnections!(p, ycoords, xcoords, weights .> 0.0, spikers)
plotconnections!(p, ycoords, xcoords, connections::Array{<:Number,<:Any}, weights::Array{<:Number,<:Any}, spiked::Array{<:Number, <:Any}) = plotconnections!(p, ycoords, xcoords, connections .==1, weights, spiked .== 1)
plotconnections!(p, ycoords, xcoords, connections::SynapseLayer) = plotconnections!(p, ycoords, xcoords,connections.data .== 1)

"""
for axis with limits 1-n, what is the ideal marker size for n neurons to all show up
"""
neuronmarkersize(numneurons) = 150 ./ (numneurons .+ 2)


"""
given a spikeplot, or some other neuronlayers object, plots a mean value of all neurons in each layer for each layer
"""
function plotmeanlayer(data::Array{<:BrainTypes,N}) where {N}
	p = plot()
	plotmeanlayer!(p,data)
	return p
end
function plotmeanlayer!(p, data::Array{<:BrainTypes,N}) where {N}

	numlayers = length(data[1].layers)
	layercolors = range(HSL(0,0,0), HSL(1,1,1), length=numlayers)
	means = meanlayer(data)
	plot!(p,means)
	return p
end
function meanlayer(data::Array{<:BrainTypes,N}) where {N}

	numsteps = length(data)
	numlayers = length(data[1].layers)
	out = Array{Float64,2}(undef, numsteps,numlayers)
	
	for i = 1:numsteps
		for l = 1:numlayers
			out[i,l] = mean(data[i].layers[l])
		end
	end
	return out
end


function display_sensory(p, sensory::AbstractInput)
	heatmap!(p, reshape(sensory,length(sensory),1), color=:grays, axis=nothing)
end

function display_sensory(sensory::InputSequence; fname="sensory_sequence.gif")
	
	p = heatmap(axis=nothing, color=:grays)
	gf = GifPlot(pwd() * "\\output\\figures\\" * fname)
	for s in sensory
		display_sensory(p, s)
		frame(gf.anim)
	end
	gif(gf.anim,gf.fname,fps=2)
end

function bar(ydata::Vector{normdata})

	means = [dp.mean for dp in ydata]
	std = [dp.std for dp in ydata]
	Plots.bar(means, yerror=std)
end

function bar!(ydata::Vector{normdata})

	means = [dp.mean for dp in ydata]
	std = [dp.std for dp in ydata]
	Plots.bar!(means, yerror=std)
end

function scatter(ydata::Array{normdata})

	means = [dp.mean for dp in ydata]
	std = [dp.std for dp in ydata]
	Plots.scatter(means[1], means[2], means[3])

end



