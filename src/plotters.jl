# plotting functions
using Plots, Debugger, Images
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
function networkplot(p, nn, connections::SynapseLayers, weights::SynapseLayers; maxneurons=20)

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
function networkplot(p, nn, weights::SynapseLayers, spiked::NeuronLayers; maxneurons=20)

	nn = min.(maxneurons, nn)
	ycoords, xcoords = neuronplot!(p, nn)

	for i in 1:length(nn)-1
		y = (ycoords[i], ycoords[i+1])
		x = (xcoords[i], xcoords[i+1])

		viewweights = weights.layers[i].data[1:nn[i],1:nn[i+1]] .> 0
		viewspiked = spiked.layers[i].data[1:nn[i]] .* viewweights
		plotconnections!(p, y, x, viewweights, viewspiked)
	end

	return p
end


"""
checks the time and shows the plot if appropriate
"""
function shownetwork(p,t, nn, m::MatrixTypes; framerate=1)
	if t%framerate==0
		display(networkplot(p, nn, m.weights, m.spiked))
	end
end
function gifnetwork(gf::GifPlot, p, t, nn, m::MatrixTypes)
	networkplot(p,nn,m.weights, m.spiked)
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

	ys = map(x -> [range(1,stop=maxn,length=x);],nn)
	ycoords = reduce(vcat, map(x -> [range(1,stop=maxn,length=x);],nn))
	markersizes = reduce(vcat, fill.(neuronmarkersize(nn), nn))
	
	maxmkr = maximum(markersizes)
	maxy = [-maxmkr/5, maxn + maxmkr/5]
	maxx = [1-maxmkr/50, numlayers + maxmkr/50]

	scatter!(p, xcoords, ycoords, markersize=markersizes, legend=false, ylims=maxy, xlims=maxx)
	return (ys, xs)
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
