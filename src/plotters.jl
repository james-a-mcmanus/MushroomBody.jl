# plotting functions
using Plots, Debugger, Images
gr()

# this takes a load of spikes and plots them on an animated graph
function spiketrain(t, spikes)


	# get co-ordinates for the spikes
	spiked= findall(spikes)
	spcoords = (t .* ones(length(spiked)), spiked)
	sc = scatter!(spcoords,xlims=(0,1000),marker=1, legend=false)
	display(sc)
end

function spiketrain(t, matrix::Array{<:Any,2}, spikes::BitArray{1})

	matrix[:,t] = spikes
	imshow(matrix)
	return(matrix)
end

function visualiseweights(weights)

	GR.imshow(repeat(weights))
end


function dashboard(plt, t, weights, activation, spt, da, rec)

	plt.p1 = plotact!(plt.p1, activation)
	plt.p2 = plotspt!(plt.p2, spt, t)
	plt.p3 = plotda!(plt.p3, [da], [t])
	plt.p4 = plotrec!(plt.p4, rec)
	p = plot(plt.p1, plt.p2, plt.p3, plt.p4, layout = grid(2,2), legend=false, show=true)
	display(p)
end
function dashboard(plt, t, l, m::MatrixTypes, da)

	plt.p1 = plotact!(plt.p1, m.activation.layers[l])
	plt.p2 = plotspt!(plt.p2, m.spt.layers[l], t)
	plt.p3 = plotda!(plt.p3, [da], [t])
	plt.p4 = plotrec!(plt.p4, m.rec.layers[l])
	p = plot(plt.p1, plt.p2, plt.p3, plt.p4, layout = grid(2,2), legend=false, show=true)
	display(p)
end

mutable struct Dashplot
	p1::Plots.Plot{<:AbstractBackend}
	p2::Plots.Plot{<:AbstractBackend}
	p3::Plots.Plot{<:AbstractBackend}
	p4::Plots.Plot{<:AbstractBackend}
	Dashplot() = new(heatmap(),heatmap(), scatter(), heatmap())
end

function initialiseplot(plottype::Function)

	return plottype()
end

function plotact!(p1, activation)

	heatmap!(p1, reshape(activation, length(activation), 1))
end

function plotspt!(p2, spt, t)

	spiked = spt.==t
	heatmap!(p2, reshape(spiked, length(spiked),1))
end

function plotda!(p3, t, da)
	scatter!(p3, da, t)
end

function plotrec!(p4, rec)
	heatmap!(p4, reshape(rec,length(rec),1))
end


abstract type PlotType{T} <: AbstractPlot{T}
end# #where T is some backend.

struct SpikePlot{T} <: PlotType{T}
		plot::Plots.Plot{T}
end

SpikePlot(m::MatrixTypes) = SpikePlot(scatter())


"""

"""



"""
Plots the neurons and their static connections
"""

function networkplot(nn, connections::SynapseLayers)

	p = plot()
	ycoords, xcoords = neuronplot!(p, nn)
	for i in 1:length(nn)-1
		y = (ycoords[i], ycoords[i+1])
		x = (xcoords[i], xcoords[i+1])
		plotconnections!(p, y, x, connections[i])
	end

	return p
end

function networkplot(nn, connections::SynapseLayers, weights::SynapseLayers; maxneurons=50)

	p = plot(framestyle=:none)

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

function networkplot(nn, connections::SynapseLayers, weights::SynapseLayers, spiked::NeuronLayers; maxneurons=50)

	p = plot(framestyle=:none);

	nn = min.(maxneurons, nn)
	ycoords, xcoords = neuronplot!(p, nn)

	for i in 1:length(nn)-1
		y = (ycoords[i], ycoords[i+1])
		x = (xcoords[i], xcoords[i+1])

		viewcons = connections.layers[i].data[1:nn[i],1:nn[i+1]]
		viewweights = weights.layers[i].data[1:nn[i],1:nn[i+1]]
		viewspiked = spiked.layers[i].data[1:nn[i]] .* viewcons

		plotconnections!(p, y, x, viewcons, viewweights, viewspiked)
	end

	return p
end

function shownetwork(t, nn, m::MatrixTypes; framerate=2)

	if t%framerate==0
		display(networkplot(nn, m.synapses, m.weights, m.spiked))
	end
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
	ylims = [-maxmkr/5, maxn + maxmkr/5]
	xlims = [1-maxmkr/50, numlayers + maxmkr/50]

	scatter!(p, xcoords, ycoords, markersize=markersizes, legend=false, ylim=ylims, xlim=xlims)
	return (ys, xs)
end


"""
Plot the ntework connections
"""
plotconnections!(p, ycoords, xcoords, connections::SynapseLayer) = plotconnections!(p, ycoords, xcoords,connections.data .== 1)

function plotconnections!(p, ycoords, xcoords, connections)

	allconnections = vcat.(ycoords[1], ycoords[2]')
	allxs = vcat.(xcoords[1], xcoords[2]')

	linexs = allxs[connections]
	lineys = allconnections[connections]

	for i in 1:length(linexs)
		plot!(p,linexs[i],lineys[i],legend=false, linecolor=RGB(0.5,0.5,0.5), linealpha=0.9)
	end
end

plotconnections!(p, ycoords, xcoords, connections::SynapseLayer, weights::SynapseLayer) = plotconnections!(p, ycoords, xcoords, connections.data .==1, weights.data)
plotconnections!(p, ycoords, xcoords, connections::Array{<:Number,<:Any}, weights::Array{<:Number,<:Any}) = plotconnections!(p, ycoords, xcoords, connections .==1, weights)
plotconnections!(p, ycoords, xcoords, connections::Array{<:Number,<:Any}, weights::Array{<:Number,<:Any}, spiked::Array{<:Number, <:Any}) = plotconnections!(p, ycoords, xcoords, connections .==1, weights, spiked .== 1)


function plotconnections!(p, ycoords, xcoords, connections, weights)

	allconnections = vcat.(ycoords[1], ycoords[2]')
	allxs = vcat.(xcoords[1], xcoords[2]')

	linexs = allxs[connections]
	lineys = allconnections[connections]
	linewidths = weights[connections]

	for i in 1:length(linexs)
		plot!(p,linexs[i],lineys[i],legend=false, linecolor=RGB(0.5,0.5,0.5), linealpha=0.9, linewidth=linewidths[i])
	end
end

function plotconnections!(p, ycoords, xcoords, connections, weights, spikers)

	
	allconnections = vcat.(ycoords[1], ycoords[2]')
	allxs = vcat.(xcoords[1], xcoords[2]')

	spikexs = allxs[spikers]
	spikeys= allconnections[spikers]

	linexs = allxs[connections]
	lineys = allconnections[connections]
	linewidths = weights[connections]

	for i in 1:length(linexs)
		plot!(p,linexs[i],lineys[i], legend=false, linecolor=RGB(0.5,0.5,0.5), linealpha=0.9, linewidth=linewidths[i])
	end
	
	plot!(p, spikexs, spikeys, legend=false, linecolor=RGB(0.9,0.1,0.1))

end


"""
for axis with limits 1-n, what is the ideal marker size for n neurons to all show up
"""
neuronmarkersize(numneurons) = 150 ./ (numneurons .+ 2)
