# plotting functions
using Plots, Debugger
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

	# hmm this would require mutable struct... maybe not so bad
	plt.p1 = plotact!(plt.p1, activation)
	plt.p2 = plotspt!(plt.p2, spt, t)
	plt.p3 = plotda!(plt.p3, [da], [t])
	plt.p4 = plotrec!(plt.p4, rec)
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
