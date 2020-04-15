# plotting functions
using GR

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