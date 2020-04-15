# this page is going to hold all of the setting-up functions 
# we will need to know how many neurons are in each layer and how many connections there are between each layer
using Random

# Function for creating the weights and connections of a layer

function create_synapses(nn; ns=10, weight=2)

	connections = Vector{Array}(undef, length(nn)-1)
	weights = Vector{Array}(undef, length(nn)-1)

	for l = 1:length(nn)-1

		connections[l] = zeros(nn[l],nn[l+1])
		syns = ones(ns, nn[l+1])
		connections[l][1:ns, 1:nn[l+1]] .= syns


		connections[l] .= connections[l][shuffle(1:end),:]
		weights[l] = weight .* connections[l]

	end

	return (connections, weights)

end




# Function for filling an array full of a specific 
function create_neurons(nn, filler)

	out = []

	if length(filler) < length(nn)
		filler = fill(filler,length(nn))
	end

	for layer = 1:length(nn)
		
		push!(out, fill(filler[layer], nn[layer]))
	
	end
	return out

end