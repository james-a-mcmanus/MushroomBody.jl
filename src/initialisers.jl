# this page is going to hold all of the setting-up functions 
# we will need to know how many neurons are in each layer and how many connections there are between each layer
using Random
import Base.size, Base.setindex!, Base.getindex

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

function create_synapses!(arraytofill, filler; ns=10)

	# if any(ns .> size(arraytofill[:](1)))

	# 	println("That matrix doesn't look very sparse!")

	# end

	for l =1:length(arraytofill)

		prenum, postnum = size(arraytofill[l])

		i = Array{Int,1}(undef, ns*postnum)
		
		for ii = 1:postnum
           i[1 + (ii-1) * ns: (ii-1)*ns + ns] .= shuffle(1:prenum)[1:ns]#vcat(out, shuffle(1:prenum)[1:ns])
       	end
		
		j = repeat(collect(1:postnum),inner=ns)
		
		v = repeat([filler],outer=ns*postnum)

		for ii =1:length(i)

			arraytofill[l][i[ii],j[ii]] = v[ii]

		end

	end
end

function fillentries!(sparray, filler)

	for l = 1:length(sparray)
	
		(i, j, v) = findnz(sparray[l])
		sparray[l] = sparse(i,j, v .* filler)

	end

	return sparray
end

function fillcells!(arraytofill, filler)


	for l = 1:length(arraytofill)

		arraytofill[l] .= filler

	end
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

struct NeuronLayer{T,1} <: AbstractArray{T,1}
	data::Array{T,1}
	dims::NTuple{1,Int}
end

Base.size(A::NeuronLayer) = A.dims
Base.getindex(A::NeuronLayer) = 
