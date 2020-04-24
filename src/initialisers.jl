# this page is going to hold all of the setting-up functions 
# we will need to know how many neurons are in each layer and how many connections there are between each layer
using Random, BenchmarkTools
import Base.size, Base.setindex!, Base.getindex, Base.zero

#-------------------------------------------------------------------------------------------------------#
#											Type Definitions
#-------------------------------------------------------------------------------------------------------#

struct NeuronLayer{T,N} <: AbstractArray{T,N}
	data::Array{T,1}
	dims::NTuple{1,Int}
end

struct NeuronLayers{T,N}
	layers::Array{NeuronLayer{T,N},1}
end

struct SynapseLayer{T,N} <: AbstractArray{T,N}
	data::Array{T,N}
	dims::NTuple{N,Int}
#	connections::Array{Int,N} # or bool?
end

struct SynapseLayers{T,N} 
	layers::Array{SynapseLayer{T,N},1}
end
#-------------------------------------------------------------------------------------------------------#				
#-------------------------------------------------------------------------------------------------------#

function create_synapses(::Type{SynapseLayer}, lyrsize::NTuple{2,Int}; syndens=0.1, weight=2)

	syns = SynapseLayer(zeros(lyrsize...))
	ns = Int(round(syndens*lyrsize[1]))
	cons = Array{Int,2}(undef,ns,lyrsize[2])

	for i = 1:lyrsize[2]
		cons[:,i] .= shuffle(1:lyrsize[1])[1:ns]
	end

	for postlyr = 1:lyrsize[2]
		syns[cons[:,postlyr],postlyr] .= weight
	end

	return syns, cons
end

function create_synapses(::Type{SynapseLayers}, nn::Array; syndens=0.1, weight=2)

	lyrsizes = [(nn[i],nn[i+1]) for i = 1:length(nn)-1]
	SynapseLayers([create_synapses(SynapseLayer, lrs) for lrs in lyrsizes])
end

function fill_synapses(::Type{SynapseLayers}, nn::Array, filler::T) where T

	lyrsizes = [(nn[i],nn[i+1]) for i = 1:length(nn)-1]
	SynapseLayers([SynapseLayer(filler, lrs) for lrs in lyrsizes])
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


Base.size(A::Union{NeuronLayer,SynapseLayer}) = A.dims
Base.getindex(A::NeuronLayer{T,N}, Ind::Int) where {T,N} = get(A.data, Ind, nothing)
Base.setindex!(A::NeuronLayer, filler, Ind::Int) where {T,N} = (A.data[Ind] = filler)
Base.zero(::Type{NeuronLayer{T,N}}) where {T<:Number,N} = NeuronLayer{T,N}([zero(T)],(1,))
 NeuronLayer(::Type{T}, len::Integer) where {T} = NeuronLayer{T,1}(Array{T,1}(undef, len), (len,))
NeuronLayer(filler::T, len::Integer) where T = NeuronLayer{T,1}(fill(filler,len), (len,))

#=Base.getindex(A::SynapseLayer{T,N}, Ind::Vararg{Int,N}) where {T,N} = get(A.data, Ind, nothing)
Base.getindex(A::SynapseLayer{T,N}, Ind) where {T,N} = a.data[Ind]

Base.setindex!(A::SynapseLayer{T,N}, filler, Ind::Int) where {T,N} = A.data[Ind] = filler
Base.setindex!(A::SynapseLayer{T,N}, filler, Ind::Vararg{Int,N}) where {T,N} = Base.setindex!(A, filler, Ind)
Base.setindex!(A::SynapseLayer{T,N}, filler, Ind::NTuple{N,Int}) where {T,N} = A.data[Ind...] = filler=#


Base.zero(::Type{SynapseLayer{T,N}}) where {T<:Number, N} = SynapseLayer{T,N}(zeros(Int,ntuple(x->1,N)),ntuple(x->1,N))
Base.getindex(A::SynapseLayer{T,N}, I::Vararg{Int,N}) where {T,N} = A.data[I...]
Base.setindex!(A::SynapseLayer{T,N}, filler, I::Vararg{Int,N}) where {T,N} = A.data[I...] = filler


SynapseLayer{T,N}(A::Array{T,N}) where {T,N} = SynapseLayer{T,N}(A,size(A))
SynapseLayer(A::Array{T,N}) where {T,N} = SynapseLayer{T,N}(A)
SynapseLayer(filler::T, dims::NTuple{N,Int}) where {T,N} = SynapseLayer{T,N}(fill(filler, dims))
