# this page is going to hold all of the setting-up functions 
# we will need to know how many neurons are in each layer and how many connections there are between each layer
using Random, BenchmarkTools
import Base.size, Base.setindex!, Base.getindex, Base.zero, Base.show, Base.length

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
end

struct SynapseLayers{T,N} 
	layers::Array{SynapseLayer{T,N},1}
end

abstract type AbstractSparseConnection{T,Int,N} <: AbstractSparseArray{T,Int,N}
end

abstract type AbstractDenseConnection{T,N} <: AbstractArray{T,N}
end

struct SparseConnection{T,N} <: AbstractSparseConnection{T,Int,N}
	data::AbstractSparseArray{T,Int,N}
	dims::NTuple{N,Int}
end

struct DenseConnection{T,N} <: AbstractDenseConnection{T,N}
	data::Array{T,N}
	dims::NTuple{N,Int}
end

Connection = Union{SparseConnection,DenseConnection}
NormProduct = Union{Number, AbstractSparseArray}
ConnectionLayer = Union{NeuronLayer, SynapseLayer}
BrainTypes = Union{NeuronLayers, SynapseLayers}


#-------------------------------------------------------------------------------------------------------#
#										Base Redefinitions
#-------------------------------------------------------------------------------------------------------#


#				Connection
#===========================================#
Base.size(A::Connection) = A.dims

Base.getindex(A::Connection, I::Any) = A.data[I]
Base.getindex(A::Connection, I::Vararg{Any,N}) where {N} = A.data[I...]
Base.setindex!(A::Connection, filler, I::Any) = A.data[I...] .= filler
Base.setindex!(A::Connection, filler, I::Vararg{Any,N}) where {N} = A.data[I...] .= filler
Base.setindex!(A::Connection, filler, I::Vararg{Int,N}) where {N} = A.data[I...] .= filler
Base.display(A::Connection) = display(A.data)

Base.:(*)(A::T, B::NormProduct) where {T<:Connection} = T(A.data .* B)
Base.:(-)(A::T, B::NormProduct) where {T<:Connection} = T(A.data .- B)
Base.:(+)(A::T, B::NormProduct) where {T<:Connection} = T(A.data .+ B)
Base.:(/)(A::T, B::NormProduct) where {T<:Connection} = T(A.data ./ B)

Base.:(*)(A::T, B::T) where {T<:Connection} = T(A.data .* B.data)
Base.:(-)(A::T, B::T) where {T<:Connection} = T(A.data .- B.data)
Base.:(+)(A::T, B::T) where {T<:Connection} = T(A.data .+ B.data)
Base.:(/)(A::T, B::T) where {T<:Connection} = T(A.data ./ B.data)


#				SparseConnection
#===========================================#
SparseConnection(A::AbstractSparseArray{T,N}) where {T,N<:Int} = SparseConnection(A, size(A))
SparseConnection(A::SparseConnection{T,N}) where {T,N} = SparseConnection(A.data, size(A))


#		NeuronLayer & SynapseLayer
#===========================================#
Base.size(A::ConnectionLayer) = A.dims
Base.getindex(A::NeuronLayer{T,N}, Ind::Int) where {T,N} = get(A.data, Ind, nothing)
Base.setindex!(A::NeuronLayer, filler, Ind::Int) where {T,N} = (A.data[Ind] = filler)
Base.zero(::Type{NeuronLayer{T,N}}) where {T<:Number,N} = NeuronLayer{T,N}([zero(T)],(1,))
 NeuronLayer(::Type{T}, len::Integer) where {T} = NeuronLayer{T,1}(Array{T,1}(undef, len), (len,))
NeuronLayer(filler::T, len::Integer) where T = NeuronLayer{T,1}(fill(filler,len), (len,))
Base.copy(A::ConnectionLayer) = typeof(A)(copy(A.data),A.dims)
Base.copy(A::BrainTypes) = typeof(A)([copy(l) for l in A.layers])
average(A::BrainTypes) = mean([average(lay) for lay in A.layers])
average(A::ConnectionLayer) = sum(A.data) == 0 ? 0 : sum(A.data)/length(A.data)
average(A::Array{<:BrainTypes,1}) = mean([average(elem) for elem in A])

Base.zero(::Type{SynapseLayer{T,N}}) where {T<:Number, N} = SynapseLayer{T,N}(zeros(Int,ntuple(x->1,N)),ntuple(x->1,N))
Base.getindex(A::SynapseLayer{T,N}, I::Vararg{Int,N}) where {T,N} = A.data[I...]
Base.setindex!(A::SynapseLayer{T,N}, filler, I::Vararg{Int,N}) where {T,N} = A.data[I...] = filler
Base.length(bt::BrainTypes) = length(bt.layers)

SynapseLayer{T,N}(A::Array{T,N}) where {T,N} = SynapseLayer{T,N}(A,size(A))
SynapseLayer(A::Array{T,N}) where {T,N} = SynapseLayer{T,N}(A)
SynapseLayer(filler::T, dims::NTuple{N,Int}) where {T,N} = SynapseLayer{T,N}(fill(filler, dims))

#-------------------------------------------------------------------------------------------------------#
#										Initialising Functions
#-------------------------------------------------------------------------------------------------------#
function create_synapses(::Type{SynapseLayer}, lyrsize::NTuple{2,Int}; syndens::Float64=0.1, weight=20.0)

	syndens==1 && return SynapseLayer(ones(lyrsize...))
	syndens==0 && return SynapseLayer(zeros(lyrsize...))
	ns = Int(round(syndens*lyrsize[1]))
	return create_synapses(SynapseLayer, lyrsize, syndens=ns, weight=weight)
end

function create_synapses(::Type{SynapseLayer}, lyrsize::NTuple{2,Int}; syndens::Int=10, weight=20.0)

	syns = SynapseLayer(zeros(lyrsize...))
	cons = Array{Int,2}(undef,syndens,lyrsize[2])
	for i = 1:lyrsize[2]
		cons[:,i] .= shuffle(1:lyrsize[1])[1:syndens]
	end

	for postlyr = 1:lyrsize[2]
		syns[cons[:,postlyr],postlyr] .= weight
	end
	return syns
end

Iterable = Union{Array, Tuple}

function create_synapses(::Type{SynapseLayers}, nn::Array; syndens=0.1, weight=2)

	lyrsizes = [(nn[i],nn[i+1]) for i = 1:length(nn)-1]
	SynapseLayers([create_synapses(SynapseLayer, lrs) for lrs in lyrsizes])
end

function create_synapses(::Type{SynapseLayers}, nn::Iterable, syndens::Iterable, weight::Iterable)
	lyrsizes = [(nn[i],nn[i+1]) for i = 1:length(nn)-1]
	out = Array{SynapseLayer,1}(undef,length(nn)-1)
	for (i, ls) in enumerate(lyrsizes)
		out[i] = create_synapses(SynapseLayer, ls, syndens=syndens[i], weight=weight[i])
	end
	return SynapseLayers{Float64,2}(out)
end

function clone_synapses(SLS::SynapseLayers; filler=1)

	SynapseLayers([SynapseLayer((l.data .> 0) .* filler) for l in SLS.layers])
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

#-------------------------------------------------------------------------------------------------------#