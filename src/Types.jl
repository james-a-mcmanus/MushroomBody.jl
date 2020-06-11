#-------------------------------------------------------------------------------------------------------#
#											Input Types
#-------------------------------------------------------------------------------------------------------#
abstract type StandardArray{T,N} <: AbstractArray{T,N} end
abstract type AbstractInput{N} <: StandardArray{Float64,N} end
struct RandInput{N} <: AbstractInput{N}
	data::Array{<:Any,N}
	stages::Vector{Int}
	input_bool::Vector{Bool}
	reward_bool::Vector{Bool}
	punishment_bool::Vector{Bool}
	cum_stages::Vector{Int}	
end
struct SparseInput{N} <: AbstractInput{N}
	data::Array{<:Any,N}
	stages::Vector{Int}
	input_bool::Vector{Bool}
	reward_bool::Vector{Bool}
	punishment_bool::Vector{Bool}
	cum_stages::Vector{Int}
end
struct RestInput{N} <: AbstractInput{N}
	data::Array{<:Any,N}
	stages::Vector{Int}
	input_bool::Vector{Bool}
	reward_bool::Vector{Bool}
	punishment_bool::Vector{Bool}
	cum_stages::Vector{Int}	
end
struct SparseRandInput{N} <: AbstractInput{N}
	data::Array{<:Any,N}
	stages::Vector{Int}
	input_bool::Vector{Bool}
	reward_bool::Vector{Bool}
	punishment_bool::Vector{Bool}
	cum_stages::Vector{Int}		
end
struct ColorInput{N} <: AbstractInput{N}
	data::Array{<:Any,N}
	stages::Vector{Int}
	input_bool::Vector{Bool}
	reward_bool::Vector{Bool}
	punishment_bool::Vector{Bool}
	cum_stages::Vector{Int}
	image::Image
end

Inputs = Union{RandInput, SparseInput, RestInput, SparseRandInput, ColorInput}



#-------------------------------------------------------------------------------------------------------#
#											Network Types
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

struct MBONLayer{T,N} <: AbstractArray{T,N}
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
ConnectionLayer = Union{NeuronLayer, SynapseLayer, MBONLayer}

struct DiverseLayers{T}
	layers::Array{T,1}
end

Iterable = Union{Array, Tuple}
BrainTypes = Union{NeuronLayers, SynapseLayers, DiverseLayers}

mutable struct Dopamine <: AbstractArray{Float64, 1}
	reward::Float64
	punishment::Float64
end

size(da::Dopamine) = (2,)
function getindex(da::Dopamine, i::Int)
	if i == 1
		return da.reward
	elseif i == 2
		return da.punishment
	else 
		error("Tried to access a dopamine type with index != 1 or 2")
	end
end
function setindex!(da::Dopamine, f, i::Int)
	if i == 1
		da.reward = f
	elseif i == 2
		da.punishment = f
	else 
		error("Tried to access a dopamine type with index != 1 or 2")
	end
end
Dopamine(num::Number) = Dopamine(num,num)