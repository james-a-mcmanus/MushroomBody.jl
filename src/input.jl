using Random
import Base.size

abstract type StandardArray{T,N} <: AbstractArray{T,N} end
Base.size(A::StandardArray) = size(A.data)
Base.getindex(A::StandardArray, I::Int) = A.data[I]
Base.getindex(A::StandardArray{T,N}, I::Vararg{Int,N}) where {T,N} = A.data[I...]
Base.setindex(A::StandardArray, filler, I::Int) = A.data[I] = filler
Base.setindex(A::StandardArray{T,N}, filler, I::Vararg{Int,N}) where {T,N} = A.data[I...] = filler

abstract type AbstractInput{N} <: StandardArray{Float64,N} end
struct RandInput{N} <: AbstractInput{N}

	data::Array{<:Any,N}
end
struct SparseInput{N} <: AbstractInput{N}

	data::Array{<:Any,N}
end
SparseInput(nn::Tuple) = SparseInput(nn, 0.1, 250)
SparseInput(nn::Tuple, filler) = SparseInput(nn, 0.1, filler)
function SparseInput(arraysize::Tuple, density, filler) # actual density is density^n. n=ndims

	init = fill(0.0, arraysize)
	fillsize = round.(Int, arraysize .* density)
	init[map(x -> 1:x, fillsize)...] .= filler * ones(fillsize)
	shuffle!(init)
	return SparseInput(init)
end

abstract type AbstractSequence{T} <: StandardArray{T,1} end
struct InputSequence{T} <: AbstractSequence{T}

	data::Vector{T}
end
struct NTSequence{T} <: AbstractSequence{T} 

	data::Vector{T}
end
function NTSequence(InSeq::InputSequence)
end


abstract type AbstractNeuroTransmitter <: AbstractFloat end
struct Dopamine <: AbstractNeuroTransmitter
	
	data::Float64
end
struct AcetylCholine <: AbstractNeuroTransmitter

	data::Float64
end

"""
take the input object, the current time, and return the input array
"""
function get_input(t, in::AbstractInput) # fallback
	in.data
end

function get_input(t, in::SparseInput)
end
















function get_input(nn, rnge::Union{Array,UnitRange}; howmany=1)

	rewinput = [rand(rnge, nn) for _ = 1:length(howmany)]

	return rewinput
end

function input_timings(ns::Int, on_off, rewtime)

	ontime = Int(round( ( on_off[1] / sum(on_off) ) * ns ))
	BAtime =  Int(round( rewtime * ontime ))

	return(BAtime, ontime)
end

function input_sequence(totalst, inst, on_off, rewtime; instart::Int=1, BAstart::Int=1)

	(BAt, ot) = input_timings(inst, on_off, rewtime)

	inputseq = zeros(totalst)
	BAseq = zeros(totalst)
	inputseq[instart:instart+ot] .= 1
	BAseq[BAstart:BAstart+BAt] .= 1

	return(inputseq, BAseq)
end

function create_input(nn, rnge, nstp, instp, on_off, rewtime; BAstart=1)

	inarray = get_input(nn, rnge)
	inseq, BAseq = input_sequence(nstp, instp, on_off, rewtime)
	return(RandInput(inarray,inseq,BAseq))
end