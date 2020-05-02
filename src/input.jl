using Random, StaticArrays, Infiltrator
import Base.size, Base.sum

abstract type StandardArray{T,N} <: AbstractArray{T,N} end
Base.size(A::StandardArray) = size(A.data)
Base.getindex(A::StandardArray, I::Int) = A.data[I]
Base.getindex(A::StandardArray{T,N}, I::Vararg{Int,N}) where {T,N} = A.data[I...]
Base.setindex(A::StandardArray, filler, I::Int) = A.data[I] = filler
Base.setindex(A::StandardArray{T,N}, filler, I::Vararg{Int,N}) where {T,N} = A.data[I...] = filler

struct DAInputTiming
	before::Int
	during::Int
	after::Int
end
DAInputTiming(A::Array) = DAInputTiming(A[1],A[2],A[3])
Base.sum(D::DAInputTiming) = D.before + D.during + D.after

"""
Input Types, they are an array to be inputed, and hold info about how long to last etc.
"""
abstract type AbstractInput{N} <: StandardArray{Float64,N} end
struct RandInput{N} <: AbstractInput{N}
	data::Array{<:Any,N}
	displaytime::Int
	datime::DAInputTiming
end
struct SparseInput{N} <: AbstractInput{N}
	data::Array{<:Any,N}
	displaytime::Int
	datime::DAInputTiming
end
struct RestInput{N} <: AbstractInput{N}
	data::Array{<:Any,N}
	displaytime::Number
	datime::DAInputTiming
end
Inputs = Union{RandInput, SparseInput, RestInput}


"""
Input type constructors
"""
AbstractInput(construct::Type{T}, I::Vararg) where {T<:AbstractInput} = construct(I...)
AbstractInput(construct::Type{T}, A::Array; it=1, dat=[0,1,0]) where {T<:AbstractInput} = construct(A, it=it, dat=dat)
AbstractInput(construct::Type{T}, nn::Tuple; it=1, dat=[0,1,0]) where {T<:AbstractInput} = construct(nn, it=it, dat=dat)

SparseInput(A::Array; it=1, dat=[0,1,0]) = SparseInput(A, it, DAInputTiming(dat))
RestInput(A::Array; it=1, dat=[0,0,0]) = RestInput(A,it,DAInputTiming(dat))

# nn constructors
RestInput(nn::Tuple; it=1, dat=[0,0,0]) = RestInput(zeros(nn), it=it, dat=dat)
function SparseInput(arraysize::Tuple; density=0.1, filler=250, it=1, dat=[0,1,0]) # actual density is density^n. n=ndims

	init = fill(0.0, arraysize)
	fillsize = round.(Int, arraysize .* density)
	init[map(x -> 1:x, fillsize)...] .= filler * ones(fillsize)
	shuffle!(init)
	return AbstractInput(SparseInput, init, it=it, dat=dat)
end



# Sequences
abstract type AbstractSequence{T} <: StandardArray{T,1} end
struct InputSequence <: AbstractSequence{Inputs} 
	data::Vector{Inputs}
	inputdurations::Vector{Int}
	default::RestInput
end
InputSequence(inputarray::Vector{Inputs}, default::RestInput) = InputSequence(inputarray, inputdurations(inputarray), default)

"""
functions for input types
"""
duration(In::AbstractInput) = In.datime.before + In.displaytime + In.datime.after
duration(Inseq::InputSequence) = sum([duration(it) for it in Inseq])
inputdurations(Inseq::InputSequence) = [duration(it) for it in Inseq]
inputdurations(inseq::Vector{Inputs}) = [duration(it) for it in inseq]
durations(input::AbstractInput) = [input.datime.before, input.displaytime, input.datime.after]
durations(da::DAInputTiming) = [da.before, da.during, da.after]

function constructinputsequence(nn, typelist; it=1, dat=[0,1,0])
	
	out = Vector{Inputs}(undef, length(typelist))
	typlen = length(typelist)

	for i in 1:typlen
		out[i] = AbstractInput(typelist[i], nn, it=it, dat=dat)
	end

	default = RestInput(nn)
	return InputSequence(out, default)
end

emptyinputsequence() = InputSequence(Vector{Inputs}(undef,0))


time_index(inseq::InputSequence, t) = findlast(cumsum(inseq.inputdurations) .<= t)
time_index(input::AbstractInput, t) = findfirst(cumsum(durations(input)) .>= t)
input_index(::AbstractInput, i) = i==2
da_index(::DAInputTiming, i) = i==2
time_index(da::DAInputTiming, t) = findfirst(cumsum(durations(da)) .>= t)

function get_input(inseq::InputSequence, t) 

	if t > duration(inseq) || t < 1
		error("no input available for that time")
	end

	nowind = time_index(inseq, t)
	tᵢ = 1 + t - cumsum(inseq.inputdurations)[nowind]
	current_input = inseq[nowind]
	
	give_input, give_da = get_input(current_input, tᵢ)

	input = give_input ? current_input.data : inseq.default.data

	return (input, give_da)
end

function get_input(input::AbstractInput, t)

	if t > duration(input) || t < 1
		error("given t is not in input")
	end

	give_input = input_index(input, time_index(input,t))
	give_da = da_index(input.datime, time_index(input.datime,t))

	return give_input, give_da

end




# add a struct to the inputsequence that is basically the empty matrix that gets given when 




















#=
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
end=#

"""
take the input object, the current time, and return the input array
"""


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