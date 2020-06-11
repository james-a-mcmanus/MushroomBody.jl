Base.size(A::StandardArray) = size(A.data)
Base.getindex(A::StandardArray, I::Int) = A.data[I]
Base.getindex(A::StandardArray{T,N}, I::Vararg{Int,N}) where {T,N} = A.data[I...]
Base.setindex(A::StandardArray, filler, I::Int) = A.data[I] = filler
Base.setindex(A::StandardArray{T,N}, filler, I::Vararg{Int,N}) where {T,N} = A.data[I...] = filler


"""
Input Types, they are an array to be inputed, and hold info about how long to last etc.
"""


"""
Input type constructors
"""
AbstractInput(construct::Type{T}, I::Vararg) where {T<:AbstractInput} = construct(I...)
AbstractInput(construct::Type{T}, A::Array; stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0]) where {T<:AbstractInput} = construct(A, stages=stages, input_bool=input_bool, reward_bool=reward_bool, punishment_bool=punishment_bool)
AbstractInput(construct::Type{T}, nn::Tuple; stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0]) where {T<:AbstractInput} = construct(nn, stages=stages, input_bool=input_bool, reward_bool=reward_bool, punishment_bool=punishment_bool)

SparseInput(A::Array; stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0]) = SparseInput(A, stages, input_bool, reward_bool=reward_bool, punishment_bool=punishment_bool)
SparseInput(A::Array, stages::Vector{Int}, input_bool::Vector{Bool}, reward_bool::Vector{Bool}, punishment_bool::Vector{Bool}) = SparseInput(A, stages, input_bool, reward_bool, punishment_bool, cumsum(stages))
function SparseInput(arraysize::Tuple; density=0.1, filler=300, stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0]) # actual density is density^n. n=ndims

	init = fill(0.0, arraysize)
	fillsize = round.(Int, arraysize .* density)
	init[map(x -> 1:x, fillsize)...] .= filler * ones(fillsize)
	shuffle!(init)
	return AbstractInput(SparseInput, init, stages=stages, input_bool=input_bool, reward_bool=reward_bool, punishment_bool=punishment_bool)
end

SparseRandInput(A::Array; stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0]) = SparseRandInput(A, stages, input_bool, reward_bool, punishment_bool)
SparseRandInput(A::Array, stages::Vector{Int}, input_bool::Vector{Bool}, reward_bool::Vector{Bool}, punishment_bool::Vector{Bool}) = SparseRandInput(A, stages, input_bool, reward_bool, punishment_bool, cumsum(stages))
function SparseRandInput(arraysize::Tuple; density=0.1, mean_filler=300, filler_range=50, stages=[0,1,1], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0])

	out = fill(0.0, arraysize)
	fillsize = round.(Int, arraysize .* density)
	out[map(x -> 1:x, fillsize)...] .= mean_filler .* ones(fillsize) .+ filler_range .* randn(fillsize)
	shuffle!(out)
	return AbstractInput(SparseRandInput, out, stages=stages, input_bool=input_bool, reward_bool=reward_bool, punishment_bool=punishment_bool)
end

RestInput(A::Array; stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0]) = RestInput(A, stages, input_bool, reward_bool, punishment_bool)
RestInput(A::Array, stages::Vector{Int}, input_bool::Vector{Bool}, reward_bool::Vector{Bool}, punishment_bool::Vector{Bool}) = RestInput(A, stages, input_bool, reward_bool, punishment_bool, cumsum(stages))
RestInput(nn::Tuple; stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,0,0], punishment_bool=Bool[0,0,0]) = RestInput(zeros(nn), stages, input_bool, reward_bool, punishment_bool) #always no DA


"""
Input Sequence
"""
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
duration(input::AbstractInput) = sum(input.stages)
duration(inseq::InputSequence) = sum(inseq.inputdurations)
inputdurations(inseq::InputSequence) = [duration(it) for it in inseq] #? does this need to be here??
inputdurations(inseq::Vector{Inputs}) = [duration(it) for it in inseq]
time_index(inseq::InputSequence, t) = findfirst(cumsum(inseq.inputdurations) .>= t)

function constructinputsequence(nn, typelist; stages=[0,1,0],input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0])
	
	out = Vector{Inputs}(undef, length(typelist))
	typlen = length(typelist)

	for i in 1:typlen
		out[i] = AbstractInput(typelist[i], nn, stages=stages, input_bool=input_bool, reward_bool=reward_bool, punishment_bool=punishment_bool)
	end

	default = RestInput(nn)
	return InputSequence(out, default)
end

emptyinputsequence() = InputSequence(Vector{Inputs}(undef,0))


"""
returns the index (or stage) that the current time corresponds to
"""
function get_input(inseq::InputSequence, t) 

	if t > duration(inseq) || t < 1
		error("no input available for that time")
	end

	nowind = time_index(inseq, t)
	tᵢ = nowind == 1 ? t : t - cumsum(inseq.inputdurations)[nowind-1]

	stage = get_stage(inseq[nowind], tᵢ)
	return inseq[nowind], inseq[nowind].input_bool[stage], inseq[nowind].reward_bool[stage], inseq[nowind].punishment_bool[stage]
end

function get_input(sensory::AbstractInput, t)
	stage = get_stage(sensory, t)
	return sensory, sensory.input_bool[stage], sensory.reward_bool[stage], sensory.punishment_bool[stage]
end


get_stage(input::AbstractInput, t) = findfirst(input.cum_stages .>= t)



"""
wrapper for sensory and reward functions
"""
function inputandreward!(t, input, sensory, τ; da=0)

	input .= input.inarrayseq[t]==1 ? input.inarray[1] : input*1
	BA = input.BAseq[t]
	da = update_da(da, BA, τ)

	return(BA,da)	
end

function inputandreward!(t, input, sensory, τ, da_on, da)

	current_sensory, give_input, give_reward, give_punishment = get_input(sensory,t)
	input .= give_input ? current_sensory.data : sensory.default.data
	reward_BA = give_reward ? da_on : 0.0
	punishment_BA = give_punishment ? da_on : 0.0	
	da.reward = update_da(da.reward, reward_BA, τ)
	da.punishment = update_da(da.punishment, punishment_BA, τ)

	return da
end

#input_or_default(sensory, give_input)

# prime exameple of where over-defininng input types is actually a hindrence.
#=function inputandreward!(t, input, sensory::AbstractInput, τ, da_on; da=0)
	give_input, give_da = get_input(sensory, t)
	input .= give_input ? sensory.data : sensory.default.data
	BA = give_da ? da_on : 0
	da = update_da(da, BA, τ)

	return(BA,da)
end
=#