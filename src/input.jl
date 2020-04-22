struct RandInput
	inarray::Array
	inarrayseq::Array # basically for what portion of that input being shown should we actually release the dop
	BAseq::Array
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