struct ParameterTypes{N}
	nn::NTuple{N,Number}
	c::NTuple{N,Number}
	d::NTuple{N,Number}
	C::NTuple{N,Number}
	noisestd::NTuple{N,Number}
	vr::NTuple{N,Number}
	cap::NTuple{N,Number}
	a::NTuple{N,Number}
	b::NTuple{N,Number}
	k::NTuple{N,Number}
	vt::NTuple{N,Number}
	dvoltage::NTuple{N,Number}
	synt::NTuple{N,Number}
	quantile::NTuple{N,Number}
	t₋::NTuple{N,Number}
	A₋::NTuple{N,Number}
	tconst::NTuple{N,Number}
	rev::NTuple{N,Number}
	Φ::NTuple{N,Number}
	τ::NTuple{N,Number}
	miniw::NTuple{N,Number}
	σ::NTuple{N,Number}
	δt::Number
end

struct MatrixTypes
	activation::NeuronLayers
	rec::NeuronLayers
	spiked::NeuronLayers
	spt::NeuronLayers
	I::NeuronLayers
	ACh::NeuronLayers
	input::NeuronLayers
	synapses::SynapseLayers
	weights::SynapseLayers
	γ::SynapseLayers
end

struct NeuroTransmitter
	da::Array
	BA::Array
end

function get_parameters()

	nn = (20, 100, 1)
	c = (-65, -65, -65)
	d = (8, 8, 8)
	C = (100, 4, 100)
	noisestd = (0.05, 0.05, 0.05)    
	vr = (-60, -60, -60)
	cap = (100, 4, 100)
	a = (0.3, 0.01, 0.3)
	b = (-0.2, -0.3, -0.2)
	k = (2, 0.035, 2)
	vt = (-40, -25, -40)
	dvoltage = (-60, -85, -60)
	synt = (3, 8, 8)
	quantile = (0.93, 8, 8)
	t₋ = (15, 15, 15)
	A₋ = (-1, -1, -1)
	tconst = (40, 40, 40)
	rev = (0, 0, 0)
	Φ = (0.5, 0.5, 0.5)
	τ = (20, 20, 20)
	miniw = (0.0, 0.0 ,0.0)
	σ = (0.05, 0.05, 0.05)
	δt = 1

	parameters = ParameterTypes(nn,c,d,C,noisestd,vr,cap,a,b,k,vt,dvoltage,synt,quantile,t₋,A₋,tconst,rev,Φ,τ,miniw,σ,δt)
	
	return parameters
end

function initialise_matrices(nn)

	activation = NeuronLayers([NeuronLayer(-60.0, i) for i in nn])
	rec = NeuronLayers([NeuronLayer(0.0, i) for i in nn])
	spiked = NeuronLayers([NeuronLayer(false, i) for i in nn])
	spt = NeuronLayers([NeuronLayer(Int(-1), i) for i in nn])
	I = NeuronLayers([NeuronLayer(250.0, i) for i in nn])
	ACh = NeuronLayers([NeuronLayer(0.0, i) for i in nn])
	input = NeuronLayers([NeuronLayer(0.0, i) for i in nn])

	weights = create_synapses(SynapseLayers,nn)
	synapses = clone_synapses(weights)
	γ = fill_synapses(SynapseLayers, nn, 0.0)

	return activation, rec, spiked, spt, I, ACh, input, synapses, weights, γ
end

function initialise_matrices(nn, weights::SynapseLayers)

	activation = NeuronLayers([NeuronLayer(-60.0, i) for i in nn])
	rec = NeuronLayers([NeuronLayer(0.0, i) for i in nn])
	spiked = NeuronLayers([NeuronLayer(false, i) for i in nn])
	spt = NeuronLayers([NeuronLayer(Int(-1), i) for i in nn])
	I = NeuronLayers([NeuronLayer(250.0, i) for i in nn])
	ACh = NeuronLayers([NeuronLayer(0.0, i) for i in nn])
	input = NeuronLayers([NeuronLayer(0.0, i) for i in nn])

	synapses = clone_synapses(weights)
	γ = fill_synapses(SynapseLayers, nn, 0.0)

	return activation, rec, spiked, spt, I, ACh, input, synapses, weights, γ
end

function get_parameters(f::typeof(update_activation!), p::ParameterTypes, l::Int=1)

	return (p.vt[l], p.vr[l], p.C[l], p.a[l], p.b[l], p.c[l], p.d[l], p.k[l], p.σ[l], p.δt)
end

function get_parameters(f::typeof(update_weights!), p::ParameterTypes, l::Int=1)

	return (p.tconst[l], p.A₋[l], p.t₋[l], p.miniw[l], p.δt)
end

function get_parameters(f::typeof(update_ACh!), p::ParameterTypes, l::Int=1)

	return (p.Φ[l], p.synt[l], p.δt)
end

function get_matrices(f::typeof(update_activation!), m::MatrixTypes, l::Int=1)

	return (m.activation.layers[l], m.spiked.layers[l], m.spt.layers[l], m.rec.layers[l], m.input.layers[l])
end

function get_matrices(f::typeof(update_weights!), m::MatrixTypes, l::Int=1)

	return (m.weights.layers[l], m.γ.layers[l], m.synapses.layers[l], m.spt.layers[l], m.spt.layers[l+1])
end

function get_matrices(f::typeof(update_ACh!), m::MatrixTypes, l::Int=1)

	return (m.ACh.layers[l], m.spt.layers[l])
end

function get_matrices(f::typeof(calc_input!), m::MatrixTypes, l::Int=1)

	return (m.input.layers[l+1], m.weights.layers[l], m.ACh.layers[l], m.activation.layers[l+1])
end