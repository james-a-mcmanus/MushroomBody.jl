import Base: push!
using JLD2, FileIO


"""
these function save variables into the output folder.
"""
function save_variables(m::ContainerTypes, vars::Vector{String})
	for v in vars
		save_variables(m,v)
	end
end

"""
takes a variable, checks if we have a file for that variable name, and saves data in that file
"""
function save_variables(m::ContainerTypes, varname::String)
	newvar = get_variable(m,varname)
	fname = "C:\\Users\\James\\.julia\\dev\\MushroomBody\\src\\output\\Variables\\" * varname * ".jld2"
	if isfile(fname)
		oldvar = load(fname, varname)
		oldvar = !isa(oldvar,Array) ? [oldvar] : oldvar
		push!(oldvar, newvar)
	else 
		oldvar = newvar
	end 

	save(fname, varname, oldvar)

end

get_variable(m::ContainerTypes, var::String) = getproperty(m,var)

#=Base.push!(layers::Union{NeuronLayers,SynapseLayers},newvar) = layers=[layers]; push!([layers],newvar)
=#
