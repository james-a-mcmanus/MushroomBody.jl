# basically the same as map but maps over each array in an array of arrays
function maparray(a::Any, f::Function)

	[f.(elem) for elem in a]
	
end