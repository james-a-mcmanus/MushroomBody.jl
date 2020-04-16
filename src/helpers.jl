# basically the same as map but maps over each array in an array of arrays
function maparray(a::Any, f::Function)

	[f.(elem) for elem in a]
	
end


# find the product of a sparse and dense matrix, faster than native julia multiplication when the density is v low (less than 1%)
function sparsedensemult(sparsemat::SparseMatrixCSC, densemat)

	outmat = sparse([],[],[],size(sparsemat,1),size(sparsemat,2))
	(i,j,v) = findnz(sparsemat)
	
	for it = 1:length(i) 

		broadcast(*, outmat[i[it],j[it]], v[it],densemat[i[it],j[it]])

	end

	return outmat

end