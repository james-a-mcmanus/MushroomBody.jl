const ImageFolder = raw"C:\Users\James\.julia\dev\MushroomBody\ImageData\kaggle\natural_images\flower"

function ColorInput(arraysize::Tuple; normalise_to=300, stages=[0,1,0], input_bool=Bool[0,1,0], da_bool=Bool[0,1,0])	
	img = random_image(ImageFolder)
	init = normalise(bin(get_hues(img),arraysize[1]),normalise_to)
	return ColorInput(init, stages, input_bool, da_bool, cumsum(stages),img)
end

function ColorInput(img::Image, arraysize::Tuple; normalise_to=300, stages=[0,1,0], input_bool=Bool[0,1,0], da_bool=Bool[0,1,0])
	
	ColorInput(normalise(bin(get_hues(img), arraysize[1]), normalise_to), stages, input_bool, da_bool, cumsum(stages), img)
end

function normalise(array, norm_to)

	# normalises the hues by the maximum
	array .* (norm_to / maximum(array))
end

function display_hues(hues)

	minH = 0
	maxH = 360
	hue_list = range(minH, maxH, length=length(hues))
	out = Array{HSL,1}(undef,length(hues))
	hues = normalise(hues,1)

	for i = 1:length(hues)

		out[i] = HSL(hue_list[i],hues[i],0.5)

	end
	return out
end

function color_sequence(arraysize, nstim; stages=[0,1,0], input_bool=Bool[0,1,0], da_bool=Bool[0,1,0])

	out = Vector{Inputs}(undef, nstim)	

	for i = 1:nstim
		out[i] = ColorInput(arraysize, stages=stages, input_bool=input_bool, da_bool=da_bool)
	end

	return InputSequence(out, RestInput(arraysize))
end