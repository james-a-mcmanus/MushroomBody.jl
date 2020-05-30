using ColorCoding

const ImageFolder = raw"C:\Users\MIKKO\.julia\dev\MushroomBody\ImageData\kaggle\natural_images\flower"

function ColorInput(arraysize::Tuple; normalise_to=300, stages=[0,1,0], input_bool=Bool[0,1,0], da_bool=Bool[0,1,0])	

	init = normalise_hues(bin(get_hues(random_image(ImageFolder)),arraysize[1]),normalise_to)
	return ColorInput(init, stages, input_bool, da_bool, cumsum(stages))
end

function normalise_hues(hues, norm_to)

	# normalises the hues by the maximum
	hues .* (norm_to / maximum(hues))
end