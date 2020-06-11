const flowerFolder = raw"C:\Users\James\.julia\dev\MushroomBody\ImageData\kaggle\natural_images\flower"
const imageFolder = raw"C:\Users\James\.julia\dev\MushroomBody\ImageData\kaggle\natural_images"

function ColorInput(arraysize::Tuple; folder=flowerFolder, normalise_to=350, stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0])	
	img = random_image(folder)
	init = normalise(bin(get_hues(img),arraysize[1]),normalise_to)
	return ColorInput(init, stages, input_bool, reward_bool, punishment_bool, cumsum(stages),img)
end

function ColorInput(img::Image, arraysize::Tuple; normalise_to=350, stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0])
	
	ColorInput(normalise(bin(get_hues(img), arraysize[1]), normalise_to), stages, input_bool, reward_bool, punishment_bool, cumsum(stages), img)
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

function many_folders_sequence(arraysize, nstim; folder=imageFolder, stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0])

	out = Vector{Inputs}(undef, nstim)	
	multifolders = readdir(folder)
	for i = 1:nstim
		image_folder = folder * "\\" * rand(multifolders)
		out[i] = ColorInput(arraysize, folder=image_folder, stages=stages, input_bool=input_bool, reward_bool=reward_bool, punishment_bool=punishment_bool)
	end

	return InputSequence(out, RestInput(arraysize))
end


function color_sequence(arraysize, nstim; folder=flowerFolder, stages=[0,1,0], input_bool=Bool[0,1,0], reward_bool=Bool[0,1,0], punishment_bool=Bool[0,0,0])

	out = Vector{Inputs}(undef, nstim)	
	for i = 1:nstim
		out[i] = ColorInput(arraysize, folder=folder, stages=stages, input_bool=input_bool, reward_bool=reward_bool, punishment_bool=punishment_bool)
	end

	return InputSequence(out, RestInput(arraysize))
end

