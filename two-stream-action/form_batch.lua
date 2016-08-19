require 'lfs'
require 'image'

local function unique_list(list)
	set = {}
	for i = 1, #list do
		if not set[list[i]] then
			set[list[i]] = true
		end
	end

	unique = {}
	for k, _ in pairs(set) do
		table.insert(unique, k)
	end

	return unique
end

local function crop_image(im)
	size = im:size()
	out_dim = 224
	slack_x = size[3] - out_dim
	slack_y = size[2] - out_dim
	gap_x = math.floor(math.random() * slack_x + 0.5)
	gap_y = math.floor(math.random() * slack_y + 0.5)	
	croped_im = image.crop(im, gap_x, gap_y, size[3] - (slack_x - gap_x), size[2] - (slack_y - gap_y))
	
	return croped_im
end

math.randomseed(os.time())
means = torch.load('UCF_means.dat')

local file = io.open("/scratch2/clear/pweinzae/UCF101/original/splits_classification/trainlist01.txt")

local videos = {}
local tree = {}
local classes = {}
local labels = {}
if file then
    for line in file:lines() do
        local vid_path, label = unpack(line:split(" "))        

        local folder, file_name = unpack(vid_path:split("/"))

	table.insert(classes, folder)
	labels[folder] = label

	local file_split, _ = file_name:match("([^.]+).([^.]+)")

	if videos[folder] == nil then
		videos[folder] = {}
	end

	table.insert(videos[folder], file_split)

	if tree[folder] == nil then
		tree[folder] = {}
	end
	videos_map = tree[folder];	
	videos_map[file_split] = {}

	for file in lfs.dir('/scratch/clear/ptokmako/datasets/UCF101/frames/resized/' .. folder .. '/' .. file_split .. '/') do		               
	    	if lfs.attributes(file, "mode") ~= "directory" then			
	              	table.insert(videos_map[file_split], file)
	    	end
	end        
    end
else
    print('File not found')
end

classes = unique_list(classes)

function get_batch()
	batch_ims = torch.Tensor(256, 3, 224, 224)
	batch_labels = torch.Tensor(256)
	for i = 1, 256 do
		class_name = classes[math.floor(math.random() * (#classes - 1) + 0.5) + 1]
		class_vids = videos[class_name]
		vid_name = class_vids[math.floor(math.random() * (#class_vids - 1) + 0.5) + 1]
	
		vid_nodes = tree[class_name]
		frames = vid_nodes[vid_name]

		frame = frames[math.floor(math.random() * (#frames - 1) + 0.5) + 1]

		im = image.load('/scratch/clear/ptokmako/datasets/UCF101/frames/resized/' .. class_name .. '/' .. vid_name .. '/' .. frame)

		im = crop_image(im)
	
		if math.random() > 0.5 then
			im = image.hflip(im)	
		end

		batch_ims[i] = im
		batch_labels[i] = labels[class_name]
	end

	batch_ims = batch_ims * 255

	for j = 1, 3 do
		batch_ims[{{}, {j}, {}, {} }]:add(-means[j])
	end

	return batch_ims, batch_labels

end
