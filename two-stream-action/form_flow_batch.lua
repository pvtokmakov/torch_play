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

function load_dataset()
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

			for file in lfs.dir('/local_sysdisk/USERTMP/ptokmako/ramflow/' .. folder .. '/' .. file_split .. '/flow_jpg/') do
				if lfs.attributes(file, "mode") ~= "directory" then
					table.insert(videos_map[file_split], file)
				end
			end
		end
	else
		print('File not found')
	end

	classes = unique_list(classes)

	return videos, tree, classes, labels
end


function get_flow_batch()
	local batch_ims = torch.Tensor(200, 20, 224, 224)
	local batch_labels = torch.Tensor(200)

	for i = 1, 200 do
		class_name = classes[math.floor(math.random() * (#classes - 1) + 0.5) + 1]
		class_vids = videos[class_name]
		vid_name = class_vids[math.floor(math.random() * (#class_vids - 1) + 0.5) + 1]

		vid_nodes = tree[class_name]
		frames = vid_nodes[vid_name]

		frame_ind = math.floor(math.random() * (#frames - 11) + 0.5) + 1
		frame = frames[frame_ind]

		first_frame = image.load('/local_sysdisk/USERTMP/ptokmako/ramflow/' .. class_name .. '/' .. vid_name .. '/flow_jpg/' .. frame)
		sz = first_frame:size()

		local ims = torch.Tensor(20, sz[2], sz[3])
		for f = 0, 9 do
			frame_name = '0000' .. frame_ind + f
			str_ind = frame_name:len() - 4
			frame_name = frame_name:sub(str_ind) .. '.jpg'
			local im = image.load('/local_sysdisk/USERTMP/ptokmako/ramflow/' .. class_name .. '/' .. vid_name .. '/flow_jpg/' .. frame_name)
			chanel_ind = f * 2 + 1
			ims[{{chanel_ind}, {}, {}}] = im[{{1}, {}, {}}] - im[{{1}, {}, {}}]:mean()
			ims[{{chanel_ind + 1}, {}, {}}] = im[{{2}, {}, {}}] - im[{{2}, {}, {}}]:mean()
		end

		ims = crop_image(ims)

		if math.random() > 0.5 then
			ims = image.hflip(ims)
		end

		batch_ims[i] = ims
		batch_labels[i] = labels[class_name]
	end

	return batch_ims, batch_labels

end
