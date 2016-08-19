require 'lfs'
require 'image'

local file = io.open("/scratch2/clear/pweinzae/UCF101/original/splits_classification/testlist01.txt")

local ind = 1
if file then
	for line in file:lines() do
		print(ind)
		local vid_path, label = unpack(line:split(" "))        

		local folder, file_name = unpack(vid_path:split("/"))

		local file_split, _ = file_name:match("([^.]+).([^.]+)")

		os.execute("mkdir " .. '/scratch/clear/ptokmako/datasets/UCF101/frames/resized89/' .. folder)
		os.execute("mkdir " .. '/scratch/clear/ptokmako/datasets/UCF101/frames/resized89/' .. folder .. '/' .. file_split)
		os.execute("mkdir " .. '/scratch/clear/ptokmako/datasets/UCF101/frames/resized89/' .. folder .. '/' .. file_split .. '/flow_jpg/')
		
		for file in lfs.dir('/scratch/clear/ptokmako/datasets/UCF101/frames/' .. folder .. '/' .. file_split .. '/') do
			if lfs.attributes(file, "mode") ~= "directory" then
--				local im = image.load('/scratch/clear/ptokmako/datasets/UCF101/frames/' .. folder .. '/' .. file_split .. '/' .. file)

--				local sz = im:size()
				--				min_dim = math.min(sz[2], sz[3])
				--				scale_factor = 256 / min_dim
				--				scaled = image.scale(im, math.floor(sz[3] * scale_factor), math.floor(sz[2] * scale_factor), 'bilinear')
--				scaled = image.scale(im, 89, 67, 'bilinear')
--				image.save('/scratch/clear/ptokmako/datasets/UCF101/frames/resized89/' .. folder .. '/' .. file_split .. '/' .. file, scaled)

				local f=io.open('/scratch/clear/ptokmako/datasets/UCF101/frames/resized/' .. folder .. '/' .. file_split .. '/flow_jpg/' .. file,"r")
				if f ~= nil then
					io.close(f)
					local flow = image.load('/scratch/clear/ptokmako/datasets/UCF101/frames/resized/' .. folder .. '/' .. file_split .. '/flow_jpg/' .. file)
					local flow_sz = flow:size()
					flow_width_factor = flow_sz[3] / 89
					flow_height_factor = flow_sz[2] / 67
					scaled_flow = image.scale(flow, 89, 67, 'bilinear')
					scaled_flow[{{1}, {}, {}}] = scaled_flow[{{1}, {}, {}}] / flow_width_factor
					scaled_flow[{{2}, {}, {}}] = scaled_flow[{{2}, {}, {}}] / flow_height_factor

					image.save('/scratch/clear/ptokmako/datasets/UCF101/frames/resized89/' .. folder .. '/' .. file_split .. '/flow_jpg/' .. file, scaled_flow)
				end
			end
		end
		ind = ind + 1
	end
end
