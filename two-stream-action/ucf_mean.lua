require 'lfs'
require 'image'

local file = io.open("/scratch2/clear/pweinzae/UCF101/original/splits_classification/trainlist01.txt")

acc = {0, 0, 0}
count = 0
if file then
	for line in file:lines() do
		local vid_path, label = unpack(line:split(" "))        

	        local folder, file_name = unpack(vid_path:split("/"))

		local file_split, _ = file_name:match("([^.]+).([^.]+)")
		
		for file in lfs.dir('/scratch/clear/ptokmako/datasets/UCF101/frames/' .. folder .. '/' .. file_split .. '/') do		               
		    	if lfs.attributes(file, "mode") ~= "directory" then
			      	im = image.load('/scratch/clear/ptokmako/datasets/UCF101/frames/' .. folder .. '/' .. file_split .. '/' .. file)
				for i = 1, 3 do
					acc[i] = acc[i] + im[{{i}, {}, {}}]:mean()
				end
				count = count + 1
		    	end
		end        
	end
end

means = {}
for i = 1, 3 do
	means[i] = 255 * (acc[i] / count)
end

torch.save('UCF_means.dat', means)
