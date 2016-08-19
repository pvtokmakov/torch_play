require 'cunn'
require 'image'
require 'lfs'
require 'optim'
require 'cutorch'

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id')

params = cmd:parse(arg)
print(params.gpu + 1)
cutorch.setDevice(params.gpu + 1)

local function cropImage(im, batch, batchInd)
	local size = im:size()
	local out_dim = 224
	
	local croped_im = image.crop(im, 0, 0, out_dim, out_dim)
	batch[batchInd]	= croped_im
	croped_im = image.hflip(croped_im)
	batch[batchInd + 1] = croped_im
        croped_im = image.crop(im, size[3] - out_dim, 0, size[3], out_dim)	
	batch[batchInd + 2] = croped_im
	croped_im = image.hflip(croped_im)
	batch[batchInd + 3] = croped_im
        croped_im = image.crop(im, 0, size[2] - out_dim, out_dim, size[2])
	batch[batchInd + 4] = croped_im
	croped_im = image.hflip(croped_im)
	batch[batchInd + 5] = croped_im
        croped_im = image.crop(im, size[3] - out_dim, size[2] - out_dim, size[3], size[2])
	batch[batchInd + 6] = croped_im
	croped_im = image.hflip(croped_im)
	batch[batchInd + 7] = croped_im
	
	local mid_x = math.floor(size[3] / 2)
	local mid_y = math.floor(size[2] / 2)
	croped_im = image.crop(im, mid_x - out_dim / 2, mid_y - out_dim / 2, mid_x + out_dim / 2, mid_y + out_dim / 2)
	batch[batchInd + 8] = croped_im
	croped_im = image.hflip(croped_im)
	batch[batchInd + 9] = croped_im

    return batch
end

model = torch.load('image_stream_finetune.dat'):float()
model:evaluate()
model = model:cuda()

means = torch.load('UCF_means.dat')

local file = io.open("/scratch2/clear/pweinzae/UCF101/original/splits_classification/classInd.txt")

local catMap = {}
local classes = {}
if file then
    for line in file:lines() do	
        local catInd, catName = unpack(line:split(" "))
        catName = string.sub(catName, 1, -2)
        catMap[catName] = tonumber(catInd)
        table.insert(classes, catName)
    end
end



local file = io.open("/scratch2/clear/pweinzae/UCF101/original/splits_classification/testlist01.txt")

confusion = optim.ConfusionMatrix(classes)

correct = {}
count = {}
for i = 1, 101 do
    table.insert(correct, 0)
    table.insert(count, 0)
end

if file then
    for line in file:lines() do
        local folder, file_name = unpack(line:split("/"))

        local file_split, _ = file_name:match("([^.]+).([^.]+)")

        catInd = catMap[folder]
        local frames = {}
        for file in lfs.dir('/scratch/clear/ptokmako/datasets/UCF101/frames/resized/' .. folder .. '/' .. file_split .. '/') do
            if lfs.attributes(file, "mode") ~= "directory" then
                table.insert(frames, file)
            end
        end

        local step = math.floor(#frames / 25)
        local batch_ims = torch.Tensor(250, 3, 224, 224)
        local batchInd = 1
        for i = 1, step * 25, step do
            local im = image.load('/scratch/clear/ptokmako/datasets/UCF101/frames/resized/' .. folder .. '/' .. file_split .. '/' .. string.format('%05d', i) .. '.jpg')
            batch_ims = cropImage(im, batch_ims, batchInd)
            batchInd = batchInd + 10
        end

        batch_ims = batch_ims:float() * 255
        for j = 1, 3 do
            batch_ims[{{}, {j}, {}, {} }]:add(-means[j])
        end
        batch_ims = batch_ims:cuda()

        local prediction = model:forward(batch_ims)
        meanPreds = torch.Tensor(101)
        for j = 1, 101 do
            meanPreds[j] = prediction[{{}, {j}}]:mean()
        end

        local confidences, indices = torch.sort(meanPreds, true)

        confusion:add(indices[1], catInd)
    end
end

print(confusion)
print('mean class accuracy: ' .. confusion.totalValid)


