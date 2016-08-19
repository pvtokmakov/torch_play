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

local function cropImage(im, batch, frame_ind)
    local size = im:size()
    local out_dim = 58

    local croped_im = image.crop(im, 0, 0, out_dim, out_dim)
    batch[{{1}, {}, {frame_ind}, {}, {}}] = croped_im
    croped_im = image.hflip(croped_im)
    batch[{{2}, {}, {frame_ind}, {}, {}}] = croped_im
    croped_im = image.crop(im, size[3] - out_dim, 0, size[3], out_dim)
    batch[{{3}, {}, {frame_ind}, {}, {}}] = croped_im
    croped_im = image.hflip(croped_im)
    batch[{{4}, {}, {frame_ind}, {}, {}}] = croped_im
    croped_im = image.crop(im, 0, size[2] - out_dim, out_dim, size[2])
    batch[{{5}, {}, {frame_ind}, {}, {}}] = croped_im
    croped_im = image.hflip(croped_im)
    batch[{{6}, {}, {frame_ind}, {}, {}}] = croped_im
    croped_im = image.crop(im, size[3] - out_dim, size[2] - out_dim, size[3], size[2])
    batch[{{7}, {}, {frame_ind}, {}, {}}] = croped_im
    croped_im = image.hflip(croped_im)
    batch[{{8}, {}, {frame_ind}, {}, {}}] = croped_im

    local mid_x = math.floor(size[3] / 2)
    local mid_y = math.floor(size[2] / 2)
    croped_im = image.crop(im, mid_x - out_dim / 2, mid_y - out_dim / 2, mid_x + out_dim / 2, mid_y + out_dim / 2)
    batch[{{9}, {}, {frame_ind}, {}, {}}] = croped_im
    croped_im = image.hflip(croped_im)
    batch[{{10}, {}, {frame_ind}, {}, {}}] = croped_im

    return batch
end

model = torch.load('flow_size58_depth60.dat'):float()
model:evaluate()
model = model:cuda()

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
        for file in lfs.dir('/local_sysdisk/USERTMP/ptokmako/ramflow89/' .. folder .. '/' .. file_split .. '/flow_jpg/') do
            if lfs.attributes(file, "mode") ~= "directory" then
                table.insert(frames, file)
            end
        end

        print(line)

        local step = 4
        local num_steps = (#frames - 60) / step
        if num_steps < 1 then
            num_steps = 1
        end
        local start_ind = 1
        local meanPreds = torch.Tensor(101)
        meanPreds:zero()
        for i = 1, num_steps do
            local batch_ims = torch.Tensor(10, 2, 60, 58, 58)
            for f = 0, 59 do
                local im = image.load('/local_sysdisk/USERTMP/ptokmako/ramflow89/' .. folder .. '/' .. file_split .. '/flow_jpg/' .. string.format('%05d', start_ind + f) .. '.jpg')
                im[{{1}, {}, {}}] = im[{{1}, {}, {}}] - im[{{1}, {}, {}}]:mean()
                im[{{2}, {}, {}}] = im[{{2}, {}, {}}] - im[{{2}, {}, {}}]:mean()

                batch_ims = cropImage(im, batch_ims, f + 1)
            end

            batch_ims = batch_ims:float() * 255
            batch_ims = batch_ims:cuda()

            local prediction = model:forward(batch_ims)
            for j = 1, 101 do
                meanPreds[j] = meanPreds[j] + prediction[{{}, {j}}]:mean()
            end

            start_ind = start_ind + step
        end

        meanPreds:div(num_steps)

        local confidences, indices = torch.sort(meanPreds, true)

        confusion:add(indices[1], catInd)
    end
end

print(confusion)
print('mean class accuracy: ' .. confusion.totalValid)


