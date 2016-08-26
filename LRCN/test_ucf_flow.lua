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
    local out_dim = 227

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

model = torch.load('flow_single_frame.dat'):float()
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

        local catInd = catMap[folder]

        print(line)

        local path = '/scratch/clear/ptokmako/gvarol/' .. folder .. '/'.. file_split .. '.avi.t7'

        local video = torch.load(path)

        local file = io.open('/scratch/clear/ptokmako/gvarol/' .. folder .. '/'.. file_split .. '.avi_minmax.txt')
        local minmaxes = {}
        local ind = 1
        if file then
            for line in file:lines() do
                local min_x, max_x, min_y, max_y = unpack(line:split(" "))
                minmaxes[ind]  = {min_x, max_x, min_y, max_y}
                ind = ind + 1
            end
        else
            print('File not found!!!!!!!!')
        end

        local first_frame = torch.cat(image.decompressJPG(video['x'][1]:byte()), image.decompressJPG(video['y'][1]:byte()), 1)
        local sz = first_frame:size()

        local step = 8
        local num_steps = (#video['x'] - 16) / step
        if num_steps < 1 then
            num_steps = 1
        end
        local start_ind = 1
        local meanPreds = torch.Tensor(101)
        meanPreds:zero()
        for i = 1, num_steps do
            local batch_ims = torch.Tensor(160, 3, 227, 227)
            im = torch.Tensor(3, sz[2], sz[3])
            for f = 0, 15 do
                if start_ind + f < #video['x'] then
                    im[{{1}, {}, {}}] = image.decompressJPG(video['x'][start_ind + f]:byte())
                    im[{{2}, {}, {}}] = image.decompressJPG(video['y'][start_ind + f]:byte())
                    
                    local mm = minmaxes[start_ind + f]
                    im[{{1}, {}, {}}] = im[{{1}, {}, {}}] * (mm[2] - mm[1]) + mm[1]
                    im[{{2}, {}, {}}] = im[{{2}, {}, {}}] * (mm[4] - mm[3]) + mm[3]
                    im[{{3}, {}, {}}] = torch.sqrt(torch.pow(im[{{1}, {}, {}}], 2) + torch.pow(im[{{2}, {}, {}}], 2))

                    im[{{1}, {}, {}}] = im[{{1}, {}, {}}] - im[{{1}, {}, {}}]:mean()
                    im[{{2}, {}, {}}] = im[{{2}, {}, {}}] - im[{{2}, {}, {}}]:mean()
                    im[{{3}, {}, {}}] = im[{{3}, {}, {}}] - im[{{3}, {}, {}}]:mean()
                end

                batch_ims = cropImage(im, batch_ims, f + 1)
            end

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




