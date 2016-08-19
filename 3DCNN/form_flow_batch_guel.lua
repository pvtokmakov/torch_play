require 'lfs'
require 'image'
--local threads = require 'threads'
--threads.serialization('threads.sharedserialize')

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

function load_dataset()
    local file = io.open("/scratch2/clear/pweinzae/UCF101/original/splits_classification/trainlist01.txt")

    local videos = {}
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
        end
    else
        print('File not found')
    end

    classes = unique_list(classes)

    return videos, classes, labels
end

--local videos, classes, labels = load_dataset()
--local manual_seed = math.random()

--local pool = threads.Threads(
--    8,
--    function()
--        require 'torch'
--        require 'image'
--    end,
--    function(threadid)
--        l_classes = classes
--        l_videos = videos
--        l_labels = labels
--        local seed = manual_seed + threadid
--        math.randomseed(seed)
--        print('starting a new thread/state number ' .. threadid)
--    end
--)

function get_flow_batch()
    local batch_ims = torch.Tensor(15, 2, 60, 58, 58)
    local batch_labels = torch.Tensor(15)

    timer = torch.Timer()
    for i = 1, 15 do
--        pool:addjob(
--            function()
                local class_name = classes[math.floor(math.random() * (#classes - 1) + 0.5) + 1]
                local class_vids = videos[class_name]
                local vid_name = class_vids[math.floor(math.random() * (#class_vids - 1) + 0.5) + 1]

                local path = '/scratch/clear/ptokmako/gvarol/' .. class_name .. '/'.. vid_name .. '.avi.t7'

                local video = torch.load(path)

                local frame_ind = math.floor(math.random() * (#video['x'] - 60) + 0.5) + 1
                if frame_ind < 1 then
                    frame_ind = 1
                end

                local file = io.open('/scratch/clear/ptokmako/gvarol/' .. class_name .. '/'.. vid_name .. '.avi_minmax.txt')
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

                local firstFrame = image.decompressJPG(video['x'][1]:byte())
                local origSize = firstFrame:size()

                local flip_flag = math.random();
                local out_dim = 58
                local slack_x = 89 - out_dim
                local slack_y = 67 - out_dim
                local gap_x = math.floor(math.random() * slack_x + 0.5)
                local gap_y = math.floor(math.random() * slack_y + 0.5)
                local ims = torch.Tensor(2, 60, 58, 58)
                for f = 0, 59 do
                    if (f + 1) < #video['x'] then
                        im = torch.cat(image.decompressJPG(video['x'][frame_ind + f]:byte()), image.decompressJPG(video['y'][frame_ind + f]:byte()), 1)

                        im = image.scale(im, 89, 67)

                        im = image.crop(im, gap_x, gap_y, 89 - (slack_x - gap_x), 67 - (slack_y - gap_y))

                        if flip_flag > 0.5 then
                            im = image.hflip(im)
                        end

                        local mm = minmaxes[frame_ind + f]
                        im[{{1}, {}, {}}] = im[{{1}, {}, {}}] * (mm[2] - mm[1]) + mm[1]
                        im[{{2}, {}, {}}] = im[{{2}, {}, {}}] * (mm[4] - mm[3]) + mm[3]

                        im[{{1}, {}, {}}]:mul(89/origSize[3]) -- ATTENTION! NOT sc_w
                        im[{{2}, {}, {}}]:mul(67/origSize[2])
                    end

                    ims[{{1}, {f + 1}, {}, {}}] = im[{{1}, {}, {}}] - im[{{1}, {}, {}}]:mean()
                    ims[{{2}, {f + 1}, {}, {}}] = im[{{2}, {}, {}}] - im[{{2}, {}, {}}]:mean()
                end

                batch_ims[i] = ims
                batch_labels[i] = labels[class_name]

--                return __threadid
--            end
--        )
    end

--    pool:synchronize()

--    print('Time for batch ' .. ' : ' .. timer:time().real .. ' seconds')

    return batch_ims, batch_labels

end



