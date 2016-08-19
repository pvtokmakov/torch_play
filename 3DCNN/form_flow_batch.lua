--
-- Created by IntelliJ IDEA.
-- User: ptokmako
-- Date: 8/3/16
-- Time: 2:03 PM
-- To change this template use File | Settings | File Templates.
--

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

            for file in lfs.dir('/local_sysdisk/USERTMP/ptokmako/ramflow89/' .. folder .. '/' .. file_split .. '/flow_jpg/') do
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
    local batch_ims = torch.Tensor(15, 2, 60, 58, 58)
    local batch_labels = torch.Tensor(15)

    for i = 1, 15 do
        local class_name = classes[math.floor(math.random() * (#classes - 1) + 0.5) + 1]
        local class_vids = videos[class_name]
        local vid_name = class_vids[math.floor(math.random() * (#class_vids - 1) + 0.5) + 1]

        local vid_nodes = tree[class_name]
        local frames = vid_nodes[vid_name]

        local frame_ind = math.floor(math.random() * (#frames - 60) + 0.5) + 1
        if frame_ind < 1 then
            frame_ind = 1
        end
        local frame = frames[frame_ind]

        local first_frame = image.load('/local_sysdisk/USERTMP/ptokmako/ramflow89/' .. class_name .. '/' .. vid_name .. '/flow_jpg/' .. frame)
        local sz = first_frame:size()

        local flip_flag = math.random();
        local out_dim = 58
        local slack_x = sz[3] - out_dim
        local slack_y = sz[2] - out_dim
        local gap_x = math.floor(math.random() * slack_x + 0.5)
        local gap_y = math.floor(math.random() * slack_y + 0.5)
        local ims = torch.Tensor(2, 60, 58, 58)
        for f = 0, 59 do
            if (f + 1) < #frames then
                local frame_name = '0000' .. frame_ind + f
                local str_ind = frame_name:len() - 4
                frame_name = frame_name:sub(str_ind) .. '.jpg'

                im = image.load('/local_sysdisk/USERTMP/ptokmako/ramflow89/' .. class_name .. '/' .. vid_name .. '/flow_jpg/' .. frame_name)

                im = image.crop(im, gap_x, gap_y, sz[3] - (slack_x - gap_x), sz[2] - (slack_y - gap_y))

                if flip_flag > 0.5 then
                    im = image.hflip(im)
                end
            end


            ims[{{1}, {f + 1}, {}, {}}] = im[{{1}, {}, {}}] - im[{{1}, {}, {}}]:mean()
            ims[{{2}, {f + 1}, {}, {}}] = im[{{2}, {}, {}}] - im[{{2}, {}, {}}]:mean()
        end

        batch_ims[i] = ims
        batch_labels[i] = labels[class_name]
    end

    return batch_ims, batch_labels

end


