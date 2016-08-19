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

function get_flow_batch()
    local out_dim = 227
    local batch_ims = torch.Tensor(128, 3, out_dim, out_dim)
    local batch_labels = torch.Tensor(128)

    for i = 1, 8 do
        local class_name = classes[math.floor(math.random() * (#classes - 1) + 0.5) + 1]
        local class_vids = videos[class_name]
        local vid_name = class_vids[math.floor(math.random() * (#class_vids - 1) + 0.5) + 1]

        local path = '/scratch/clear/ptokmako/gvarol/' .. class_name .. '/'.. vid_name .. '.avi.t7'

        local video = torch.load(path)

        local frame_ind = math.floor(math.random() * (#video['x'] - 16) + 0.5) + 1
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

        local first_frame = torch.cat(image.decompressJPG(video['x'][1]:byte()), image.decompressJPG(video['y'][1]:byte()), 1)
        local sz = first_frame:size()

        local flip_flag = math.random()
        local mid_x = math.floor(sz[3] / 2)
        local mid_y = math.floor(sz[2] / 2)
        for f = 0, 15 do
            im = torch.Tensor(3, sz[2], sz[3])
            if (f + 1) < #video['x'] then
                im[{{1}, {}, {}}] = image.decompressJPG(video['x'][frame_ind + f]:byte())
                im[{{2}, {}, {}}] = image.decompressJPG(video['y'][frame_ind + f]:byte())

                im = image.crop(im, mid_x - out_dim / 2, mid_y - out_dim / 2, mid_x + out_dim / 2, mid_y + out_dim / 2)

                if flip_flag > 0.5 then
                    im = image.hflip(im)
                end

                local mm = minmaxes[frame_ind + f]
                im[{{1}, {}, {}}] = im[{{1}, {}, {}}] * (mm[2] - mm[1]) + mm[1]
                im[{{2}, {}, {}}] = im[{{2}, {}, {}}] * (mm[4] - mm[3]) + mm[3]
            end

            im[{{3}, {}, {}}] = torch.sqrt(torch.pow(im[{{1}, {}, {}}], 2) +  torch.pow(im[{{2}, {}, {}}], 2)) --sqrt(x^2+y^2)

            batch_ims[{{16 * (i - 1) + f + 1}, {1}, {}, {}}] = im[{{1}, {}, {}}] - im[{{1}, {}, {}}]:mean()
            batch_ims[{{16 * (i - 1) + f + 1}, {2}, {}, {}}] = im[{{2}, {}, {}}] - im[{{2}, {}, {}}]:mean()
            batch_ims[{{16 * (i - 1) + f + 1}, {3}, {}, {}}] = im[{{3}, {}, {}}] - im[{{3}, {}, {}}]:mean();
            batch_labels[16 * (i - 1) + f + 1] = labels[class_name]
        end
    end

    return batch_ims, batch_labels

end



