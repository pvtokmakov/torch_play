--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--if not ffmpeg then paths.dofile('loadVidWithFfmpeg.lua') end
--require 'image'
cjson = require 'cjson'
paths.dofile('dataset.lua')
paths.dofile('util.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
   error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = opt.loadSize   --3, 16, 128 171                 
local sampleSize = opt.sampleSize --3, 16, 112 112

function splitPath(inputstr, sep)
   if sep == nil then
      sep = "%s"
   end
   local t={} ; i=1
   for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
      t[i] = str
      i = i + 1
   end
   return t
end

local function getInterval(chunkNo, W, slide)
   t_beg = (chunkNo - 1)*slide + 1
   t_end = t_beg + W - 1
   return t_beg, t_end
end

local function getDuration(path)
   local _className = paths.basename(paths.dirname(path))
   local _videoName = paths.basename(path)

   local minmaxfile = paths.concat(opt.framesRoot, _className, _videoName) .. '_minmax.txt'
   local totalDuration = 0
   for l in io.lines(minmaxfile) do
      totalDuration = totalDuration + 1
   end
   return totalDuration
end

-- Load a sequence of flow images given the window length (W) and the index of the beginning frame (t_beg).
-- depth = 3 --> x, y, norm
-- depth = 2 --> x, y
local function loadFlowImgInterval(path, W, t_beg, depth, totalDuration)
   local nPadding = 0
   if(totalDuration < W) then
      nPadding = W - totalDuration
   end

   local t_end = t_beg + W - 1
   local imgRoot = opt.framesRoot .. paths.basename(paths.dirname(path)) .. '/' .. paths.basename(path) .. '/' .. paths.basename(path).. '_'
   local _className = paths.basename(paths.dirname(path))
   local _videoName = paths.basename(path)
   local video = torch.load(paths.concat(opt.framesRoot, _className, _videoName..'.t7'))

   local minmaxfile = paths.concat(opt.framesRoot, _className, _videoName) .. '_minmax.txt'
   minmax = torch.Tensor(totalDuration, 4) -- THIS is 4 because (minx, maxx, miny, maxy)
   local ii = 1
   for l in io.lines(minmaxfile) do
      local jj = 1
      for word in string.gmatch(l, "%g+") do
         minmax[{ii, jj}] = word
         jj = jj + 1
      end
      ii = ii + 1
   end


   local imgx = image.decompressJPG(video.x[t_beg]:byte())
   local imgy = image.decompressJPG(video.y[t_beg]:byte())

   local iH = imgx:size(2)
   local iW = imgx:size(3)
   imgx = image.scale(imgx, loadSize[4], loadSize[3])
   imgy = image.scale(imgy, loadSize[4], loadSize[3])

   local height = imgx:size(2)
   local width = imgx:size(3)

   -- Allocate memory
   local input = torch.FloatTensor(depth, W, height, width)
   if(opt.minmax) then
      input[{{1}, {1}, {}, {}}] = (torch.mul(imgx, minmax[{t_beg, 2}] - minmax[{t_beg, 1}]) + minmax[{t_beg, 1}]):mul(opt.coeff*iW/loadSize[4]) -- (maxx-minx)*X+minx 
      input[{{2}, {1}, {}, {}}] = (torch.mul(imgy, minmax[{t_beg, 4}] - minmax[{t_beg, 3}]) + minmax[{t_beg, 3}]):mul(opt.coeff*iH/loadSize[3]) -- (maxy-miny)*Y+miny
   end
   if(depth == 3) then
      input[{{3}, {1}, {}, {}}] = torch.sqrt(torch.pow(input[{{1}, {1}, {}, {}}], 2) +  torch.pow(input[{{2}, {1}, {}, {}}], 2)) --sqrt(x^2+y^2)
      if(opt.perframemean) then
         input[{{3}, {1}, {}, {}}] = input[{{3}, {1}, {}, {}}] - torch.mean(input[{{3}, {1}, {}, {}}]);
      end
   end

   if(opt.perframemean) then
      input[{{1}, {1}, {}, {}}] = input[{{1}, {1}, {}, {}}] - torch.mean(input[{{1}, {1}, {}, {}}]);
      input[{{2}, {1}, {}, {}}] = input[{{2}, {1}, {}, {}}] - torch.mean(input[{{2}, {1}, {}, {}}]);
   end

   --print(input[{{1}, {1}, {30}, {30}}])

   -- Read the remaining frames
   for tt = t_beg+1,t_end - nPadding do
      --imgx = image.load(imgRoot .. string.format("%05d_x.jpg", tt), 1)--:mul(255):byte() 
      --imgy = image.load(imgRoot .. string.format("%05d_y.jpg", tt), 1)--:mul(255):byte() 
      imgx = image.decompressJPG(video.x[tt]:byte())
      imgy = image.decompressJPG(video.y[tt]:byte())
      imgx = image.scale(imgx, loadSize[4], loadSize[3])
      imgy = image.scale(imgy, loadSize[4], loadSize[3])
      local T = tt - t_beg + 1

      if(opt.minmax) then
         input[{{1}, {T}, {}, {}}] = (torch.mul(imgx, minmax[{tt, 2}] - minmax[{tt, 1}]) + minmax[{tt, 1}]):mul(opt.coeff*iW/loadSize[4]) -- tt not T !
         input[{{2}, {T}, {}, {}}] = (torch.mul(imgy, minmax[{tt, 4}] - minmax[{tt, 3}]) + minmax[{tt, 3}]):mul(opt.coeff*iH/loadSize[3]) -- ATTENTION NOT /iH
      end

      if(depth == 3) then
         input[{{3}, {T}, {}, {}}] = torch.sqrt(torch.pow(input[{{1}, {T}, {}, {}}], 2) +  torch.pow(input[{{2}, {T}, {}, {}}], 2)) --sqrt(x^2+y^2)
         if(opt.perframemean) then
            input[{{3}, {T}, {}, {}}] = input[{{3}, {T}, {}, {}}] - torch.mean(input[{{3}, {T}, {}, {}}]);
         end
      end
      if(opt.perframemean) then
         input[{{1}, {T}, {}, {}}] = input[{{1}, {T}, {}, {}}] - torch.mean(input[{{1}, {T}, {}, {}}]);
         input[{{2}, {T}, {}, {}}] = input[{{2}, {T}, {}, {}}] - torch.mean(input[{{2}, {T}, {}, {}}]);
      end
   end

   if(input == nil) then
      print(path .. ' is nil (input)')
      return nil
   elseif(input:size(1) ~= loadSize[1] or input:size(2) ~= loadSize[2] or input:size(3) ~= loadSize[3]) then
      print(path .. ': input size mismatch')
      return nil
   else
      return input
   end
end

--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

local trainHook = function(self, path)
   collectgarbage()
   local input
   -- Video loading
   if(opt.loadMethod == 'vid') then
      input = loadvidtotensor(path)
      -- Image sequence loading
   elseif(opt.loadMethod == 'img') then
      -- Random interval loading
      if(opt.interval) then
         local oT = loadSize[2]
         local iT
         if(opt.flow) then
            iT = getDuration(path)
         else
            iT = getDuration_rgb(path)
         end
         -- Not enough frames
         if(iT < oT) then
            --print(path .. ' not enough frames, padding...')
            if(opt.flow) then
               input = loadFlowImgInterval(path, loadSize[2], 1, opt.channels, iT)
            else
               input = loadRgbImgInterval(path, loadSize[2], 1, iT) --!
            end
            --return nil --do return end
            -- Enough frames
         else
            local t1 = math.ceil(torch.uniform(1e-2, iT-oT+1))
            if(opt.flow) then
               input = loadFlowImgInterval(path, loadSize[2], t1, opt.channels, iT)
            else
               input = loadRgbImgInterval(path, loadSize[2], t1, iT) --!
            end
         end
         -- Sliding fix window
      else
         if(opt.flow) then
            input = loadFlowImgSequence(path, loadSize[2], opt.slide, opt.channels)
         else
            input = loadRgbImgSequence(path, loadSize[2], opt.slide)
         end
      end
   end

   if(input == nil) then
      print('Nil!')
      return nil
   else
      ---------------------------------------------------------
      iW = input:size(4)
      iH = input:size(3)

      local oW
      local oH
      local sc_w
      local sc_h
      if(opt.scales) then
         -- do random multiscale crop
         --select scale
         sc_w = opt.scales[torch.random(#opt.scales)]
         sc_h = opt.scales[torch.random(#opt.scales)]
         oW = math.ceil(loadSize[3]*sc_w)
         oH = math.ceil(loadSize[3]*sc_h)
      else
         oW = sampleSize[4]
         oH = sampleSize[3]
      end
      local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
      local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
      local out = input[{{}, {}, {h1, h1+oH-1}, {w1, w1+oW-1}}]

      -- resize to sample size
      if(out:size(1) ~= sampleSize[1] or out:size(2) ~= sampleSize[2] or out:size(3) ~= sampleSize[3] or out:size(4) ~= sampleSize[4]) then
         out_res = torch.Tensor(sampleSize[1], sampleSize[2], sampleSize[3], sampleSize[4])
         for jj = 1, sampleSize[1] do
            for ii = 1, sampleSize[2] do
               out_res[{{jj}, {ii}, {}, {}}] = image.scale(out[{{jj}, {ii}, {}, {}}]:squeeze(), sampleSize[4], sampleSize[3])
            end
         end
         out = out_res

         -- multiply the flow by the scale factor
         if(opt.flow) then
            out[{{1},{},{},{}}]:mul(sampleSize[4]/oW) -- ATTENTION! NOT sc_w
            out[{{2},{},{},{}}]:mul(sampleSize[3]/oH) -- ATTENTION! NOT sc_h but 112/(128*sc) = 112/oH
            if(opt.channels == 3) then
               for i=1,input:size(1) do
                  out[{{3},{i},{},{}}] = torch.sqrt(torch.pow(out[{{1}, {i}, {}, {}}], 2) +  torch.pow(out[{{2}, {i}, {}, {}}], 2))
               end
            end
         end
      end

      assert(out:size(4) == sampleSize[4])
      assert(out:size(3) == sampleSize[3])

      -- mean/std

      --for t=1,input:size(2) do --frames
      -- if type(mean) ~= 'number' then -- mean:dim() == 4 then
      -- if type(mean) ~= 'number' then
      --   out:add(-mean:float())
      -- else
      out:add(-mean)
      -- end
      --out:div(std)

      --if true then
      --else
      --    for i=1,input:size(1) do -- channels
      --       if mean then out[{{i},{},{},{}}]:add(-mean[i]) end
      --       if std then out[{{i},{},{},{}}]:div(std[i]) end
      --    end
      --end
      ---------------------------------------------------------
      -- do hflip with probability 0.5
      if torch.uniform() > 0.5 then out = image.flip(out:contiguous(), 4); end

      -- randomly shuffle frames
      if(opt.trainshuffle) then
         local shuffle = torch.randperm(out:size(2)):type('torch.LongTensor')
         out = out:index(2, shuffle)
      end

      return out
   end
end

