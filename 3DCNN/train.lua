--
-- Created by IntelliJ IDEA.
-- User: ptokmako
-- Date: 8/2/16
-- Time: 2:56 PM
-- To change this template use File | Settings | File Templates.
--

require 'cunn'
require 'optim'
require 'cutorch'
--require 'form_flow_batch'

local threads = require 'threads'
threads.serialization('threads.sharedserialize')

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id')

local params = cmd:parse(arg)
print(params.gpu + 1)
cutorch.setDevice(params.gpu + 1)

math.randomseed(os.time())
local seed = math.random()

local pool = threads.Threads(
    8,
    function()
        require 'torch'
        require 'cunn'
        require 'cutorch'
        require 'form_flow_batch_guel'
        cutorch.setDevice(params.gpu + 1)
    end,
    function(threadid)
        math.randomseed(seed + threadid)
        videos, classes, labels = load_dataset()
        print('starting a new thread/state number ' .. threadid)
    end
)

net = nn.Sequential()

net:add(nn.VolumetricConvolution(2, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2))
net:add(nn.VolumetricConvolution(64, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
net:add(nn.VolumetricConvolution(128, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
net:add(nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
net:add(nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2))
net:add(nn.View(256*3))
net:add(nn.Linear(256*3, 2048))
net:add(nn.ReLU())
net:add(nn.Dropout(0.9))
net:add(nn.Linear(2048, 2048))
net:add(nn.ReLU())
net:add(nn.Dropout(0.9))
net:add(nn.Linear(2048, 101))
net:add(nn.LogSoftMax())

print(net)

net:float()
net:training()
net = net:cuda()

local params, gradParams = net:getParameters()
local criterion = nn.ClassNLLCriterion()
criterion = criterion:float()
criterion = criterion:cuda()

local optimState = {learningRate=0.003,momentum=0.9,weightDecay=0.005 }

print "Loading dataset"
--videos, tree, classes, labels = load_dataset()

for iter = 1, 290000 do
    if iter == 160000 then
        optimState.learningRate = 0.0003
        optimState.weightDecay=0.0005
    end
    if iter == 250000 then
        optimState.learningRate = 0.00003
        optimState.weightDecay=0.00005
    end



    pool:addjob(
    -- the job callback (runs in data-worker thread)
        function()
            local batch_ims, batch_labels = get_flow_batch()
            --    batch_ims = 255 * batch_ims
            batch_ims = batch_ims:float():cuda()
            batch_labels = batch_labels:float():cuda()
            return batch_ims, batch_labels
        end,
        -- the end callback (runs in the main thread)
        function (batch_ims, batch_labels)
            timer = torch.Timer()

            local function feval(params)
                collectgarbage()

                gradParams:zero()

                local outputs = net:forward(batch_ims)

                local loss = criterion:forward(outputs, batch_labels)
                print("Loss: " .. loss)
                local dloss_doutput = criterion:backward(outputs, batch_labels)
                net:backward(batch_ims, dloss_doutput)

                return loss,gradParams
            end

            optim.sgd(feval, params, optimState)

            print('Time for network pass ' .. iter .. ' : ' .. timer:time().real .. ' seconds')
        end
    )


end

pool:synchronize()

net:clearState()
torch.save('flow_guel_size58_depth60_drop09_scale_value.dat', net)


