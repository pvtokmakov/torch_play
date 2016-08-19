require 'loadcaffe'
require 'cunn'
require 'optim'
require 'cutorch'

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
        require 'form_flow_batch_single_frame'
        cutorch.setDevice(params.gpu + 1)
    end,
    function(threadid)
        math.randomseed(seed + threadid)
        videos, classes, labels = load_dataset()
        print('starting a new thread/state number ' .. threadid)
    end
)


net = loadcaffe.load('deploy.prototxt', 'VGG_CNN_S.caffemodel', 'nn')
net:remove(22)
net:remove(22)
net:add(nn.Linear(4096, 101))
net:add(nn.LogSoftMax())
net:float()

print(net)

net = net:cuda()

local params, gradParams = net:getParameters()
criterion = nn.ClassNLLCriterion()
criterion = criterion:float()
criterion = criterion:cuda()

local optimState = {learningRate=0.001,momentum=0.9,weightDecay=0.005 }

for iter = 1, 50000 do
    if iter == 20000 then
        optimState.learningRate = 0.0001
        optimState.weightDecay=0.0005
    end
    if iter == 40000 then
        optimState.learningRate = 0.00001
        optimState.weightDecay=0.00005
    end

    pool:addjob(
    -- the job callback (runs in data-worker thread)
        function()
            local batch_ims, batch_labels = get_flow_batch()
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
torch.save('flow_single_frame.dat', net)