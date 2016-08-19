require 'image'
require 'cunn'
require 'optim'
require 'form_flow_batch'
require 'loadcaffe'
require 'cutorch'

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id')

params = cmd:parse(arg)
cutorch.setDevice(params.gpu + 1)

math.randomseed(os.time())

net = loadcaffe.load('deploy.prototxt', 'VGG_CNN_S.caffemodel', 'nn')
net:remove(22)
net:remove(22)
net:add(nn.Linear(4096, 101))
net:add(nn.LogSoftMax())
net:remove(1)
net:insert(nn.SpatialConvolution(20, 96, 7, 7, 2, 2), 1)

print(net)

net:float()
net = net:cuda()

local params, gradParams = net:getParameters()
local criterion = nn.ClassNLLCriterion()
criterion = criterion:float()
criterion = criterion:cuda()

local optimState = {learningRate=0.01,momentum=0.9 }

videos, tree, classes, labels = load_dataset()

for iter = 1, 80000 do
    if iter == 50000 then
        optimState.learningRate = 0.001
    end
    if iter == 70000 then
        optimState.learningRate = 0.0001
    end

    if iter % 1000 == 0 then
        torch.save('temporal_stream_finetune_' .. iter .. '.dat', net)
    end

    timer = torch.Timer()
    local batch_ims, batch_labels = get_flow_batch()
    batch_ims = 255 * batch_ims
    batch_ims = batch_ims:float():cuda()
    batch_labels = batch_labels:float():cuda()

    local function feval(params)
        collectgarbage()

        gradParams:zero()

        local outputs = net:forward(batch_ims)

        local loss = criterion:forward(outputs, batch_labels)
        print('Loss: ' .. loss)
        local dloss_doutput = criterion:backward(outputs, batch_labels)
        net:backward(batch_ims, dloss_doutput)

        return loss,gradParams
    end

    optim.sgd(feval, params, optimState)

    print('Time for iteration: ' .. timer:time().real .. ' seconds')
end

torch.save('temporal_stream_finetune.dat', net)
