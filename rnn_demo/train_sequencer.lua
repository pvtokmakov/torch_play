require 'rnn'
require 'optim'

local batchSize = 8
local seqLen = 5
local hiddenSize = 7
local nIndex = 10
local lr = 0.1

local r = nn.LSTM(hiddenSize, hiddenSize, seqLen)
--local r = nn.Recurrent(hiddenSize, nn.LookupTable(nIndex, hiddenSize), nn.Linear(hiddenSize, hiddenSize), nn.Sigmoid(),
    --seqLen)

local rnn = nn.Sequential():add(nn.LookupTable(nIndex, hiddenSize)):add(r):add(nn.Linear(hiddenSize, nIndex)):add(nn.LogSoftMax())
--local rnn = nn.Sequential():add(r):add(nn.Linear(hiddenSize, nIndex)):add(nn.LogSoftMax())

rnn = nn.Sequencer(rnn)

local params, gradParams = rnn:getParameters()

print(rnn)

local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

local sequence_ = torch.DoubleTensor():range(1,10) -- 1,2,3,4,5,6,7,8,9,10
local sequence = torch.DoubleTensor(100,10):copy(sequence_:view(1,10):expand(100,10))
sequence:resize(100*10) -- one long sequence of 1,2,3...,10,1,2,3...10...

local offsets = {}
for i = 1, batchSize do
    table.insert(offsets, math.ceil(math.random() * sequence:size(1)))
end
offsets = torch.LongTensor(offsets)

local optimState = {learningRate=lr }

local iteration = 1
while true do
    local inputs, targets = {}, {}
    for step = 1, seqLen do
        inputs[step] = sequence:index(1, offsets)
        offsets:add(1)

        for j=1,batchSize do
            if offsets[j] > sequence:size(1) then
                offsets[j] = 1
            end
        end

        targets[step] = sequence:index(1, offsets)
    end

    --rnn:zeroGradParameters()

    local function feval(params)
        collectgarbage()

        rnn:zeroGradParameters()

        local outputs = rnn:forward(inputs)

        local out = outputs[2]
        out = torch.exp(out[{{3}, {}}])
        local confidences, indices = torch.sort(out, true)
        local target = targets[2]
        local index = indices[{{}, {1}}]
        print(target[3] .. ': ' .. index[1][1])

        local loss = criterion:forward(outputs, targets)
        print('Loss: ' .. loss)
        local dloss_doutput = criterion:backward(outputs, targets)
        rnn:backward(inputs, dloss_doutput)

        return loss,gradParams
    end

    optim.sgd(feval, params, optimState)

--    local outputs = rnn:forward(inputs)
--    local out = outputs[1]
--    out = torch.exp(out[{{4}, {}}])
--    local confidences, indices = torch.sort(out, true)
--    local target = targets[1]
--    local index = indices[{{}, {4}}]
--    print(target[1] .. ': ' .. index[1][1])
--    local err = criterion:forward(outputs, targets)
--
--    print(string.format("Iteration %d ; NLL err = %f ", iteration, err))
--
--    local gradOutputs = criterion:backward(outputs, targets)
--    local gradInputs = rnn:backward(inputs, gradOutputs)
--
--    rnn:updateParameters(lr)

    iteration = iteration + 1
end