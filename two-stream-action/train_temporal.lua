require 'image'
require 'cunn'
require 'optim'
require 'form_flow_batch'
--require 'parallel'
require 'cutorch'

cmd = torch.CmdLine()
cmd:option('-gpu', 0, 'GPU id')

params = cmd:parse(arg)
print(params.gpu + 1)
cutorch.setDevice(params.gpu + 1)

math.randomseed(os.time())

--function child()
--	require 'form_flow_batch'
--
--	math.randomseed(os.time())
--
--	classes = parallel.parent:receive()
--	videos = parallel.parent:receive()
--	tree = parallel.parent:receive()
--	labels = parallel.parent:receive()
--
--	while true do
--		local batch_ims, batch_labels = get_flow_batch()
--
--		parallel.parent:send(batch_ims)
--		parallel.parent:send(batch_labels)
--
--		m = parallel.yield()
--		if m == 'stop' then
--			break
--		end
--	end
--end

net = nn.Sequential()

net:add(nn.SpatialConvolution(20, 96, 7, 7, 2, 2))
net:add(nn.ReLU())
net:add(nn.SpatialCrossMapLRN(5))
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.SpatialConvolution(96, 256, 5, 5, 2, 2))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.View(512*6*6))
net:add(nn.Linear(512*6*6, 4096))
net:add(nn.ReLU())
net:add(nn.Dropout(0.9))
net:add(nn.Linear(4096, 2048))
net:add(nn.ReLU())
net:add(nn.Dropout(0.9))
net:add(nn.Linear(2048, 101))
net:add(nn.LogSoftMax())

net:float()
net = net:cuda()

local params, gradParams = net:getParameters()
local criterion = nn.ClassNLLCriterion()
criterion = criterion:float()
criterion = criterion:cuda()

local optimState = {learningRate=0.01,momentum=0.9 }

videos, tree, classes, labels = load_dataset()

--local c = parallel.fork()
--c:exec(child)
--
--parallel.children:send(classes)
--parallel.children:send(videos)
--parallel.children:send(tree)
--parallel.children:send(labels)
--
--local batch_ims = parallel.children:receive()
--batch_ims = batch_ims[1]
--local batch_labels = parallel.children:receive()
--batch_labels = batch_labels[1]
--c:join()

for iter = 1, 80000 do
	if iter == 50000 then
		optimState.learningRate = 0.001
	end
	if iter == 70000 then
		optimState.learningRate = 0.0001
	end

	if iter % 1000 == 0 then
		torch.save('temporal_stream_' .. iter .. '.dat', net)
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

--	batch_ims = parallel.children:receive()
--	batch_ims = batch_ims[1]
--	batch_labels = parallel.children:receive()
--	batch_labels = batch_labels[1]
--	if iter == 80000 then
--		c:join('stop')
--	else
--		c:join()
--	end

	print('Time elapsed: ' .. timer:time().real .. ' seconds')
end

torch.save('temporal_stream255.dat', net)
