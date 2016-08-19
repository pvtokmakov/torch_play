require 'loadcaffe'
require 'image'
require 'cunn'
require 'optim'
require 'form_batch'

model = loadcaffe.load('deploy.prototxt', 'VGG_CNN_S.caffemodel', 'nn')
model:remove(22)
model:remove(22)
model:add(nn.Linear(4096, 101))
model:add(nn.LogSoftMax())
model:float()

print(model)

model = model:cuda()

local params, gradParams = model:getParameters()
criterion = nn.ClassNLLCriterion()
criterion = criterion:float()
criterion = criterion:cuda()

local optimState = {learningRate=0.01,momentum=0.9}

for iter = 1, 20000 do
	if iter == 14000 then
		optimState.learningRate = 0.001
	end

	print('Iter: ' .. iter)

	batch_ims, batch_labels = get_batch()
	batch_ims = batch_ims:float():cuda()
	batch_labels = batch_labels:float():cuda()

	local function feval(params)
                collectgarbage()
   
		gradParams:zero()

		local outputs = model:forward(batch_ims)

		local loss = criterion:forward(outputs, batch_labels)
		print('Loss: ' .. loss)
		local dloss_doutput = criterion:backward(outputs, batch_labels)
		model:backward(batch_ims, dloss_doutput)

		return loss,gradParams
	end

  	optim.sgd(feval, params, optimState)
end

torch.save('image_stream_finetune.dat', model)
