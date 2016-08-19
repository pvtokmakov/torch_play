require 'loadcaffe'
require 'image'
require 'hdf5'


--means = {0.48462227599918, 0.45624044862054, 0.40588363755159}

model = loadcaffe.load('deploy.prototxt', 'VGG_CNN_S.caffemodel', 'nn')
model:float()
--model = torch.load('nin_nobn_final.t7'):unpack():float()
model:evaluate()

mean_file = hdf5.open('VGG_mean.h5', 'r')
mean = mean_file:read('/mean'):all()
mean = mean:permute(3, 1, 2):float()
temp = mean[{{1}, {}, {}}]
mean[{{1}, {}, {}}] = mean[{{3}, {}, {}}]
mean[{{3}, {}, {}}] = temp

cat_im = image.load('cat.jpg')
cat_im = image.scale(cat_im, 224, 224,'bilinear')

local synset_words = {}
for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end

--for i=1,3 do cat_im[i]:add(-means[i]) end

local I = cat_im:view(1,3,224,224):float()

I = I * 255
I = I - mean

conf, pred = model:forward(I):view(-1):sort(true)

print(conf[1])
print('predicted class'  ..': ', synset_words[pred[1]])
