require 'torch'
require 'image'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
sid = require 'sid'
local class = require 'class'

local mnist = {}
local State = class('State')

mnist.State = State

function State:__init(use_cuda)
  self.use_cuda = use_cuda
end

-- Loads and preprocesses train/val/test data.
function State:load_data(dir)
  -- Load MNIST data
  local train_file = dir .. '/train_32x32.t7'
  local test_file = dir .. '/test_32x32.t7'
  self.train_data = torch.load(train_file,'ascii')
  self.train_data.data = self.train_data.data:float()
  self.train_data.labels = self.train_data.labels:float()
  self.val_data = { data=self.train_data.data[{{50001, 60000}}],
                    labels=self.train_data.labels[{{50001, 60000}}] }
  self.train_data.data = self.train_data.data[{{1, 50000}}]
  self.train_data.labels = self.train_data.labels[{{1, 50000}}]
  self.test_data = torch.load(test_file,'ascii')
  self.test_data.data = self.test_data.data:float()
  self.test_data.labels = self.test_data.labels:float()

  -- Preprocess train, val and test data.

  -- 1. Calculate mean for train data and subtract the mean from train, val and test data.
  self.mean = self.train_data.data:mean()
  self.train_data.data:add(-self.mean)
  self.val_data.data:add(-self.mean)
  self.test_data.data:add(-self.mean)

  -- 2. Calculate std deviation for train data and divide by it
  self.std = self.train_data.data:std()
  self.train_data.data:div(self.std)
  self.val_data.data:div(self.std)
  self.test_data.data:div(self.std)

  print('Train data:')
  print(self.train_data.labels[{{1, 6}}])
  print("size: ", self.train_data.data:size(), self.train_data.labels:size())
  print()

  print('Test data:')
  print(self.test_data.data:size())
  print(self.test_data.labels[{{1, 6}}])
  print()

  self:init_train()
end

function State:init_train()
  -- Training facilities
  self.reg = 1000 / self.dog.params:size(1) -- L2 regularization strength
  self.gradClip = 5

  self.batchSize = 50
  self.maxBatch = self.train_data.data:size(1) / self.batchSize
  print('maxBatch: ', self.maxBatch)
  self.curBatch = 1
end

local angles = { [1]=3, [2]=10, [3]=3, [4]=10, [5]=3, [6]=10, [7]=3, [8]=10, [9]=10, [0]=10 }

-- Returns the next batch, inputs and labels.
function State:next_batch()
    local batchStart = (self.curBatch-1)*self.batchSize + 1
    local batchEnd = batchStart + self.batchSize - 1
    self.curBatch = self.curBatch + 1
    if self.curBatch > self.maxBatch then
        self.curBatch = 1
    end
    local x = self.train_data.data[{{batchStart, batchEnd}}]:float():clone()
    local y = self.train_data.labels[{{batchStart, batchEnd}}]:float():clone()

    for i = 1, x:size(1) do
      -- Random rotate
      local lim = angles[y[i]-1] * math.pi / 180
      local theta = torch.uniform(-lim, lim) -- about 3 degrees in each direction
      image.rotate(x[i], theta, 'bilinear')
      -- Random translate
      local dx = torch.uniform(-4, 4)
      local dy = torch.uniform(-4, 4)
      image.translate(x[i], dx, dy)
    end

    if self.use_cuda then
        x = x:cuda()
        y = y:cuda()
    end
    return x, y
end

function State:feval(x)
    local dog = self.dog
    if x ~= dog.params then
        dog.params:copy(x)
    end
    dog.grad_params:zero()

    local input, labels = self:next_batch()

    ------------------- forward pass -------------------
    dog.module:training() -- make sure we are in correct mode 
    local prediction = dog.module:forward(input)
    local paramNorm = dog.params:norm()
    local loss = self.criterion:forward(prediction, labels) + self.reg * paramNorm * paramNorm / 2

    ------------------ backward pass -------------------
    local dprediction = self.criterion:backward(prediction, labels)
    dog.module:backward(input, dprediction)
    
    dog.grad_params:add(self.reg, dog.params) -- apply regularization gradient
    dog.grad_params:clamp(-self.gradClip, self.gradClip)
    return loss, dog.grad_params
end

-- Create a dog to train and a criterion.
function State:create_new(arch, args)
  local dog = sid.create(arch, args, self.use_cuda)
  local criterion = nn.ClassNLLCriterion()
  if self.use_cuda then
    criterion:cuda()
  end

  print('params: ', dog.params:size(), dog.params:type())
  print('gradParams: ', dog.grad_params:size(), dog.grad_params:type())

  -- initialization
  dog.params:uniform(-0.08, 0.08) -- small numbers uniform
  self.dog = dog
  self.criterion = criterion
end

function State:predict(input)
    local x = input
    if self.use_cuda then
        x = x:float():cuda()
    end
    self.dog.module:evaluate() -- turn off dropout
    local prediction = self.dog.module:forward(x)
    local _, classes = prediction:max(2)
    return classes
end

function State:eval_accuracy(input, labels)
    local matches = 0
    local batchSize = 1000
    local from = 1
    for i = 1, input:size(1) do
        if i - from + 1 >= batchSize or i == input:size(1) then
            local curLabels = labels[{{from, i}}]
            local predictions = self:predict(input[{{from, i}}], curLabels):float()
            curLabels:map(predictions, function(xx, yy) if xx == yy then matches = matches + 1 end end)
            from = i+1
        end
    end
    
    return matches / labels:size(1)
end

function State:train_accuracy()
  return self:eval_accuracy(self.train_data.data, self.train_data.labels)
end

function State:val_accuracy()
  return self:eval_accuracy(self.val_data.data, self.val_data.labels)
end

function State:test_accuracy()
  return self:eval_accuracy(self.test_data.data, self.test_data.labels)
end

function State:save(filename)
  local dog_obj = sid.to_save(self.dog)
  local checkpoint = {
    dog = dog_obj,
    mean = self.mean,
    std = self.std
  }
  torch.save(filename, checkpoint)
end

function State:save_checkpoint(dir)
  print("Saving a checkpoint ...")
  if not path.exists(dir) then lfs.mkdir(dir) end
  local date = os.date('*t', os.time())
  local val_acc = self:val_accuracy()
  local filename = string.format('%s/mnist-%s-%s-%s-%s-%s-%s.nn',
            dir, date.year, date.month, date.day, date.hour, date.min, val_acc)
  self:save(filename)
  print(string.format("Saved to %s", filename))
end

function mnist.load(filename, use_cuda)
  local checkpoint = torch.load(filename)
  local state = State(use_cuda)
  state.dog = sid.load_from(checkpoint.dog, use_cuda)
  state.mean = checkpoint.mean
  state.std = checkpoint.std
  return state
end

-- Define the network creation

inputSize = 32*32
numConvFilters = 50
numLayers = 1
layerSize = 400
numLabels = 10

-- Dropout
convDropout = 0.3
dropout = 0.5

local function create_mnist_net(arch, args)
  if arch ~= 'mnist_conv' then return nil end
  if args == nil then args = {} end

  local conv_layers = 4
  if args.conv_layers ~= nil then conv_layers = args.conv_layers end

  local module = nn.Sequential()
  module:add(nn.SpatialConvolution(1, numConvFilters, 5, 5, 1, 1, 2))
  module:add(nn.ReLU(false))
  module:add(nn.Dropout(convDropout))

  for i = 1, conv_layers do
    module:add(nn.SpatialConvolution(numConvFilters, numConvFilters, 3, 3, 1, 1, 1))
    module:add(nn.ReLU(false))
    module:add(nn.Dropout(convDropout))
  end
    
  linearInputSize = numConvFilters*inputSize
  module:add(nn.Reshape(linearInputSize))

  for i = 1, numLayers do
    if i == 1 then
      module:add(nn.Linear(linearInputSize, layerSize))
    else
      module:add(nn.Linear(layerSize, layerSize))
    end
    module:add(nn.ReLU(false))
    module:add(nn.Dropout(dropout))
  end
  module:add(nn.Linear(layerSize, numLabels))

  --module:add(nn.Linear(linearInputSize, numLabels))
  module:add(nn.LogSoftMax())
  return module
end

sid.register_arch('mnist_conv', create_mnist_net)

return mnist
