require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
sid = require 'sid'
local class = require 'class'


local State = class('State')

function State:__init(use_cuda)
  self.use_cuda = use_cuda
end

-- Loads and preprocesses train/val/test data.
function State:load_data(dir)
  -- Load MNIST data
  local trainFile = dir .. '/train_32x32.t7'
  local testFile = dir .. '/test_32x32.t7'
  self.trainData = torch.load(trainFile,'ascii')
  self.trainData.data = self.trainData.data:float()
  self.trainData.labels = self.trainData.labels:float()
  self.valData = { data=self.trainData.data[{{50001, 60000}}],
                    labels=self.trainData.labels[{{50001, 60000}}] }
  self.trainData.data = self.trainData.data[{{1, 50000}}]
  self.trainData.labels = self.trainData.labels[{{1, 50000}}]
  self.testData = torch.load(testFile,'ascii')
  self.testData.data = self.testData.data:float()
  self.testData.labels = self.testData.labels:float()

  -- Preprocess train, val and test data.

  -- 1. Calculate mean for train data and subtract the mean from train, val and test data.
  self.mean = self.trainData.data:mean()
  self.trainData.data:add(-self.mean)
  self.valData.data:add(-self.mean)
  self.testData.data:add(-self.mean)

  -- 2. Calculate std deviation for train data and divide by it
  self.std = self.trainData.data:std()
  self.trainData.data:div(self.std)
  self.valData.data:div(self.std)
  self.testData.data:div(self.std)

  print('Train data:')
  print(self.trainData.labels[{{1, 6}}])
  print("size: ", self.trainData.data:size(), self.trainData.labels:size())
  print()

  print('Test data:')
  print(self.testData.data:size())
  print(self.testData.labels[{{1, 6}}])
  print()

  self:init_train()
end

function State:init_train()
  -- Training facilities
  self.reg = 1000 / self.dog.params:size(1) -- L2 regularization strength
  self.gradClip = 5

  self.batchSize = 50
  self.maxBatch = self.trainData.data:size(1) / self.batchSize
  print('maxBatch: ', self.maxBatch)
  self.curBatch = 1
end

function State:feval(x)
    local dog = self.dog
    if x ~= dog.params then
        dog.params:copy(x)
    end
    dog.grad_params:zero()
    ------------------ get minibatch -------------------
    local batchStart = (self.curBatch-1)*self.batchSize + 1
    local batchEnd = batchStart + self.batchSize - 1
    self.curBatch = self.curBatch + 1
    if self.curBatch > self.maxBatch then
        self.curBatch = 1
    end
    local x = self.trainData.data[{{batchStart, batchEnd}}]
    local y = self.trainData.labels[{{batchStart, batchEnd}}]
    if self.use_cuda then
        x = x:float():cuda()
        y = y:float():cuda()
    end

    ------------------- forward pass -------------------
    dog.module:training() -- make sure we are in correct mode 
    local prediction = dog.module:forward(x)
    local paramNorm = dog.params:norm()
    local loss = self.criterion:forward(prediction, y) + self.reg * paramNorm * paramNorm / 2

    ------------------ backward pass -------------------
    local dprediction = self.criterion:backward(prediction, y)
    dog.module:backward(x, dprediction)
    
    dog.grad_params:add(self.reg, dog.params) -- apply regularization gradient
    dog.grad_params:clamp(-self.gradClip, self.gradClip)
    return loss, dog.grad_params
end

-- Create a dog to train and a criterion.
function State:create_new()
  local dog = sid.create('mnist_conv', nil, use_cuda)
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

function State:evalAccuracy(input, labels)
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

function State:save(filename)
  local dog_obj = sid.to_save(self.dog)
  local checkpoint = {
    dog = dog_obj,
    mean = self.mean,
    std = self.std
  }
  torch.save(filename, checkpoint)
end

-- Define the network creation

inputSize = 32*32
convLayers = 3
numConvFilters = 50
numLayers = 1
layerSize = 400
numLabels = 10

-- Dropout
convDropout = 0.3
dropout = 0.5

local function create_mnist_net(arch, args)
    if arch ~= 'mnist_conv' then return nil end

    local module = nn.Sequential()
    module:add(nn.SpatialConvolution(1, numConvFilters, 5, 5, 1, 1, 2))
    module:add(nn.ReLU(false))
    module:add(nn.Dropout(convDropout))

    for i = 1, convLayers do
        module:add(nn.SpatialConvolution(numConvFilters, numConvFilters, 3, 3, 1, 1, 1))
        module:add(nn.ReLU(false))
        module:add(nn.Dropout(convDropout))
    end
    
    linearInputSize = numConvFilters*inputSize
    module:add(nn.Reshape(linearInputSize))

    --for i = 1, numLayers do
    --    if i == 1 then
    --        mlp:add(nn.Linear(linearInputSize, layerSize))
    --    else
    --        mlp:add(nn.Linear(layerSize, layerSize))
    --    end
    --    mlp:add(nn.ReLU(false))
    --    mlp:add(nn.Dropout(dropout))
    --end
    --mlp:add(nn.Linear(layerSize, numLabels))

    module:add(nn.Linear(linearInputSize, numLabels))
    module:add(nn.LogSoftMax())
    return module
end

sid.register_arch('mnist_conv', create_mnist_net)

-- Whether to use CUDA, -1: use CPU, >=0: use corresponding GPU
gpuid = 0
if gpuid >= 0 then
    use_cuda = true
    print('using CUDA on GPU ' .. gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end

state = State(use_cuda)
state:create_new()
state:load_data('data/mnist.t7')

print('mean: ', state.mean)
print('std: ', state.std)

optimState = {learningRate = 0.0003, alpha = 0.9}
learningRateDecay = 0.9
learningRateDecayAfter = 10

iterations = 2000

epoch = 0
minLoss = 10
maxLoss = 0
sumLoss = 0
lossCnt = 0

for i = 1, iterations do
    if curBatch == 1 then
        epoch = epoch + 1
        if epoch >= learningRateDecayAfter then
            optimState.learningRate = optimState.learningRate * learningRateDecay
        end
        --print(string.format('Starting epoch %d, lr: %f', epoch, optimState.learningRate))
    end
    local _, loss = optim.rmsprop(function(x) return state:feval(x) end, state.dog.params, optimState)
    trainLoss = loss[1]
    if trainLoss < minLoss then minLoss = trainLoss end
    if trainLoss > maxLoss then maxLoss = trainLoss end
    sumLoss = sumLoss + trainLoss
    lossCnt = lossCnt + 1
    if i == 1 or i % 1000 == 0 then
        print(string.format('epoch=%d, i=%d, train loss: %f .. %f .. %f, lr: %f',
                epoch, i, minLoss, sumLoss / lossCnt, maxLoss, optimState.learningRate))
        minLoss = 10
        maxLoss = 0
        sumLoss = 0
        lossCnt = 0
    end
end

classes = state:predict(state.trainData.data[{{1, 2}}])
print("predicted classes: ", classes)
print("ground truth: ", state.trainData.labels[{{1, 2}}])

state:save(string.format('mnist-%s.nn', os.time()))

trainAcc = state:evalAccuracy(state.trainData.data, state.trainData.labels)
print('train accuracy: ', trainAcc)

valAcc = state:evalAccuracy(state.valData.data, state.valData.labels)
print('validation accuracy: ', valAcc)

-- testAcc = state:evalAccuracy(state.testData.data, state.testData.labels)
-- print('test accuracy: ', testAcc)
