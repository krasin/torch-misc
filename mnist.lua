require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
sid = require 'sid'

-- Whether to use CUDA, -1: use CPU, >=0: use corresponding GPU
gpuid = 0
if gpuid >= 0 then
    use_cuda = true
    print('using CUDA on GPU ' .. gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end

-- Load MNIST data
trainFile = 'data/mnist.t7/train_32x32.t7'
testFile = 'data/mnist.t7/test_32x32.t7'
trainData = torch.load(trainFile,'ascii')
trainData.data = trainData.data:float()
trainData.labels = trainData.labels:float()
valData = { data=trainData.data[{{50001, 60000}}],
            labels=trainData.labels[{{50001, 60000}}] }
trainData.data = trainData.data[{{1, 50000}}]
trainData.labels = trainData.labels[{{1, 50000}}]
testData = torch.load(testFile,'ascii')
testData.data = testData.data:float()
testData.labels = testData.labels:float()

-- Preprocess train, val and test data.

-- 1. Calculate mean for train data and subtract the mean from train, val and test data.
mean = trainData.data:mean()
print('mean: ', mean)
trainData.data:add(-mean)
valData.data:add(-mean)
testData.data:add(-mean)

-- 2. Calculate std deviation for train data and divide by it
std = trainData.data:std()
print('std: ', std)
trainData.data:div(std)
valData.data:div(std)
testData.data:div(std)

print('Train data:')
print(trainData.labels[{{1, 6}}])
print("size: ", trainData.data:size(), trainData.labels:size())
print()

print('Test data:')
print(testData.data:size())
print(testData.labels[{{1, 6}}])
print()

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

function create_mnist_net(arch, args)
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

-- Create a dog to train.
dog = sid.create('mnist_conv', nil, use_cuda)

criterion = nn.ClassNLLCriterion()
if use_cuda then
    criterion:cuda()
end

print('params: ', dog.params:size(), dog.params:type())
print('gradParams: ', dog.grad_params:size(), dog.grad_params:type())

-- initialization
dog.params:uniform(-0.08, 0.08) -- small numbers uniform

-- Training facilities
reg = 1000 / dog.params:size(1) -- L2 regularization strength
gradClip = 5

batchSize = 50
maxBatch = trainData.data:size(1) / batchSize
--maxBatch = maxBatch / 10 -- test for overfit; TODO: remove before commit.
print('maxBatch: ', maxBatch)
curBatch = 1

function feval(x)
    if x ~= dog.params then
        dog.params:copy(x)
    end
    dog.grad_params:zero()
    ------------------ get minibatch -------------------
    local batchStart = (curBatch-1)*batchSize + 1
    local batchEnd = batchStart + batchSize - 1
    curBatch = curBatch + 1
    if curBatch > maxBatch then
        curBatch = 1
    end
    --local x = torch.reshape(trainData.data[{{batchStart, batchEnd}}], batchSize, inputSize)
    local x = trainData.data[{{batchStart, batchEnd}}]
    local y = trainData.labels[{{batchStart, batchEnd}}]
    if use_cuda then
        x = x:float():cuda()
        y = y:float():cuda()
    end

    ------------------- forward pass -------------------
    dog.module:training() -- make sure we are in correct mode 
    local prediction = dog.module:forward(x)
    local paramNorm = dog.params:norm()
    local loss = criterion:forward(prediction, y) + reg * paramNorm * paramNorm / 2

    ------------------ backward pass -------------------
    local dprediction = criterion:backward(prediction, y)
    dog.module:backward(x, dprediction)
    
    dog.grad_params:add(reg, dog.params) -- apply regularization gradient
    dog.grad_params:clamp(-gradClip, gradClip)
    return loss, dog.grad_params
end

loss, _ = feval(dog.params)
print('loss: ', loss)

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
    local _, loss = optim.rmsprop(feval, dog.params, optimState)
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

function predict(input)
    --print ('input: ', input:size())
    --local x = torch.reshape(input, input:size(1), inputSize)
    local x = input
    if gpuid >= 0 then
        x = x:float():cuda()
    end
    dog.module:evaluate() -- turn off dropout
    local prediction = dog.module:forward(x)
    local _, classes = prediction:max(2)
    return classes
end
classes = predict(trainData.data[{{1, 2}}])
print("predicted classes: ", classes)
print("ground truth: ", trainData.labels[{{1, 2}}])

function evalAccuracy(input, labels)
    local matches = 0
    local batchSize = 1000
    local from = 1
    for i = 1, input:size(1) do
        if i - from + 1 >= batchSize or i == input:size(1) then
            --print ('i=', i, ' from: ', from)
            local curLabels = labels[{{from, i}}]
            local predictions = predict(input[{{from, i}}], curLabels):float()
            --print ('predictions: ', predictions:size(), predictions:type())
            curLabels:map(predictions, function(xx, yy) if xx == yy then matches = matches + 1 end end)
            from = i+1
        end
    end
    
    return matches / labels:size(1)
end

sid.save(string.format('mnist-%s.nn', os.time()), dog)

trainAcc = evalAccuracy(trainData.data, trainData.labels)
print('train accuracy: ', trainAcc)

valAcc = evalAccuracy(valData.data, valData.labels)
print('validation accuracy: ', valAcc)

-- testAcc = evalAccuracy(testData.data, testData.labels)
-- print('test accuracy: ', testAcc)

