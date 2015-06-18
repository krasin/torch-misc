require 'torch'
require 'nn'
local mnist = require 'mnist'

-- Whether to use CUDA, -1: use CPU, >=0: use corresponding GPU
gpuid = 0
if gpuid >= 0 then
    use_cuda = true
    print('using CUDA on GPU ' .. gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end

state = mnist.State(use_cuda)
state:create_new('mnist_conv', nil)
state:load_data('data/mnist.t7')

print('mean: ', state.mean)
print('std: ', state.std)

optimState = {learningRate = 0.0003, alpha = 0.9}
learningRateDecay = 0.95
learningRateDecayAfter = 10

iterations = 2000000

epoch = 0
minLoss = 10
maxLoss = 0
sumLoss = 0
lossCnt = 0

for i = 1, iterations do
    if state.curBatch == 1 then
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
	state:save_checkpoint('checkpoints')
        minLoss = 10
        maxLoss = 0
        sumLoss = 0
        lossCnt = 0
    end
end

classes = state:predict(state.train_data.data[{{1, 2}}])
print("predicted classes: ", classes)
print("ground truth: ", state.train_data.labels[{{1, 2}}])

state:save_checkpoint('checkpoints')

print('train accuracy: ', state:train_accuracy())
print('validation accuracy: ', state:val_accuracy())

-- print('test accuracy: ', state:test_accuracy())
