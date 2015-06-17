require 'torch'
require 'nn'
local sid = require 'sid'

-- Create an instance of the test framework
local mytester = torch.Tester()
local precision_mean = 1e-3
local test = {}

local function create_module(arch, args)
  if arch == 'test_sid' then
    local module = nn.Sequential()
    local lin1 = nn.Linear(3, 3)
    lin1.weight:fill(1)
    lin1.bias:fill(2)
    module:add(lin1)
    module:add(nn.ReLU(false))
    local lin2 = nn.Linear(3, 3)
    lin2:share(lin1, 'weight', 'bias')
    module:add(lin2)
    module:add(nn.LogSoftMax())
    return module
  end
  error(string.format('unknown arch=%s', arch))
end

local function check_simple_create(use_cuda)
  local title = string.format('check_simple_create(use_cuda=%s): ', use_cuda)
  local dog = sid.create('test_sid', nil, use_cuda)

  mytester:assertne(dog.module, nil, title .. title .. 'module vs nil')
  mytester:assertne(dog.params, nil, title .. title .. 'params vs nil')
  mytester:assertne(dog.grad_params, nil, title .. 'grad_params vs nil')

  -- We share weights and biases, but not grads.
  mytester:asserteq(dog.params:storage():size(), 12, title .. 'params:size()')
  mytester:asserteq(dog.grad_params:storage():size(), 24, title .. 'grad_params:size()')

  local want_params = torch.FloatTensor({1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2})
  mytester:assertTensorEq(dog.params:float(), want_params, precision_mean, title .. 'params value')
end

local function check_create_with_params(use_cuda)
  local title = string.format('check_create_with_params(use_cuda=%s): ', use_cuda)
  local orig_params = torch.FloatTensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
  if use_cuda then orig_params = orig_params:cuda() end
  local dog = sid.create('test_sid', nil, use_cuda, orig_params)

  mytester:assertne(dog.module, nil, title .. 'module vs nil')
  mytester:assertne(dog.params, nil, title .. 'params vs nil')
  mytester:assertne(dog.grad_params, nil, title .. 'grad_params vs nil')
  mytester:asserteq(dog.params, orig_params, title .. 'params pointer')

  -- We share weights and biases, but not grads.
  mytester:asserteq(dog.params:storage():size(), 12, title .. 'params:size()')
  mytester:asserteq(dog.grad_params:storage():size(), 24, title .. 'grad_params:size()')

  -- Check that orig_params have been used
  local want_params = torch.FloatTensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
  mytester:assertTensorEq(dog.params:float(), want_params, precision_mean, title .. 'params value')
  local lin1 = dog.module.modules[1]
  mytester:assertTensorEq(lin1.weight:float(), torch.FloatTensor({1, 2, 3, 4, 5, 6, 7, 8, 9}),
                          precision_mean, title .. 'lin1.weight')
  mytester:assertTensorEq(lin1.bias:float(), torch.FloatTensor({10, 11, 12}),
                          precision_mean, title .. 'lin1.bias')

  -- Check that weights and biases are really shared
  local lin2 = dog.module.modules[3]
  mytester:assertTensorEq(lin1.weight, lin2.weight, precision_mean, title .. 'lin1 vs lin2, weight')
  mytester:assertTensorEq(lin1.bias, lin2.bias, precision_mean, title .. 'lin1 vs lin2, bias')
  mytester:asserteq(lin1.weight:storage(), lin2.weight:storage())

end

function test.create()
  -- Checks that we can create a new module given an arch name and args.
  check_simple_create(false)
  check_simple_create(true)

  -- Checks that we can create a module given an arch name and args, and then load given params.
  check_create_with_params(false)
  check_create_with_params(true)
end

local function check_save_and_load(use_cuda)
  local title = string.format('check_load_and_save(use_cuda=%s): ', use_cuda)
  local filename = os.tmpname()
  local arch = 'test_sid'
  local args = nil
  local dog = sid.create(arch, args, use_cuda)
  torch.save(filename, sid.to_save(dog))
  local obj = torch.load(filename)
  local dog2 = sid.load_from(obj, use_cuda)

  mytester:asserteq(dog.arch, dog2.arch, title .. 'arch')
  mytester:asserteq(dog.args, dog2.args, title .. 'args')
  mytester:assertTensorEq(dog.params, dog2.params, precision_mean, title .. 'params')
end

function test.save_and_load()
  check_save_and_load(false)
  check_save_and_load(true)
end

-- Register our test arch
sid.register_arch('test_sid', create_module)

-- Now run the test above
mytester:add(test)
mytester:run()
