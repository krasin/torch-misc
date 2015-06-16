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
  local model, params, grad_params = sid.create('test_sid', nil, use_cuda)

  mytester:assertne(model, nil, title .. 'model not nil')
  mytester:assertne(params, nil, title .. 'params not nil')
  mytester:assertne(grad_params, nil, title .. 'grad_params not nil')
  mytester:asserteq(params:storage():size(), 12, title .. 'params:size()')
  mytester:asserteq(grad_params:storage():size(), 24, title .. 'grad_params:size()')

  local want_params = torch.FloatTensor({{{1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2}}})
  mytester:assertTensorEq(params:float(), want_params, precision_mean, title .. 'params value')
end

function test.create()
  -- Register our test arch
  sid.register_arch('test_sid', create_module)

  check_simple_create(false)
  check_simple_create(true)
end

-- Now run the test above
mytester:add(test)
mytester:run()
