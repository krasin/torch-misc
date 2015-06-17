require 'torch'
require 'nn'
local mnist = require 'mnist'

-- Create an instance of the test framework
local mytester = torch.Tester()
local precision_mean = 1e-3
local test = {}

local function create_module(arch, args)
  if arch == 'test_sid' then
    local module = nn.Sequential()
    module:add(nn.Linear(3, 3))
    module:add(nn.LogSoftMax())
    return module
  end
  error(string.format('unknown arch=%s', arch))
end

local function check_save_and_load(use_cuda)
  local title = string.format('check_load_and_save(use_cuda=%s): ', use_cuda)
  local filename = os.tmpname()
  local title = string.format('check_simple_create(use_cuda=%s): ', use_cuda)
  local arch = 'test_sid'
  local args = nil
  local state = mnist.State(use_cuda)
  state.mean = 30
  state.std = 2
  state:create_new('mnist_conv', nil)
  state:save(filename)
  local state2 = mnist.load(filename)

  mytester:asserteq(state.dog.arch, state2.dog.arch, title .. 'arch')
  mytester:asserteq(state.dog.args, state2.dog.args, title .. 'args')
  mytester:assertTensorEq(state.dog.params, state2.dog.params, precision_mean, title .. 'params')
  mytester:asserteq(state.mean, state2.mean, title .. 'mean')
  mytester:asserteq(state.std, state2.std, title .. 'mean')
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
