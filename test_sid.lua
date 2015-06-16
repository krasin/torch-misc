require 'sid'

-- Create an instance of the test framework
local mytester = torch.Tester()
local precision_mean = 1e-3
local test = {}

function test.create()
end

-- Now run the test above
mytester:add(test)
mytester:run()
