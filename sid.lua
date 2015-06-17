-- this file provides functions to save and load trainable modules.
-- work in progress, anything and everything could be changed.

require 'torch'
require 'nn'

local sid = { arch_protos = {} }
local Dog = torch.class('nn.Dog')

function Dog:__init(arch, args, params, grad_params, module)
  self.arch = arch
  self.args = args
  self.params = params
  self.grad_params = grad_params
  self.module = module
end

-- Loads a trainable model (nn.Dog) from an object previously created with sid.to_save.
function sid.load_from(obj, use_cuda)
  local params = obj.params
  if use_cuda then
    params = params:cuda()
  end
  return sid.create(obj.arch, obj.args, use_cuda, params)
end

-- Registers a new network arch. Whenever a new network is requested with the given arch name,
-- create_new(arch, args) will be passed.
function sid.register_arch(arch, create_new)
  if sid.arch_protos[arch] ~= nil then
    error(string.format('sid.register_arch(%s): the arch is already registered', arch))
  end
  sid.arch_protos[arch] = create_new
end

-- Creates a trainable model (nn.Dog) of the specified arch and args.
-- If params are specified, it will also fill the internal module with them.
-- Note: it will reuse the instance of params, if they were specified in the call of sid.create.
function sid.create(arch, args, use_cuda, params)
  if params ~= nil and use_cuda and params:type() ~= 'torch.CudaTensor' then
    error(string.format('sid.create(arch=%s, use_cuda=%s): params:type(): %s, want torch.CudaTensor',
          arch, use_cuda, params:type()))
  end

  local create_new = sid.arch_protos[arch]
  if create_new == nil then
    error(string.format('sid.create: unknown arch=%s.', arch))
  end

  local def_tensor_type = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  local module = create_new(arch, args)
  torch.setdefaulttensortype(def_tensor_type)

  if module == nil then
    error(string.format('create_new for arch=%s returned nil, instead of a module', arch))
  end

  local params, grad_params = sid.load_params(module, use_cuda, params)
  local dog = nn.Dog(arch, args, params, grad_params, module)
  return dog
end

-- sid.to_save returns an object that can be saved with torch.save and later be loaded with sid.load_from.
function sid.to_save(dog)
  if dog.params == nil then error('sid.save: dog.params == nil') end
  local checkpoint = {
    arch = dog.arch,
    args = dog.args,
    params = dog.params:float()
  }
  return checkpoint
end

-- sid.load_params takes a module and merges all parameters into a single Tensor.
-- If params is not given, a new Tensor is created.
-- sid.load returns params and grad_params.
function sid.load_params(module, use_cuda, params)
  local params_list, grad_params_list = module:parameters()

  -- Returns max index used in the underlying storage.
  function get_max_index(tensor)
    local res = tensor:storageOffset()
    for i = 1, tensor:dim() do
      res = res + (tensor:size(i)-1)*tensor:stride(i)
    end
    return res
  end

  -- Returns a list of offsets in the unified storage and the size of the required storage
  function compute_offsets(vals_list)
    -- 1. Group vals by storage and compute which continuous parts
    -- of the shared storage do they use.
    local chunks = {}
    for i = 1, #vals_list do
      local val = vals_list[i]
      local key = torch.pointer(val:storage())
      if chunks[key] == nil then
        chunks[key] = { storage = val:storage(),
                        min_index = val:storageOffset(),
                        max_index = get_max_index(val) }
      else
        chunks[key].min_index = math.min(chunks[key].min_index, val:storageOffset())
        chunks[key].max_index = math.max(chunks[key].max_index, get_max_index(val))
      end
    end

    -- 2. Compute the total size for the result tensor, and assign ranges.
    local curOffset = 1
    for i = 1, #vals_list do
      local val = vals_list[i]
      local key = torch.pointer(val:storage())
      local chunk = chunks[key]
      if chunk.targetOffset == nil then
        chunk.targetOffset = curOffset
        curOffset = curOffset + chunk.max_index - chunk.min_index + 1
      end
    end
    local targetSize = curOffset - 1


    -- 3. Compute offsets for values in the target storage
    local targetOffsets = {}
    for i = 1, #vals_list do
      local val = vals_list[i]
      local key = torch.pointer(val:storage())
      local chunk = chunks[key]
      table.insert(targetOffsets, chunk.targetOffset + val:storageOffset() - chunk.min_index)
    end

    return targetOffsets, targetSize
  end

  -- Takes a list of tensors, prepares target storage, if needed,
  -- and points these tensors to the unified storage.
  -- Returns the unified storage. If vals is specified, it becomes the unified storage.
  function flatten(use_cuda, targetSize, vals_list, vals_offsets, vals)
    local targetTensor = vals
    if targetTensor == nil then
      if use_cuda then
        targetTensor = torch.CudaTensor(targetSize)
      else
        targetTensor = vals_list[1].new(targetSize)
      end
      targetTensor:zero()

      -- Now, copy the values from vals_list to targetStorage
      for i = 1, #vals_list do
        local val = vals_list[i]
        local dest = targetTensor:new()
        dest:set(targetTensor:storage(), vals_offsets[i], val:size(), val:stride())
        dest:copy(val)
      end
    end

    for i = 1, #vals_list do
      local val = vals_list[i]
      --print('val: ', val:type(), val:size())
      --print('targetTensor: ', targetTensor:type(), targetTensor:size())
      val:set(targetTensor:storage(), vals_offsets[i], val:size(), val:stride())
    end

    return targetTensor
  end

  local params_offsets, params_size = compute_offsets(params_list)
  local grad_params_offsets, grad_params_size = compute_offsets(grad_params_list)

  if use_cuda then
     require 'cutorch'
     require 'cunn'
     module:cuda()
     params_list, grad_params_list = module:parameters()
  end

  local flat_params = flatten(use_cuda, params_size, params_list, params_offsets, params)
  local flat_grad_params = flatten(use_cuda, grad_params_size, grad_params_list, grad_params_offsets)

  return flat_params, flat_grad_params
end

return sid
