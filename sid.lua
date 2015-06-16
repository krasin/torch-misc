-- this file provides functions to save and load trainable models.
-- work in progress, anything and everything could be changed.

require 'torch'

local sid = { arch_protos = {} }

-- Loads a trainable model from the file.
-- Return model, arch, args, params and grad_params.
function sid.load(filename, use_cuda)
  local checkpoint = torch.load(filename)
  if use_cuda then
    checkpoint.params:cuda()
  end
  local model, params, grad_params = sid.create(checkpoint.arch, checkpoint.args, use_cuda, checkpoint.params)
  return model, checkpoint.arch, checkpoint.args, params, grad_params
end

-- Registers a new network arch. Whenever a new network is requested with the given arch name,
-- create_new(arch, args) will be passed.
function sid.register_arch(arch, create_new)
  if sid.arch_protos[arch] ~= nil then
    error(string.format('sid.register_arch(%s): the arch is already registered', arch))
  end
  sid.arch_protos[arch] = create_new
end

-- Creates a trainable model of the specified arch and args.
-- If params are specified, it will also fill the model with them.
-- The function returns model, params, grad_params.
-- Note: it will reuse the instance of params, if they were specified in the call of sid.create.
function sid.create(arch, args, use_cuda, params)
  local create_new = sid.arch_protos[arch]
  if create_new == nil then
    error(string.format('sid.create: unknown arch=%s.', arch))
  end

  local def_tensor_type = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  local model = create_new(arch, args)
  torch.setdefaulttensortype(def_tensor_type)
  local params, grad_params = sid.load_params(model, use_cuda, params)
  return model, params, grad_params
end

-- sid.save writes a trainable model into a file
function sid.save(filename, arch, args, params)
  local checkpoint = {
    arch = arch,
    args = args,
  }
  if params ~= nil then
    checkpoint.params = params:float()
  end
  torch.save(filename, checkpoint)
end

-- sid.load_params takes a trainable model and merges all parameters into a single Tensor.
-- If params is not given, a new Tensor is created.
-- sid.load returns params and grad_params.
function sid.load_params(model, use_cuda, params)
  local params_list, grad_params_list = model:parameters()

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
        --print('New storage found: ', val:storage())
        chunks[key] = { storage = val:storage(),
                        min_index = val:storageOffset(),
                        max_index = get_max_index(val) }
      else
        --print('Detected shared storage: ', chunks[key].storage)
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
     model:cuda()
     params_list, grad_params_list = model:parameters()
  end

  local flat_params = flatten(use_cuda, params_size, params_list, params_offsets, params)
  local flat_grad_params = flatten(use_cuda, grad_params_size, grad_params_list, grad_params_offsets)
  print('flat_params: ', flat_params)
  print('flat_grad_params: ', flat_grad_params)

  return flat_params, flat_grad_params
end

return sid
