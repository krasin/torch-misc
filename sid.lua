-- this file provides functions to save and load trainable models.
-- work in progress, anything and everything could be changed.

require 'torch'

local sid = {}

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

-- Creates a trainable model of the specified arch and args.
-- If params are specified, it will also fill the model with them.
-- The function returns model, params, grad_params.
-- Note: it will reuse the instance of params, if they were specified in the call of sid.create.
function sid.create(arch, args, use_cuda, params)
  function create_new(arch, args)
    if arch == 'lala' then
      local mlp = nn.Sequential()
      mlp:add(nn.Linear(10, 4))
      mlp:add(nn.ReLU(false))
      mlp:add(nn.Linear(4, 2))
      mlp:add(nn.LogSoftMax())
      return mlp
    end
    error(string.format('sid.create: unknown arch=%s.', arch))
  end
  local def_tensor_type = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  local model = create_new(arch, args)
  torch.setdefaulttensortype(def_tensor_type)
  if use_cuda then
    require 'cutorch'
    require 'cunn'
    model:cuda()
  end
  local params, grad_params = sid.load_params(model, params)
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
function sid.load_params(model, params)
  local params_list, grad_params_list = model:parameters()

  -- Returns max index used in the underlying storage.
  function get_max_index(tensor)
    local res = tensor:storageOffset()
    for i = 1, tensor:dim() do
      res = res + (tensor:size(i)-1)*tensor:stride(i)
    end
    return res
  end

  -- Convert a list of tensors with possibly shared storages into a single tensor.
  -- If vals are specified, this is the tensors and values to use.
  function flatten(vals_list, vals)
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

    -- 3. Prepare the target storage.
    local targetTensor = nil
    if vals == nil then
      targetTensor = vals_list[1].new(targetSize)
      targetTensor:zero()
    else
      if vals:storage():size() ~= targetSize then
        error(string.format('Failed to flatten values. Want a storage with %d elements' +
              ', but a storage with %d elements provided.', targetSize, vals:storage():size()))
      end
      targetTensor = torch.reshape(vals, targetSize)
    end

    -- 4. Set target storage to the value tensors.
    for i = 1, #vals_list do
      local val = vals_list[i]
      local key = torch.pointer(val:storage())
      local chunk = chunks[key]
      val:set(targetTensor:storage(), chunk.targetOffset + val:storageOffset() - chunk.min_index,
              val:size(), val:stride())
    end

    return targetTensor
  end

  local flat_params = flatten(params_list, params)
  local flat_grad_params = flatten(grad_params_list)
  return flat_params, flat_grad_params
end

return sid
