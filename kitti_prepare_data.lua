-- This script prepares training and testing data for KITTI stereo.
-- See details in the paper:
-- 'Computing the Stereo Matching Cost with a Convolutional Neural Network',
-- Jure Zbontar, Yann LeCun
-- http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1B_053.pdf

require 'torch'
require 'image'
require 'lfs'

function extract_snippets(base_dir, fname, snippets, perm, index)
  local left = image.load(base_dir .. '/image_0/' .. fname, 1, 'byte');
  local right = image.load(base_dir .. '/image_1/' .. fname, 1, 'byte');
  local disp = image.load(base_dir .. '/disp_noc/' .. fname, 1, 'float');
  disp = disp:mul(65535/256)

  local count = 0
  for y = 9, left:size(2) - 9 do
    for x = 9, left:size(3) - 17 do
      local x2 = x - disp[1][y][x]
      if disp[1][y][x] > 0 and x2 > 17 then
        local left_snip = left:narrow(2, y-4, 9):narrow(3, x-4, 9)
        count = count + 1
        if snippets ~= nil then
	  -- Generate negative example.
          -- Random shift will be inside [-8;-4] U [4; 8]
	  local shift = torch.uniform(-4, 4)
          if shift > 0 then
            shift = shift + 4
	  else
	    shift = shift - 4
          end
          local x3 = math.floor(x2 + shift)
          local right_snip = right:narrow(2, y-4, 9):narrow(3, x3-4, 9)
          snippets[1][perm[index+count-1]]:copy(left_snip[1])
          snippets[2][perm[index+count-1]]:copy(right_snip[1])
          snippets[3][perm[index+count-1]]:zero()
        end
        count = count + 1
        if snippets ~= nil then
          -- Generate positive example (+/- 1 px)
          if torch.uniform() > 0.5 then
	    x2 = math.floor(x2-0.5)
          else
            x2 = math.ceil(x2+0.5)
          end
          local right_snip = right:narrow(2, y-4, 9):narrow(3, x2-4, 9)
          snippets[1][perm[index+count-1]]:copy(left_snip[1])
          snippets[2][perm[index+count-1]]:copy(right_snip[1])
          snippets[3][perm[index+count-1]]:zero()
          snippets[3][perm[index+count-1]][1] = 1
        end
      end
    end
  end
  collectgarbage()
  return count
end

function extract_all_snippets(base_dir, snippets, perm)
  local path_noc = base_dir .. '/disp_noc'
  local total = 0
  for fname in lfs.dir(path_noc) do
    if fname ~= "." and fname ~= ".." then
      local f = path_noc .. '/' .. fname
      local attr = lfs.attributes(f)
      if attr.mode == "file" and string.match(fname, '[.]png$') then
        local count = extract_snippets(base_dir, fname, snippets, perm, total+1)
        total = total + count
        print(string.format('%s - %d snippets (%d total)', fname, count, total))
      end
    end
  end
  return total
end

function prepare_data(base_dir, dest_prefix)
  print('Estimating the number of snippets...')
  local expected_total = extract_all_snippets(base_dir)
  print(string.format('%s snippets found', expected_total))

  local perm = torch.randperm(expected_total)
  local file_name = string.format('%s.3x%dx9x9.byte_tensor', dest_prefix, expected_total)
  local file_size = 3 * expected_total * 9 * 9

  print(string.format('Creating a storage backed by %s ...', file_name))
  local snippets_storage = torch.ByteStorage(file_name, true, file_size)
  print('Storage created. Creating a tensor ...')

  local snippets = torch.ByteTensor(snippets_storage, 1, torch.LongStorage({3, expected_total, 9, 9}))
  print('snippets: ', snippets:size())

  local total = extract_all_snippets(base_dir, snippets, perm)
  if expected_total == total then
    print('Expectations and reality match - OK')
  else
    print('Unexpected number of snippets - Error')
  end
end

prepare_data('data/kitti_stereo/training', '/mnt/kitti_stereo/snippets')
