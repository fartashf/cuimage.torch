-- from cuda-convnet2
local ffi = require 'ffi'

ffi.cdef[[
    void culab_rgb2lab(THCState *state, THCudaTensor *input, THCudaTensor *output);
]]

local path = package.searchpath('libculab', package.cpath)
if not path then
   path = require 'culab.config'
end
assert(path, 'could not find libculab.so')
culab.C = ffi.load(path)
