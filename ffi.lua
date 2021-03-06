-- from cuda-convnet2
local ffi = require 'ffi'

ffi.cdef[[
    void cuimage_rgb2lab(THCState *state, THCudaTensor *input,
            THCudaTensor *output, const bool do_gamma_expand);
    void cuimage_scaleBilinear(THCState *state, THCudaTensor *input,
            THCudaTensor *output, THCudaTensor *tmp,
            const int nOutputRows, const int nOutputCols);
]]

local path = package.searchpath('libcuimage', package.cpath)
if not path then
   path = require 'cuimage.config'
end
assert(path, 'could not find libcuimage.so')
cuimage.C = ffi.load(path)
