function culab.typecheck(i)
   if torch.type(i) ~= 'torch.CudaTensor' then 
      error('Input is expected to be torch.CudaTensor') 
   end
end

----------------------------------------------------------------------
-- culab.rgb2lab(image)
-- converts an RGB image to LAB
-- assumes sRGB input in the range [0, 1]
--
function culab.rgb2lab(...)
   -- arg check
   local output,input
   local args = {...}
   if select('#',...) == 2 then
      output = args[1]
      input = args[2]
   elseif select('#',...) == 1 then
      input = args[1]
   else
      print(dok.usage('culab.rgb2lab',
                      'transforms an image from sRGB to LAB', nil,
                      {type='torch.CudaTensor', help='input image', req=true},
                      '',
                      {type='torch.CudaTensor', help='output image', req=true},
                      {type='torch.CudaTensor', help='input image', req=true}
                      ))
      dok.error('missing input', 'culab.rgb2lab')
   end
   culab.typecheck(input)

   -- resize
   output = output or input.new()
   output:resizeAs(input)

   -- compute
   culab.C['culab_rgb2lab'](cutorch.getState(), input:cdata(), output:cdata())

   -- return LAB image
   return output
end

local RGB2Lab, parent = torch.class('nn.RGB2Lab', 'nn.Module')

function RGB2Lab:__init()
   parent.__init(self)

   self.output = torch.CudaTensor()
   self.gradInput = torch.CudaTensor()
end

function RGB2Lab:updateOutput(input)
   culab.typecheck(input)
   culab.C['culab_rgb2lab'](cutorch.getState(), input:cdata(), self.output:cdata())
   return self.output
end
