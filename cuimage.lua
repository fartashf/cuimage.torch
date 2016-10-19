function cuimage.typecheck(i)
   if torch.type(i) ~= 'torch.CudaTensor' then 
      error('Input is expected to be torch.CudaTensor') 
   end
end

----------------------------------------------------------------------
-- cuimage.rgb2lab(image)
-- converts an RGB image to LAB
-- assumes sRGB input in the range [0, 1]
--
function cuimage.rgb2lab(...)
   -- arg check
   local output,input,do_gamma_expand
   local args = {...}
   if select('#',...) == 3 then
       output = args[1]
       input = args[2]
       do_gamma_expand = args[3]
   elseif select('#',...) == 2 then
       if type(args[2]) == 'boolean' then
           input = args[1]
           do_gamma_expand = args[2]
       else
           output = args[1]
           input = args[2]
       end
   elseif select('#',...) == 1 then
      input = args[1]
   else
      print(dok.usage('cuimage.rgb2lab',
                      'transforms an image from sRGB to LAB', nil,
                      {type='torch.CudaTensor', help='input image', req=true},
                      '',
                      {type='torch.CudaTensor', help='output image', req=true},
                      {type='torch.CudaTensor', help='input image', req=true}
                      ))
      dok.error('missing input', 'cuimage.rgb2lab')
   end
   cuimage.typecheck(input)

   -- resize
   output = output or input.new()
   do_gamma_expand = do_gamma_expand or false
   output:resizeAs(input)

   -- compute
   cuimage.C['cuimage_rgb2lab'](cutorch.getState(), input:cdata(), output:cdata(),
           do_gamma_expand)

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
   cuimage.typecheck(input)
   cuimage.C['cuimage_rgb2lab'](cutorch.getState(), input:cdata(), self.output:cdata(), false)
   return self.output
end

local ScaleBilinear, parent = torch.class('nn.ScaleBilinear', 'nn.Module')

function ScaleBilinear:__init(oheight, owidth)
   parent.__init(self)

   self.oheight = oheight
   self.owidth = owidth

   self.output = torch.CudaTensor()
   self.tmp = torch.CudaTensor()
   self.gradInput = torch.CudaTensor()
end

function cuimage.scale(...)
   -- arg check
   local output,input
   local args = {...}
   if select('#',...) == 4 then
      output = args[1]
      input = args[2]
      owidth = args[3]
      oheight = args[4]
   elseif select('#',...) == 3 then
      input = args[1]
      owidth = args[2]
      oheight = args[3]
   else
      print(dok.usage('cuimage.scale',
                      'scale an image with bilinear interpolation', nil,
                      {type='torch.CudaTensor', help='input image', req=true},
                      {type='number', help='output width', req=true},
                      {type='number', help='output height', req=true},
                      '',
                      {type='torch.CudaTensor', help='output image', req=true},
                      {type='torch.CudaTensor', help='input image', req=true},
                      {type='number', help='output width', req=true},
                      {type='number', help='output height', req=true}
                      ))
      dok.error('missing input', 'cuimage.scale')
   end
   cuimage.typecheck(input)

   -- resize
   output = output or input.new()
   local tmp = input.new()

   -- compute
   cuimage.C['cuimage_scaleBilinear'](cutorch.getState(), input:cdata(), output:cdata(),
               tmp:cdata(), oheight, owidth)

   return output
end

function ScaleBilinear:reset(oheight, owidth)
   self.oheight = oheight
   self.owidth = owidth
end

function ScaleBilinear:updateOutput(input)
   cuimage.typecheck(input)
   cuimage.C['cuimage_scaleBilinear'](cutorch.getState(), input:cdata(), self.output:cdata(),
               self.tmp:cdata(), self.oheight, self.owidth)
   return self.output
end

