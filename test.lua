require 'culab'
require 'image'

im=image.load('test.jpg')
imlab=image.rgb2lab(im)
im_cuda=torch.CudaTensor()
im_cuda:resize(im:size()):copy(im)
imlab_cuda = culab.rgb2lab(im_cuda)
print(torch.sum(torch.pow(torch.add(imlab, -imlab_cuda:typeAs(imlab)),2)))
