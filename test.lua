require 'cuimage'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')
im=image.load('test.jpg')
imlab=image.rgb2lab(im)
im_cuda=torch.CudaTensor()
im_cuda:resize(im:size()):copy(im)
imlab_cuda = cuimage.rgb2lab(im_cuda, true)
print(torch.sum(torch.pow(torch.add(imlab, -imlab_cuda:typeAs(imlab)),2)))

imscale=image.scale(im,60,80,'bilinear')
imscale_cuda = cuimage.scale(im_cuda,60, 80)
print(torch.sum(torch.pow(torch.add(imscale, -imscale_cuda:typeAs(imlab)),2)))
