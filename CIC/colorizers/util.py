
from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed

from PIL import Image

def load_img(img_path):
    img = Image.open(img_path)
    img = img.resize((256, 256))  # Resize the image
    out_np = np.asarray(img)
    if(out_np.ndim==2):
        out_np = np.tile(out_np[:,:,None],3)
    return out_np

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
	
	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]
	
	return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l_batch, out_ab_batch, mode='bilinear'):
    # tens_orig_l_batch   batch_size x 1 x H_orig x W_orig
    # out_ab_batch        batch_size x 2 x H x W

    batch_size, _, H_orig, W_orig = tens_orig_l_batch.shape
    _, _, H, W = out_ab_batch.shape

    # call resize function if needed
    if H_orig != H or W_orig != W:
        out_ab_orig = F.interpolate(out_ab_batch, size=(H_orig, W_orig), mode=mode)
    else:
        out_ab_orig = out_ab_batch

    # concatenate tensors along the channel dimension
    out_lab_orig = torch.cat((tens_orig_l_batch, out_ab_orig), dim=1)

    # convert the result to numpy and transpose dimensions
    rgb_images = color.lab2rgb(out_lab_orig.data.cpu().numpy().transpose((0, 2, 3, 1)))

    return rgb_images
