import argparse

import matplotlib.pyplot as plt
from metrics import *
from colorizers import *
import math 

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default='images/mycoco_val2017/000000002532.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                    help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True, local_pth="models/final_colorizer_eccv16.pth").eval()
colorizer_siggraph17 = siggraph17(pretrained=True, local_pth="models/final_colorizer_siggraph17.pth").eval()
if (opt.use_gpu):
    colorizer_eccv16.cuda()
    colorizer_siggraph17.cuda()

img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
if (opt.use_gpu):
    tens_l_rs = tens_l_rs.cuda()

img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1)).squeeze(0)
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu()).squeeze(0)
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu()).squeeze(0)

def PSNR(pred, target):
    pred = torch.from_numpy(pred.copy()).float()  
    target = torch.from_numpy(target.copy()).float()
    mse = torch.mean((pred - target) ** 2)
    psnr = abs(10 * math.log10(1 / math.sqrt(mse.item())))
    return psnr

##################################################################################
# Plot of the results of both methods
##################################################################################

plt.imsave('%s_eccv16.png' % opt.save_prefix, out_img_eccv16)
plt.imsave('%s_siggraph17.png' % opt.save_prefix, out_img_siggraph17)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(out_img_siggraph17)

plt.title('Output (SIGGRAPH 17)')
plt.axis('off')
plt.savefig("result.png")

print("PSNR ECCV16: ", round(PSNR(out_img_eccv16, img),2))
print("PSNR SIGGRAPH17: ", round(PSNR(out_img_siggraph17, img),2))

##################################################################################
# Plot of the results of each method individually
##################################################################################

"""                         ECCV16                         """

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_bw)
plt.title('GrayScale Input')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(out_img_eccv16)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img)
plt.title('Predicted')
plt.axis('off')
plt.savefig("result_eccv16.png")

"""                        SIGGRAPH17                      """

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_bw)
plt.title('GrayScale Input')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(out_img_siggraph17)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img)
plt.title('Predicted')
plt.axis('off')

plt.savefig("result_siggraph17.png")
