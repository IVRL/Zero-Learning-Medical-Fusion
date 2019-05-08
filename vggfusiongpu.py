import math
import numbers
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.vgg import vgg19
import cv2

def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)/255.
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    return im_ycbcr

def ycbcr2rgb(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycrcb = im_ycbcr[:,:,(0,2,1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return im_rgb

def get_activation(model, input_image):
    outs = []
    out = input_image
    out = model.features[0](out)
    out = model.features[1](out)
    l1out = torch.sum(out, dim=1, keepdim=True)
    outs.append(l1out)
    return outs, out

def softmax(feats, with_exp=False):
    if with_exp:
        feats = torch.exp(feats)
    feats = feats / (feats.sum(dim=1, keepdim=True) + 1e-3)
    return feats

def fusion_strategy(feats, sources, with_exp=False):
    feats = torch.cat(feats, dim=1)
    D = softmax(F.interpolate(feats, size=sources[0].shape[2:]), with_exp)
    D = F.interpolate(D, size=sources[0].shape[2:])
    final = torch.zeros(sources[0].shape).cuda()
    for idx, source in enumerate(sources):
        final += source * D[:,idx]
    return final, D

def to_pytorch(image):
    np_input = image.astype(np.float32)
    if np_input.ndim == 2:
        np_input = np.repeat(np_input[None, None], 3, axis=1)
    else:
        np_input = np.transpose(np_input, (2, 0, 1))[None]
    return torch.from_numpy(np_input).cuda()

def _fuse(inputs, model=None, with_exp=False):   
    with torch.no_grad():
        if model is None:
            model = vgg19(True)
        model.cuda().eval()

        tc_inputs = []
        relus_acts = []
        for input_img in inputs:
            tc_input = to_pytorch(input_img)

            relus_act, out = get_activation(model, tc_input)
            
            tc_inputs.append(tc_input)
            relus_acts.append(relus_act)

        saliency_max = None
        idx = 0
        for relus_list in zip(*relus_acts):
            saliency_current, D = fusion_strategy(relus_list, tc_inputs, with_exp)
            idx += 1
            if saliency_max is None:
                saliency_max = saliency_current
            else:
                saliency_max = torch.max(saliency_max, saliency_current)

        output = np.squeeze(saliency_max.cpu().numpy())
        if output.ndim == 3:
            output = np.transpose(output, (1, 2, 0))
        return output
    
def fuse(inputs, model=None, with_exp=None):
    im1, im2 = inputs
    if im2.ndim == 3:
        ycbcr = rgb2ycbcr(im2)
        i2 = ycbcr[:,:,0]
    else:
        i2 = im2 / 255.

    fT = _fuse([im1/255., i2], model, with_exp=False)

    fT = fT[:,:,0]
    if im2.ndim==3:
        ycbcr[:,:,0] = fT
        fT = ycbcr2rgb(ycbcr)
        fT = np.clip(fT, 0, 1)
    fT = (fT*255).astype(np.uint8)
    return fT    