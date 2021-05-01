import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage.transform
LOWEST = -1
HIGHEST = 1
EPSILON = 0.01
Z_EPSILON = 1e-7
LOGIT_BETA = 4
RELEVANCE_RECT = -1e-6

def safe_divide(numerator, divisor):
    # Save divide in iNNvestigate: a / (b + iK.to_floatx(K.equal(b, K.constant(0))) * K.epsilon())
    return numerator / (divisor + Z_EPSILON * (divisor == 0).float())

def zero_bias(module):
    """Set the bias of the module to zero."""
    if module.bias is not None:
        module.bias = nn.Parameter(torch.zeros_like(module.bias))

def zero_grad_tensor(p):
    """Sets gradient of the input tensor to zero, in line with
    torch.nn.Module.zero_grad"""
    p.grad.detach_()
    p.grad.zero_()

def lrp_backward(_input, layer, relevance_output):
    """
    Performs the LRP backward pass, implemented as standard forward and backward passes.
    """
    relevance_output = relevance_output.clone().detach()
    with torch.enable_grad():
        Z = layer(_input)
        S = safe_divide(relevance_output, Z)
        Z.backward(S)
        relevance_input = _input * _input.grad
    return relevance_input

def project(X, output_range=(0, 1), absmax=None, input_is_postive_only=False):

    if absmax is None:
        absmax = np.max(np.abs(X),
                        axis=tuple(range(1, len(X.shape))))
    absmax = np.asarray(absmax)
    # print(absmax.shape)
    # print(X.shape)
    mask = absmax != 0
    # print(mask)
    if mask.sum() > 0:
        X[mask] /= absmax[mask]

    if input_is_postive_only is False:
        X = (X+1)/2  # [0, 1]
    X = X.clip(0, 1)

    X = output_range[0] + (X * (output_range[1]-output_range[0]))
    return X

def normalize_relevance(X, dim=-1, temperature=1):
    '''temperature controls the range of the lrp weight'''
    value, indice = torch.max(torch.abs(X), dim=dim)
    value.masked_fill_(value==0, 1)
    X = X/value.unsqueeze(dim)

    if temperature > 1:
        return X*temperature + temperature
    else:
        return X*temperature + 1.

def heatmap(X, cmap_type="seismic", reduce_op="sum", reduce_axis=-1, **kwargs):
    cmap = plt.cm.get_cmap(cmap_type)

    tmp = X
    shape = tmp.shape

    if reduce_op == "sum":
        tmp = tmp.sum(axis=reduce_axis)
    elif reduce_op == "absmax":
        pos_max = tmp.max(axis=reduce_axis)
        neg_max = (-tmp).max(axis=reduce_axis)
        abs_neg_max = -neg_max
        tmp = np.select([pos_max >= abs_neg_max, pos_max < abs_neg_max],
                        [pos_max, neg_max])
    else:
        raise NotImplementedError()
    # print(tmp.shape)
    tmp = project(tmp, output_range=(0, 255), **kwargs).astype(np.int64)

    tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[reduce_axis] = 3
    return tmp.reshape(shape).astype(np.float32)

def graymap(X, **kwargs):
    return heatmap(X, cmap_type="gray", **kwargs)

def gamma(X, gamma = 0.5, minamp=0, maxamp=None):
    """
    apply gamma correction to an input array X
    while maintaining the relative order of entries,
    also for negative vs positive values in X.
    the fxn firstly determines the max
    amplitude in both positive and negative
    direction and then applies gamma scaling
    to the positive and negative values of the
    array separately, according to the common amplitude.

    :param gamma: the gamma parameter for gamma scaling
    :param minamp: the smallest absolute value to consider.
    if not given assumed to be zero (neutral value for relevance,
        min value for saliency, ...). values above and below
        minamp are treated separately.
    :param maxamp: the largest absolute value to consider relative
    to the neutral value minamp
    if not given determined from the given data.
    """

    #prepare return array
    # print(X.shape)
    Y = np.zeros_like(X)
    # print(Y.shape)

    X = X - minamp # shift to given/assumed center
    if maxamp is None: maxamp = np.abs(X).max() #infer maxamp if not given
    assert maxamp!=0
    X = X / maxamp # scale linearly

    #apply gamma correction for both positive and negative values.
    # try:
    i_pos = X >= 0
    print(i_pos.sum())
    if i_pos.sum() > 0:
        Y[i_pos] = X[i_pos] ** gamma
    i_neg = np.invert(i_pos)
    if i_neg.sum() > 0:

        Y[i_neg] = -(-X[i_neg])**gamma

    #reconstruct original scale and center
    if maxamp != 0:
        Y *= maxamp
    Y += minamp
    return Y
    # except RuntimeWarning:
    #     raise RuntimeError

def visuallize_attention(image, attention, reshape_size, cmap_type="seismic",):
    '''this function will blend the original highlightened by the attention
        the input images are with shape (channel, height, width) an Image object
        the attentions are with shape (1, height*width)) a pytorch tensor'''

    def project_inside(x):
        absmax = np.max(np.abs(x))
        x = 1.0 * x / absmax
        if np.sum(x < 0):
            x = (x + 1) / 2
        else:
            x = x
        return x
    attention = attention.view(reshape_size)
    attention = attention.cpu().detach().numpy()-1
    # print(attention.shape)
    attention = project_inside(attention)
    # print(attention)
    atn = skimage.transform.pyramid_expand(attention, upscale=14,
                                           multichannel=False)
    # print(atn)
    # plt.imshow(atn)
    # plt.show()
    cm = plt.get_cmap('jet')
    atn_heatmap = cm(atn)
    # print(atn_heatmap[:,:,0])
    # plt.imshow(atn_heatmap[:,:,:3])
    # plt.show()
    # print(atn.shape)
    attention_heatmap = Image.fromarray(np.uint8(atn_heatmap[:, :, :3]*255))
    # attention_heatmap.show()
    merged_heatmap = Image.blend(image, attention_heatmap, 0.5)
    return merged_heatmap

def get_class_label(image_roots, class_to_readable, num_query):
    '''image_roots is a list of tuples, len(image_roots) == n_support+n_querry,  len(image_roots[i]) == n_way'''
    n_way = len(image_roots[0])
    batch_size = len(image_roots)
    num_support = batch_size - num_query
    label_to_classlabel = {}
    query_img_path = [[]]
    query_gt_class = [[]]
    for i in range(batch_size):
        for j in range(n_way):
            image_root = image_roots[i][j]

            class_index = image_root.split('/')[-2]
            if i > 0:
                assert class_to_readable[class_index] == label_to_classlabel[j]
                if i >=num_support:
                    query_img_path[j].append(image_root)
                    query_gt_class[j].append(class_to_readable[class_index])
                if j>=len(query_img_path):
                    query_img_path.append([])
                    query_gt_class.append([])
                continue
            readable_label = class_to_readable[class_index]
            label_to_classlabel[j] = readable_label
    print(label_to_classlabel, query_img_path)
    return label_to_classlabel, query_img_path, query_gt_class

def get_xtrain_idx(labels, index):
     labels = labels.cpu().numpy()
     # print(labels)
     return np.where(labels==index)