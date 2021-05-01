import torch
import torch.nn as nn
import torch.nn.functional as F
from . import lrp_modules, lrp_presets

def get_lrp_hook(lrp_method, lrp_params=None):
    def lrp_hook(module, relevance_input, relevance_output):
        lrp_module = lrp_modules.get_lrp_module(module)
        return lrp_module.propagate_relevance(module, relevance_input, relevance_output, lrp_method, lrp_params=lrp_params)
    return lrp_hook

def save_input_hook(module, input_, output):
    module.input = input_

class LRPLoss(nn.Module):
    # Dummy loss to provide anchor for LRP recursion
    def forward(self, x):
        return x

    def backward(self, x):
        return x

def add_lrp(model, preset=None):
    if preset is None:
        preset = lrp_presets.LRPPreset()

    # Override default parameters if provided
    lrp_method_curr = preset.lrp_method_input
    for module in model.modules():
        # Take only the leaf modules
        num_modules = len(list(module.children()))
        if num_modules == 0:
            if type(module) == nn.Linear:
                lrp_method_curr = preset.lrp_method_linear
            if type(module) in [nn.BatchNorm2d, nn.BatchNorm1d]:
                lrp_method_curr = preset.lrp_method_batchnorm
            if type(module) == nn.ReLU:
                lrp_method_curr = preset.lrp_method_relu
            # print(type(module))
            module.register_forward_hook(save_input_hook)
            module.register_backward_hook(
                get_lrp_hook(lrp_method_curr, lrp_params=preset.lrp_params))
            # for param in module.parameters():
            #     print(param.grad_fn)

            # Use lrp_method_input for first layer, lrp_method otherwise
            lrp_method_curr = preset.lrp_method

    # Add LRP computation as member function for convenience
    model.compute_lrp = lambda sample, **kwargs: compute_lrp(model, sample, **kwargs)

def compute_lrp(model, sample, target=None, return_output=False, rectify_logits=False, explain_diff=False):
    # target: list of class labels
    if sample.requires_grad==False:
        sample.requires_grad = True  # We need to compute LRP until input layer
    # sample.register_hook(get_tensor_hook)
    criterion = LRPLoss()
    logits = model(sample)
    # print(logits.shape)
    logits_plus = F.relu(logits) if rectify_logits else logits

    if explain_diff:
        if logits_plus.shape[1] != 2:
            raise ValueError(
                "Explanation of difference only possible for two classes, not for {}".format(logits_plus.shape[1]))

    loss = criterion(logits_plus)
    anchor = target
    model.zero_grad()
    sample.retain_grad()

    loss.backward(anchor, retain_graph=True)
    assert sample.grad.sum()!=0
    if return_output:
        return sample.grad, logits
    else:
        return sample.grad

# TODO remove hooks again to restore model to original state
def remove_lrp(model):
    # TODO can be done like this:
    preset = lrp_presets.LRPPreset()
    lrp_method_curr = preset.lrp_method_input
    for module in model.modules():
        # Take only the leaf modules
        num_modules = len(list(module.children()))
        if num_modules == 0:
            if type(module) == nn.Linear:
                lrp_method_curr = preset.lrp_method_linear
            if type(module) == nn.BatchNorm2d:
                lrp_method_curr = preset.lrp_method_batchnorm
            h = module.register_forward_hook(save_input_hook)
            h.remove()
            h = module.register_backward_hook(
                get_lrp_hook(lrp_method_curr, lrp_params=preset.lrp_params))
            h.remove()
            lrp_method_curr = preset.lrp_method

# TODO move to add_lrp?
def preprocess_model(model):
    # Merge BatchNorm with preceding convolutional layer
    module_prev = None
    for name, module in model.named_modules():
        if type(module) == nn.BatchNorm2d:
            # TODO handle case when BN is before conv
            assert type(module_prev) == nn.Conv2d, type(module_prev)
            bn_weight = module.weight
            bn_bias = module.bias
            bn_mean = module.running_mean
            bn_var = module.running_var + module.eps

            # Merge parameters of BatchNorm and Conv and write new parameters to Conv layer
            new_conv_weight = bn_weight.view(-1, 1, 1, 1) * module_prev.weight / torch.sqrt(bn_var.view(-1, 1, 1, 1))
            if module_prev.bias is not None:
                new_conv_bias = (module_prev.bias - bn_mean) * bn_weight / torch.sqrt(bn_var) + bn_bias
                module_prev.bias.data = new_conv_bias
            else:
                # Equivalent to bias = 0
                new_conv_bias = - bn_mean * bn_weight / torch.sqrt(bn_var) + bn_bias
                module_prev.bias = nn.Parameter(new_conv_bias)

            module_prev.weight.data = new_conv_weight

            # Make BatchNorm layer do nothing
            module.weight.data = torch.ones_like(bn_weight)
            module.bias.data = torch.zeros_like(bn_bias)
            module.eps = 0.0
            module.reset_running_stats()
        module_prev = module