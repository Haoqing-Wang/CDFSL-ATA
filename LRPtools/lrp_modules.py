import torch
import torch.nn as nn
import LRPtools.utils as util
from methods.backbone import Add as resAdd
from methods.backbone import Flatten as resFlatten

class Linear:
    def propagate_relevance(self, module, relevance_input, relevance_output, lrp_method, lrp_params=None):
        ignore_bias = lrp_params.get("ignore_bias", True)
        input_ = module.input[0]
        input_.masked_fill_(input_ == 0, util.RELEVANCE_RECT)
        V = module.weight.clone().detach()
        Z = torch.mm(input_.clone().detach(), V.t())
        if ignore_bias:
            Z += util.EPSILON * Z.sign()  # Z.sign() returns -1 or 0 or 1
            Z.masked_fill_(Z == 0, util.EPSILON)
        if not ignore_bias:
            Z += module.bias.clone().detach()
        S = relevance_output[0].clone().detach() / Z
        C = torch.mm(S, V)
        R = input_ * C
        assert R.shape == input_.shape
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        module.zero_grad()
        if len(relevance_input) == 3:
            assert relevance_input[0].shape == module.bias.shape
            assert relevance_input[1].shape == R.shape
            assert relevance_input[2].shape == module.weight.t().shape
            return relevance_input[0], R, relevance_input[2]
        elif len(relevance_input) == 2:
            assert relevance_input[0].shape == R.shape
            assert relevance_input[1].shape == module.weight.t().shape
            return R, relevance_input[1]

class ReLU:
    def propagate_relevance(self, module, relevance_input, relevance_output, lrp_method, lrp_params=None):
        if lrp_method == 'identity':
            R = relevance_output[0].clone().detach()
            assert not torch.isnan(R.sum())
            assert not torch.isinf(R.sum())
            return(relevance_output[0],)
        else:
            v_input = module.input[0]
            mask = torch.where(v_input>0, torch.full_like(v_input,1), torch.full_like(v_input, 0))
            R = relevance_output[0].clone().detach()
            R = R*mask
            assert not torch.isnan(R.sum())
            assert not torch.isinf(R.sum())
            return (R,)

class PosNetConv(nn.Module):
    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                          **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
        super(PosNetConv, self).__init__()

        self.posconv = self._clone_module(conv)
        self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0),requires_grad=False)

        self.negconv = self._clone_module(conv)
        self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0),requires_grad=False)

        if ignorebias == True:
            self.posconv.bias = None
            self.negconv.bias = None
        else:
            if conv.bias is not None:
                self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0),requires_grad=False)
                self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0),requires_grad=False)


    def forward(self, x):
        vp = self.posconv(torch.clamp(x, min=0))
        vn = self.negconv(torch.clamp(x, max=0))
        return vp + vn

class NegNetConv(nn.Module):
    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
        super(NegNetConv, self).__init__()

        self.posconv = self._clone_module(conv)
        self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0),requires_grad=False)

        self.negconv = self._clone_module(conv)
        self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0),requires_grad=False)

        if ignorebias == True:
            self.posconv.bias = None
            self.negconv.bias = None
        else:
            if conv.bias is not None:
                self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0),requires_grad=False)
                self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0),requires_grad=False)

    def forward(self, x):
        vp = self.posconv(torch.clamp(x, max=0))
        vn = self.negconv(torch.clamp(x, min=0))
        return vp + vn

class Conv2d:
    def _clone_module(self, module):

        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                         **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):

        ignore_bias = lrp_params.get("ignore_bias", True)
        input_ = module.input[0]
        if lrp_method =="alpha_beta":
            pnconv = PosNetConv(module, ignore_bias)
            nnconv = NegNetConv(module, ignore_bias)
            with torch.enable_grad():
                X = input_.clone().detach().requires_grad_(True)

                # Positive contribution
                R_pos = lrp_params["alpha"] * (
                        util.lrp_backward(_input=X, layer=pnconv, relevance_output=relevance_output[0])
                )
                # Clear gradients
                X.grad.detach_()
                X.grad.zero_()

                # Negative contribution
                R_neg = lrp_params["beta"] * (
                    util.lrp_backward(_input=X, layer=nnconv, relevance_output=relevance_output[0])
                )
                R = R_pos - R_neg

                del pnconv
                del nnconv
        else:
            raise NotImplementedError('Only adopt alpha 1 rule for conv layer')
        assert R.shape == input_.shape
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        module.zero_grad()
        if len(relevance_input) == 3:
            assert relevance_input[0].shape == R.shape
            assert relevance_input[1].shape == module.weight.shape
            if module.bias is None:
                assert relevance_input[2] is None
            else:
                assert relevance_input[2].shape == module.bias.shape
            torch.cuda.empty_cache()
            return R, relevance_input[1], relevance_input[2]
        elif len(relevance_input) == 2:
            assert relevance_input[0].shape == R.shape
            assert relevance_input[1].shape == module.weight.shape
            torch.cuda.empty_cache()
            return R, relevance_input[1]

class Pool2d:
    def _clone_module(self, module):
        if type(module) == nn.MaxPool2d:
            clone = nn.MaxPool2d(module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']})
        elif type(module) == nn.AvgPool2d:
            clone = nn.AvgPool2d(module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'count_include_pad', 'ceil_mode']})
        else:
            raise ValueError(type(module))
        return clone

    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        input_ = module.input[0]
        module_clone = self._clone_module(module)
        with torch.enable_grad():
            X = input_.clone().detach().requires_grad_(True)
            Z = module_clone(X)
            S = util.safe_divide(relevance_output[0].clone().detach(), Z)
            Z.backward(S)
            R = X * X.grad
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        module.zero_grad()
        return (R, )

class BatchNorm2d:
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        # todo how is batchnorm handled by default in innvestigate?
        if lrp_method == 'identity':
            R = relevance_output[0]
        else:
            input_ = module.input[0]
            mean = module._buffers['running_mean']
            var = module._buffers['running_var']
            gamma = module._parameters['weight']
            beta = module._parameters['bias']

            w = (gamma / torch.sqrt(var + module.eps))[:, None, None]
            b = (beta - (mean * gamma) / torch.sqrt(var + module.eps))[:, None, None]
            xw = input_ * w

            R = util.safe_divide(torch.abs(xw), (torch.abs(xw) + torch.abs(b))) * \
                relevance_output[0]
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        assert R.sum() !=0
        module.zero_grad()
        return R, relevance_input[1], relevance_input[2]

class BatchNorm1d:
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        # todo how is batchnorm handled by default in innvestigate?
        if lrp_method == 'identity':
            R = relevance_output[0]

        else:
            input_ = module.input[0]
            mean = module._buffers['running_mean']
            var = module._buffers['running_var']
            gamma = module._parameters['weight']
            beta = module._parameters['bias']

            w = (gamma / torch.sqrt(var + module.eps))[:, None, None]
            b = (beta - (mean * gamma) / torch.sqrt(var + module.eps))[:, None, None]
            xw = input_ * w

            R = util.safe_divide(torch.abs(xw), (torch.abs(xw) + torch.abs(b))) * \
                relevance_output[0]
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        assert R.sum() !=0
        module.zero_grad()
        return R, relevance_input[1], relevance_input[2]

class Dropout:
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        # TODO how to handle this?
        assert ((relevance_output[0] - relevance_input[0]).abs().max() < 1e-7).cpu().item() == 1
        module.zero_grad()
        return relevance_input

class Add:
    def _clone_module(self, module):
        clone = resAdd()
        return clone.to(module.weight.device)
    def propagate_relevance(self, module, relevance_input, relevance_output, lrp_method, lrp_params=None):
        input_1, input_2 = module.input[0].detach(), module.input[1].detach()
        out = input_1 + input_2
        mask = out == 0
        out_mask = torch.zeros_like(out).masked_fill_(mask, 0.5)

        out += util.EPSILON * out.sign()
        rele_out = relevance_output[0].clone().detach()
        R_mask = rele_out * out_mask
        R1 =  rele_out * input_1 / out
        R2 = rele_out * input_2 / out
        R1[R1!=R1] = 0
        R2[R2!=R2] = 0
        R1 += R_mask
        R2 += R_mask
        assert not torch.isnan(R1.sum())
        assert not torch.isnan(R2.sum())
        assert not torch.isinf(R1.sum())
        assert not torch.isinf(R2.sum())
        return R1, R2

class Flatten:
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        v_input = module.input[0]
        v_size = v_input.size()
        R = relevance_output[0].clone().detach()
        R = R.view(v_size)
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        return (R,)

def compute_lrp_sum(sum_output, sum_input, relevance_sum_output, dim=-1):  # 400  (400, 512)  (400)
    assert (sum_output == torch.sum(sum_input, dim=dim)).all()
    fea_dim = sum_input.size()[-1]  # 512
    relevance = relevance_sum_output.unsqueeze(-1).repeat(1, fea_dim)  # (400, 512)
    out = sum_output.unsqueeze(-1).repeat(1,fea_dim)  # (400, 512)
    mask = out==0
    out.masked_fill_(mask, 1./fea_dim)
    relevance_sum_input = relevance * sum_input / (out + util.EPSILON * out.sign())
    return relevance_sum_input

def compute_lrp_mean(mean_output, mean_input, relevance_mean_output, dim=-1):
    assert (mean_output == torch.mean(mean_input, dim=dim)).all()
    fea_dim = mean_input.size()[-1]
    input_dim = len(mean_input.shape)
    repeat_param = [1]*input_dim
    repeat_param[-1] *= fea_dim

    relevance = relevance_mean_output.unsqueeze(-1).repeat(repeat_param)
    out = mean_input.sum(dim=dim).unsqueeze(-1).repeat(repeat_param)
    mask = out==0
    out.masked_fill_(mask, 1./fea_dim)
    relevance_mean_input = relevance * mean_input / (out + util.EPSILON * out.sign())
    return relevance_mean_input

def get_lrp_module(module):
    try:
        lrp_module_class = {
            nn.Linear: Linear,
            nn.ReLU: ReLU,
            nn.Conv2d: Conv2d,
            nn.MaxPool2d: Pool2d,
            nn.AvgPool2d: Pool2d,  # TODO
            nn.BatchNorm2d: BatchNorm2d,
            nn.BatchNorm1d: BatchNorm1d,
            nn.Dropout: Dropout,
            nn.Dropout2d: Dropout,
            resFlatten: Flatten,
            resAdd:Add,
        }[type(module)]
    except KeyError:
        ###print(type(module))
        raise ValueError("Layer type {} not known.".format(type(module)))

    lrp_module = lrp_module_class()
    return lrp_module