import dataclasses
from dataclasses import dataclass

@dataclass
class LRPPreset(object):
    """
    lrp_method LRP method to use in all but the "special" layers
    lrp_method_input LRP method to use in the input layer
    lrp_method_linear LRP method to use in the Linear layers
    """
    lrp_method: str = "epsilon"
    lrp_method_input: str = "epsilon"
    lrp_method_linear: str = "epsilon"
    lrp_method_batchnorm: str ="alphabetax"# "alphabetax"
    lrp_method_relu: str = 'identity'
    lrp_params: dict = dataclasses.field(default_factory=dict)

@dataclass
class SequentialPresetA(LRPPreset):
    lrp_method: str = "alpha_beta"
    lrp_method_input: str = "alpha_beta"
    lrp_method_linear: str = "epsilon"
    lrp_method_relu: str = 'alpha_beta'
    lrp_method_batchnorm: str = "identity"
    # TODO find nicer solution to update dict
    def __post_init__(self):
        self.lrp_params["alpha"] = 1
        self.lrp_params["beta"] = 0
        self.lrp_params["ignore_bias"] = True

