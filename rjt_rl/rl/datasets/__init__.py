from rjt_rl.utils.config_wrapper import load_class

from .expert_dataset2 import ExpertDataset2  # NOQA


def get_dataset_class(clsnm: str) -> type:
    return load_class(clsnm, default_module_name="rjt_rl.rl.datasets")
