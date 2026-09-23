from typing import Any, Dict, no_type_check

import torch
from omegaconf import DictConfig, OmegaConf


@no_type_check
def convert_dtype_in_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dtype string to torch dtype in kwargs dictionary.

    Args:
        kwargs: Dictionary that may contain a 'dtype' key with string value

    Returns:
        Dictionary with dtype converted to torch dtype object

    Example:
        >>> kwargs = {"dtype": "bfloat16", "device_map": "auto"}
        >>> convert_dtype_in_kwargs(kwargs)
        {"dtype": torch.bfloat16, "device_map": "auto"}
    """
    if isinstance(kwargs, DictConfig):
        kwargs = OmegaConf.to_container(kwargs, resolve=True)

    if "dtype" in kwargs:
        dtype_str = kwargs["dtype"]
        if dtype_str == "bfloat16":
            kwargs["dtype"] = torch.bfloat16
        elif dtype_str == "float16":
            kwargs["dtype"] = torch.float16
        elif dtype_str == "float32":
            kwargs["dtype"] = torch.float32
        elif dtype_str == "float64":
            kwargs["dtype"] = torch.float64

    return kwargs
