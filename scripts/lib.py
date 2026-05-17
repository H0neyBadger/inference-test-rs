import sys

import numpy as np


def weight_to_rust(name, data, comment=None):
    rust_type = str(data.dtype).replace("float", "f")
    for v in data.shape[::-1]:
        rust_type = f"[{rust_type}; {v}]"
    template = ""
    str_data = np.array2string(
        data,
        separator=", ",
        threshold=sys.maxsize,
        suppress_small=True,
        precision=8,
        max_line_width=120,
    )
    if comment:
        template += f"// {comment}\n"
    template += f"const {name}: {rust_type} = {str_data};"
    return template


def layer_to_rust(layer):
    ret = ""
    for idx, weight in enumerate(layer.get_weights()):
        ret += "\n" + weight_to_rust(
            f"{layer.name}_{idx}",
            weight,
            comment=f"{layer}",
        )
    return ret


def model_to_rust(model):
    ret = ""
    for layer in model.layers:
        ret += layer_to_rust(layer)
    return ret
