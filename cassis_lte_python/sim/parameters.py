from lmfit import Parameter


def parameter_infos(value=None, min=None, max=None, expr=None, vary=True,
                    factor=False, difference=False):
    if factor and difference:
        raise KeyError("Can only have factor=True OR difference=True")
    if factor and value is not None and min is not None:
        min *= value
        max *= value
    if difference and value is not None and max is not None:
        min += value
        max += value
    return {'value': value, 'min': min, 'max': max, 'expr': expr, 'vary': vary}


def create_parameter(name, param):
    if isinstance(param, (float, int)):
        return Parameter(name, value=param)

    elif isinstance(param, dict):
        return Parameter(name, **parameter_infos(**param))

    elif isinstance(param, Parameter):
        return param

    else:
        raise TypeError(f"{name} must be a float, an integer, a dictionary or an instance of the Parameter class.")
