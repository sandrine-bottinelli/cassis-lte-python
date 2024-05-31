import numpy as np
from lmfit import Parameter


def parameter_infos(value=None, min=None, max=None, expr=None, vary=True,
                    factor=False, difference=False):
    if factor and difference:
        raise KeyError("Can only have factor=True OR difference=True")
    user_data = {}
    if factor and value is not None:
        user_data['factor'] = True
        if min is not None:
            user_data['min_fact'] = min
            min *= value
        if max is not None:
            user_data['max_fact'] = max
            max *= value
    if difference and value is not None:
        user_data['difference'] = True
        if min is not None:
            user_data['min_diff'] = min
            min += value
        if max is not None:
            user_data['max_diff'] = max
            max += value
    if len(user_data) > 0:
        return {'value': value, 'min': min, 'max': max, 'expr': expr, 'vary': vary, 'user_data':user_data}
    else:
        return {'value': value, 'min': min, 'max': max, 'expr': expr, 'vary': vary}


def create_parameter(name, param):
    if isinstance(param, (float, int)):
        return Parameter(name, value=param)

    elif isinstance(param, dict):
        # if bounds are equal, fix parameter and reset bounds (otherwise lmfit complains)
        if param.get('min', -np.inf) == param.get('max', np.inf):
            param['vary'] = False
            param['min'] = -np.inf
            param['max'] = np.inf
        return Parameter(name, **param)

    elif isinstance(param, Parameter):
        return param

    else:
        raise TypeError(f"{name} must be a float, an integer, a dictionary or an instance of the Parameter class.")
