import numpy as np
# from cassis_lte_python.utils.utils import format_float
# from lmfit import Parameter
from lmfit import Parameter as lmfitParameter


# class Parameter(lmfitParameter):
#     def __init__(self, name, value=None, vary=True, min=-np.inf, max=np.inf,
#                  expr=None, brute_step=None, user_data=None):
#         super().__init__(name, value=value, vary=vary, min=min, max=max,
#                  expr=expr, brute_step=brute_step, user_data=user_data)
#
#         if any([var in self.name for var in ['size', 'fwhm', 'ntot']]):
#             self.abs_min = 0.
#         elif 'tex' in self.name:
#             self.abs_min = 2.73
#         else:
#             self.abs_min = min
#
#         self.abs_max = max
#
#         self.check_bounds()
#
#     def set(self, value=None, vary=None, min=None, max=None, expr=None,
#             brute_step=None, is_init_value=True):
#         super().set(value=value, vary=vary, min=min, max=max, expr=expr,
#                     brute_step=brute_step, is_init_value=is_init_value)
#         self.check_bounds()
#
#     def check_bounds(self):
#         self.min = self.abs_min if not isinstance(self.min, (float, int)) else max(self.min, self.abs_min)
#         self.max = self.abs_max if not isinstance(self.max, (float, int)) else min(self.max, self.abs_max)


class Parameter:
    def __init__(self, name: str, value: float | int, vary=True, min=-np.inf, max=np.inf,
                 expr=None, factor=False, difference=False, user_data=None):

        if factor and difference:
            raise KeyError("Can only have factor=True OR difference=True")

        self._name = name
        self._value = value
        self._init_value = value
        self._vary = vary
        self._expr = expr
        self._stderr = None

        self._min = min
        self._max = max

        self.factors = None
        self.diffs = None
        self.user_data = user_data

        if user_data is not None:
            if factor in user_data:
                factor = user_data['factor']
                min = user_data['min_fact']
                max = user_data['max_fact']
            if difference in user_data:
                difference = user_data['difference']
                min = user_data['min_diff']
                max = user_data['max_diff']

        if factor:
            self.factors = np.array([min, max])
            self._min = self.value * self.factors[0]
            self._max = self.value * self.factors[1]
        elif 'ntot' in self._name:
            self.factors = np.array([min, max]) / self.value

        if difference:
            self.diffs = np.array([-np.abs(min), max])
            self._min = self.value + self.diffs[0]
            self._max = self.value + self.diffs[1]
        elif 'ntot' not in self._name:
            self.diffs = np.array([min, max]) - self.value

        if any([var in self._name for var in ['size', 'fwhm', 'ntot']]):
            self._abs_min = 0.
        elif 'tex' in self._name:
            self._abs_min = 2.73
        else:
            self._abs_min = -np.inf

        self._abs_max = np.inf

    def __repr__(self):
        from cassis_lte_python.utils.utils import format_float
        res = []
        sval = f"value={format_float(self._value)}"
        if not self._vary and self._expr is None:
            sval += " (fixed)"
        elif self.stderr is not None:
            sval += f" +/- {format_float(self.stderr)}"
        res.append(sval)
        res.append(f"bounds=[{format_float(self.min)}:{format_float(self.max)}]")
        if self._expr is not None:
            res.append(f"expr='{self.expr}'")
        return f"<Parameter '{self.name}', {', '.join(res)}>"

    def to_dict(self):
        return {'value': self.value, 'min': self.min, 'max': self.max, 'expr': self.expr, 'vary': self.vary,
                'user_data': self.user_data}

    def set(self, value=None, min=None, max=None, expr=None, vary=None, stderr=None,
            abs_min=None, abs_max=None, init=False, param=None):
        inputs = vars()
        inputs.pop('self')
        inputs.pop('init')
        # param = None
        # atts = ['value', 'min', 'max', 'expr', 'vary', 'stderr']
        atts = ['value', 'stderr']
        if param is not None:
            for att in atts:
                try:
                    setattr(self, att, getattr(param, att))
                except AttributeError:
                    print(f'Could not set {att} from {param}, ignoring.')
        else:
            for arg, val in inputs.items():
                if val is not None:
                    setattr(self, arg, val)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        if self.factors is not None or self.diffs is not None:
            try:
                mini, maxi = self.factors * value  # assume factors is not None
            except TypeError:
                mini, maxi = self.diffs + value
            self.min = mini
            self.max = maxi

    @property
    def stderr(self):
        return self._stderr

    @stderr.setter
    def stderr(self, value):
        self._stderr = value

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, value):
        self._expr = value

    @property
    def vary(self):
        return self._vary

    @vary.setter
    def vary(self, value):
        self._vary = value

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        self._min = value if value >= self._abs_min else self._abs_min

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self._max = value if value <= self._abs_max else self._abs_max

    @property
    def abs_min(self):
        return self._abs_min

    @abs_min.setter
    def abs_min(self, value):
        self._abs_min = value

    @property
    def abs_max(self):
        return self._abs_max

    @abs_max.setter
    def abs_max(self, value):
        self._abs_max = value


def parameter_infos(value=None, min=None, max=None, expr=None, vary=True,
                    factor=False, difference=False):
    if factor and difference:
        raise KeyError("Can only have factor=True OR difference=True")
    user_data = {}
    if factor and value is not None:
        user_data['factor'] = True
        if min is not None and expr is None:
            user_data['min_fact'] = min
            min *= value
        if max is not None and expr is None:
            user_data['max_fact'] = max
            max *= value
    if difference and value is not None:
        user_data['difference'] = True
        if min is not None and expr is None:
            user_data['min_diff'] = min
            min += value
        if max is not None and expr is None:
            user_data['max_diff'] = max
            max += value
    if len(user_data) > 0:
        return {'value': value, 'min': min, 'max': max, 'expr': expr, 'vary': vary, 'user_data': user_data}
    else:
        return {'value': value, 'min': min, 'max': max, 'expr': expr, 'vary': vary}


def create_parameter(name, param):
    if isinstance(param, (float, int)):
        return lmfitParameter(name, value=param)

    elif isinstance(param, dict):
        # if bounds are equal, fix parameter and reset bounds (otherwise lmfit complains)
        if param.get('min', -np.inf) == param.get('max', np.inf):
            param['vary'] = False
            param['min'] = -np.inf
            param['max'] = np.inf
        return lmfitParameter(name, **param)

    elif isinstance(param, lmfitParameter):
        return param

    else:
        raise TypeError(f"{name} must be a float, an integer, a dictionary or an instance of the Parameter class.")
