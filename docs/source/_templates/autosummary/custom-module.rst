{{ fullname | escape | underline}}


.. currentmodule:: {{ module }}

.. automodule:: {{ fullname }}
   :no-members:

Functions
---------

.. autosummary::
{% for item in functions %}
   {{ item }}
{%- endfor %}

{% for item in functions %}
.. autofunction:: {{ item }}
{% endfor %}
