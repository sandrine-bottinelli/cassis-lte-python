Formats of input dictionaries or files
======================================

Some information can be provided as a dictionary or as an input file. The formats are described here.

.. _components:

Components
----------

The information for the components is given with a dictionary, which can have one of the following format :

* if you don't have too many species, you can provide all the information directly in the dictionary ;
  it should be a dictionary of dictionaries: ::

    {
    'c1': {
        'size': parameter_infos(min=1., max=50., value=30.0, vary=True),
        'tex': parameter_infos(min=10., max=300., value=100., vary=True),
        'vlsr': parameter_infos(min=0., max=10.0, value=4.0, vary=True),
        'fwhm': parameter_infos(min=1.0, max=15.0, value=3., vary=True)
        'interacting': True,
        'species': [
           {'tag': 41505,
            'ntot': parameter_infos(min=0.001, max=1000., value=2.e16, factor=True),
            },
           {'tag': 28502,
            'ntot': parameter_infos(min=0.001, max=1000., value=2.e15, factor=True),
            },
            ]
        },
    'c2': {...}
    }


.. _thresholds:

Thresholds
----------

In all cases, you can provide information on more species than needed, the program will only look for the
species given by the user via the keyword :confval:`inspect` or :confval:`components`.

* As a dictionary, species by species: ::

    thresholds = {
        28501: {'eup_max': 150},
        28502: {'eup_max': 250, 'aij_min': 1.e-4}
    }

* As a dictionary, for all species: ::

    thresholds = {
        '*': {'eup_max': 150, 'aij_min': 1.e-4}
    }

* As a file: coming soon

.. _v_range:

Velocity ranges
---------------

.. literalinclude:: ../../examples/data-fitting/inputs/velocityRanges.txt

