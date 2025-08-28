.. _usecase_minimize:

Minimize a set of parameters in a single spectrum
====================================================

Overview of the configuration keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each key, a short description is given ; for more information, click on the link (name of the key).

Mandatory keys
----------------

* :confval:`data_file` : path to the datafile
* :confval:`minimize` : in this case must be ``True``
* :confval:`components` : a file or a dictionary with the parameters of the components
* :confval:`tuning_info` : a dictionary used to specify which telescope(s) to use over a given (list of) frequency range(s);
  if the ranges are smaller
* :confval:`rms_cal` : a file or a dictionary providing noise and calibration information
* (:confval:`v_range` and :confval:`bandwidth`) or :confval:`fit_freq_except` ; see `Workflow`_

Recommended keys
----------------

* :confval:`output_dir` : the path to the directory for writing output files.

Optional keys
----------------

* :confval:`thresholds` : thresholds on the spectroscopic parameters
* :confval:`snr_threshold` : when provided, only line with a modeled signal-to-noise ratio greater than this value will be saved.
* :confval:`oversampling` : oversampling of the model (for display purposes)
* :confval:`line_shift_kms` : if the data are in sky frequency, use this value to search for lines outside the observed range.
* :confval:`tc` : value(s) for the continuum
* :confval:`tcmb` : temperature of the CMB
* :confval:`max_iter` : maximum number of iterations
* :confval:`plot_gui`
* :confval:`plot_pdf`
* :confval:`v_range`
* options for the display (to come)


Workflow
^^^^^^^^^

* Components setup : for each component, the program finds the list of species
* The program reads the file given with :confval:`data_file`
* If :confval:`franges_ghz` is provided, data are selected over these frequency ranges
* For a given species, the program searches for transitions within the range(s) specified with :confval:`tuning_info`. For display purposes, only transitions within the spectroscopic constraints provided by the user are selected.

  .. warning:: The constraints are only used to select lines to be displayed. The model will always use all the transitions found within a certain frequency range, regardless of the values of their spectroscopic parameters.

* Data points used for the fit are:

  * either within the velocity ranges provided by the user

  * *or* all points except those in some (small) frequency windows


Example
^^^^^^^

