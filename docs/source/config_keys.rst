Accepted keys in the configuration directory
=============================================

.. confval:: bandwidth
    :type: ``float``

    The bandwidth for the display, in km/s.

.. confval:: components
    :type: ``dict``

    A dictionary giving the information on the components.

.. confval:: data_file
    :type: ``str``

    The path to the data file.

.. confval:: df_mhz
    :type: ``float``

    The frequency spacing in MHz when computing a model alone.

.. confval:: fit_freq_except
    :type: ``list``

    A list of 2-element lists containing the min and max frequencies to avoid when fitting.

    .. note::
        This is a mandatory keyword when fitting an entire spectrum. It can be an empty list.

.. confval:: franges_ghz
    :type: ``list``

    A list of 2-element lists containing the min and max frequencies for all ranges the user wants to select.

.. confval:: inspect
    :type: ``list``

    The list of tags of the species of interest.

.. confval:: line_shift_kms
    :type: ``float``

    For data in sky frequency scale, the velocity shift in km/s to take into account when searching transitions.

.. confval:: max_iter
    :type: ``int``

    The maximum number of iterations for the minimization.
    Default depends on the chosen algorithm.

.. confval:: minimize
    :type: ``bool``

    Whether to minimize or not.

.. confval:: noise_mk
    :type: ``float``
    :default: 0.

    The noise in mK when computing a model alone.

.. confval:: output_dir
    :type: ``str``

    The path to the directory where the files created by the program will be saved.

.. confval:: oversampling
    :type: ``int`` or ``float``

    Oversampling used to **display** the model.
    Default : 1

.. confval:: plot_gui
    :type: ``bool``
    :default: True

    Whether to display on screen.

.. confval:: plot_pdf
    :type: ``bool``
    :default: False

    Whether to create a pdf file for the display.

.. confval:: rms_cal

.. confval:: snr_threshold
    :type: ``int`` or ``float``

    The cut-off in signal-to-noise ratio (SNR) to save a given transition in a CASSIS linelist.
    The SNR is computed on the modeled intensities.

.. confval:: tc
    :type: ``float`` or ``str``
    :default: 0.0

    The continuum in data units. Can be a single value of a two-column file with frequency and continuum values.

.. confval:: tcmb
    :type: ``float``
    :default: 2.73

    The temperature in K of the CMB.

.. confval:: thresholds
    :type: ``dict`` or ``str``
    :default: no thresholds

    A dictionary or a path to a valid file containing the thresholds on spectroscopic parameters (see :ref:`thresholds`).

.. confval:: tuning_info
    :type: ``dict``

    A dictionary providing the lists of frequency ranges (values) for the telescope(s) used in the observations (keys).

.. confval:: v_range
    :type: ``dict`` or ``str``

    A dictionary or a path to a valid file containing velocity ranges to be displayed (see :ref:`v_range`).

