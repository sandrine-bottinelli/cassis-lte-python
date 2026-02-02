.. _usecase_la:

Inspect a dataset
==================

This mode works in the same way as the Line Analysis module in CASSIS.
For a given species, the program searches for transitions within the observed range(s).
Only transitions within the spectroscopic constraints provided by the user are selected.
The transitions are displayed over the bandwidth provided by the user.

Overview of the configuration keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


+-----------+-----------------------+----------------------------------------------------+
| Mandatory | :confval:`data_file`  | path to the datafile                               |
+           +-----------------------+----------------------------------------------------+
|           | :confval:`inspect`    | list of tags                                       |
+           +-----------------------+----------------------------------------------------+
|           | :confval:`bandwidth`  | width in km/s of the window around each transition |
+-----------+-----------------------+----------------------------------------------------+
| Optional  | :confval:`franges_ghz`| list of frequency ranges to inspect                |
+           +-----------------------+----------------------------------------------------+
|           | :confval:`thresholds` |  thresholds on the spectroscopic parameters        |
+-----------+-----------------------+----------------------------------------------------+

Mandatory keys
----------------

* :confval:`data_file`
* :confval:`inspect`
* :confval:`bandwidth`

Optional keys
----------------

* :confval:`franges_ghz`
* :confval:`thresholds`
* :confval:`plot_gui`
* :confval:`plot_pdf`
* :confval:`v_range`
* options for the display (to come)

Workflow
^^^^^^^^

* The program reads the file given with :confval:`data_file`
* If :confval:`franges_ghz` is provided, data are selected over these frequency ranges
* For the list of tags provided by :confval:`inspect`, the program searches the list of transitions (eventually matching the constraints provided by :confval:`thresholds`)
* It creates a list of windows centered on the line's frequency and of width :confval:`bandwidth`
* If :confval:`plot_gui` is true, the windows are displayed on the screen.
* If :confval:`plot_pdf` is true, the windows are displayed in a multi-page pdf (new page for each species).
* If :confval:`v_range` is provided, the specified velocity ranges are displayed as green regions.

Examples
^^^^^^^^
