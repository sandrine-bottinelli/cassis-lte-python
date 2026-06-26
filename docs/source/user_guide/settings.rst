Settings
=========

cassis_lte_python needs two pieces of information :

- the path to the database (sqlite) file
- the path to the directory containing telescope files (for modeling purposes)

This information is read by cassis_lte_python from a file called settings.ini.

To use your own settings :

* download the 'settings_defaults.ini' file
* rename it 'settings.ini'
* edit it as necessary

For cassis_lte_python to find this file, your need to define a system variable
called CASSIS_LTE_PYTHON_SETTINGS. To do so, follow the instructions below depending
on your shell.

.. tabs::

   .. tab:: csh or tcsh

      In your ~/.cshrc or ~/.tcshrc, add: ::

         setenv CASSIS_LTE_PYTHON_SETTINGS /path/to/the/directory/settings.ini

   .. tab:: bash

        In your ~/.bashrc, add: ::

           export CASSIS_LTE_PYTHON_SETTINGS="/path/to/the/directory/settings.ini"

Source the shell configuration file or open a new terminal so that the above take effect.
