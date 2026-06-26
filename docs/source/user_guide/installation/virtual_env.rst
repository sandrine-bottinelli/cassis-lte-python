Setup a virtual environment
-----------------------------

Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend you create a dedicated virtual environment in which you can install the required packages.

cassis_lte_python requires a python version >= 3.11 ; to check your python version, do: ::

   which python3

If you have an older version, we recommend using
`uv <https://docs.astral.sh/uv/>`_
to manage python versions and packages. To install uv, follow
`these instructions <https://docs.astral.sh/uv/getting-started/installation/>`_.

Create a virtual environment:

.. tabs::

   .. code-tab:: bash with uv

       uv venv --python 3.11 env_name

   .. code-tab:: bash without uv

       python3 -m venv env_name

where env_name is the name of the environment.

Activate it, depending on your shell:

.. tabs::

   .. code-tab:: bash bash

       source env_name/bin/activate

   .. code-tab:: tcsh csh/tcsh

       source env_name/bin/activate.csh


Download and install the cassis_lte_python package:

.. tabs::

   .. code-tab:: bash with uv

       uv pip install cassis_lte_python-0.3.0-py3-none-any.whl

   .. code-tab:: bash without uv

       python3 -m pip install cassis_lte_python-0.3.0-py3-none-any.whl

(if the whl file is in the current directory, otherwise use the appropriate path to this file)

IDE (e.g., spyder, VScode)
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are working with an IDE, you need to specify the location of the
Python executable, which is something like ``/path/to/env_name/bin/python`` ; for example :

* Spyder : go to Preferences > Python interpreter ; check the "Use the following interpreter:" radio button and
  enter the appropriate path.

* Visual Studio Code : got to View > Command Palette ; select the "Python: Select Interpreter" command (see the
  `VScode documentation <https://code.visualstudio.com/docs/python/environments#_working-with-python-interpreters>`_),
  choose a project or the workspace level, click on "Enter interpreter path..." and type in the appropriate path.

Jupyter notebook
^^^^^^^^^^^^^^^^

Start jupyter notebook from within your virtual environment.
