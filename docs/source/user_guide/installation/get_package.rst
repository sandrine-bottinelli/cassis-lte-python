Download and install the package
================================

Download the `.whl` file from the "Assets" drop-down item of the
`latest release <https://github.com/sandrine-bottinelli/cassis-lte-python/releases>`_.

Install the cassis_lte_python package:

.. tabs::

   .. code-tab:: bash with uv

       uv pip install cassis_lte_python-*-py3-none-any.whl

   .. code-tab:: bash without uv

       python3 -m pip install cassis_lte_python-*-py3-none-any.whl

Where * replaces a number of characters representing the version.

NB: the above commands assume the whl file is in the current directory,
otherwise use the appropriate path to this file)

If you plan on using jupyter notebook, install the package with the notebook option :


.. tabs::

   .. code-tab:: bash with uv

       uv pip install "cassis_lte_python-*-py3-none-any.whl[notebook]"

   .. code-tab:: bash without uv

       python3 -m pip install "cassis_lte_python-*-py3-none-any.whl[notebook]"
