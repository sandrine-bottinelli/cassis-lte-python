Update your PYTHONPATH
----------------------

Make sure that the parent directory of cassis_lte_python (called cassis-lte-python unless you renamed it) is in your PYTHONPATH.

You can do so temporarily within python : ::

    import sys
    sys.path.append('/path/to/the/directory/cassis-lte-python')

or more "permanently" :

* in your .bashrc : ::

    export PYTHONPATH="${PYTHONPATH}:/path/to/the/directory/cassis-lte-python"

* in your .cshrc or .tcshrc : ::

    setenv PYTHONPATH $PYTHONPATH:/path/to/the/directory/cassis-lte-python

