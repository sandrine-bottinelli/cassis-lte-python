Create a virtual environment
-----------------------------

We recommend you create a dedicated virtual environment in which you can install the required packages.

Considering some packages are only available with pip, we advise using this method to manage the virtual environment.

Create a new virtual environment : ::

    python3 -m venv env_name

where env_name is the name of the environment.

Activate it : ::

    source env_name/bin/activate


or : ::

    source env_name/bin/activate.csh

(depending on your shell)

Make sure pip is up-to-date and install requirements : ::

    python3 -m pip install --upgrade pip
    python3 -m pip install requirements.txt

If you are working with an IDE, you need to specify the location of the
Python executable, which is something like ``/path/to/env_name/bin/python`` ; for example :

* Spyder : go to Preferences > Python interpreter ; check the "Use the following interpreter:" radio button and
  enter the appropriate path.

* Visual Studio Code : got to View > Command Palette ; select the "Python: Select Interpreter" command (see the
  `VScode documentation <https://code.visualstudio.com/docs/python/environments#_working-with-python-interpreters>`_),
  choose a project or the workspace level, click on "Enter interpreter path..." and type in the appropriate path.

