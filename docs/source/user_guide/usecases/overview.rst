Overview
=========

For all cases, you need to :

* include one of the following imports : ::

    from cassis_lte_python.LTEmodel import ModelSpectrum

  or : ::

    from cassis_lte_python.LTEmodel import ModelCube

* create a configuration dictionary suited for your needs
* call ``ModelSpectrum`` or ``ModelCube`` with the configuration dictionary as argument

Currently, the following use cases are possible :

============================  =======================
Use case                      Import
============================  =======================
Inspect a dataset             ``ModelSpectrum``
Minimize a single spectrum    ``ModelSpectrum``
Minimize spectra from a cube  ``ModelCube``
Create a single model         ``ModelSpectrum``
============================  =======================

