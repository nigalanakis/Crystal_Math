Installation
============

We highly recommend using **Anaconda** for its ease of package management and environment handling, as it includes numerous scientific computing packages that facilitate a smoother setup process.

Download and Install Anaconda
-----------------------------

Visit the `Anaconda Distribution page <https://www.anaconda.com/products/distribution>`_ to download and install the distribution. Please ensure you download the version that includes ``Python 3.9`` or higher.

Required Python Packages
------------------------
The following Python packages are necessary for running Crystal Math:

- ``ast``
- ``datetime``
- ``itertools``
- ``json``
- ``matplotlib``
- ``networkx``
- ``numpy``
- ``os``
- ``scipy``
- ``re``
- ``time``

These can be installed using the following command:

.. code-block:: bash

    pip install matplotlib networkx numpy scipy

Note that some packages (``ast``, ``datetime``, ``itertools``, ``json``, ``os``, ``re``, ``time``) are part of the Python Standard Library and do not need installation via pip.

Installing the CSD Python API
-----------------------------
The current version requires the installation of the CSD Python API, which is crucial for the statistical analysis phase and for retrieving molecular structure data. Due to specific installation instructions and licensing, please refer to the `official installation notes <https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html>`_ for detailed guidance. Adhere strictly to their guidelines to ensure full functionality within the CSP algorithm environment.
