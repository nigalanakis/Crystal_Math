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

Installing the code
-------------------
The code itself requires **no installation** of additional software packages or libraries, other than Git for obtaining the code. Simply follow the steps below to clone the repository to your local machine and run the code directly.

#. **Git**: Git is a version control system that lets you manage and keep track of your source code history. If you don't already have Git installed, you can download it from `the Git website <https://git-scm.com/downloads>`_.

Cloning the Repository
^^^^^^^^^^^^^^^^^^^^^^

Cloning a repository means making a copy of the code on your local machine. This is done via Git. To clone the repository, follow these steps:

1. Open a terminal window. On Windows, you can search for ``CMD`` or ``Command Prompt`` in your start menu. On macOS, you can open the Terminal from your Applications folder under Utilities.

2. Use the following command to clone the repository:
   
   .. code-block:: bash
   
       git clone https://github.com/nigalanakis/Crystal_Math

3. After the cloning process is complete, navigate to the newly created directory:

   .. code-block:: bash
   
       cd your-repository