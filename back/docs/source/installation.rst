Installation
============

This section explains how to set up Crystal_Math on your system.

Clone the repository
--------------------

.. code-block:: bash

   $ git clone https://github.com/nigalanakis/Crystal_Math.git
   $ cd Crystal_Math

Create a Python 3.9 environment
--------------------------------

We recommend using Conda to isolate dependencies:

.. code-block:: bash

   $ conda create -n crystalmath python=3.9 -y
   $ conda activate crystalmath

Install dependencies
--------------------

All required packages are listed in **requirements.txt**. Install them with pip:

.. code-block:: bash

   $ pip install -r requirements.txt

Minimum **requirements.txt**:

.. code-block:: text

   numpy
   pandas
   h5py
   networkx
   torch
   ccdc    # CCDC Python API; requires a valid CSD license

Verify the installation
-----------------------

Run the main entry-point to confirm everything is working:

.. code-block:: bash

   $ python src/csa_main.py --help

You should see usage instructions and available options.

Next Steps
----------

After successfully installing, move on to :doc:`usage` for examples and quickstart guidance.
