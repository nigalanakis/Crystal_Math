Crystal Structure Analysis (CSA) Documentation
==============================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :alt: Python Version

.. image:: https://img.shields.io/badge/pytorch-GPU%20Accelerated-orange
   :alt: PyTorch GPU

.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: License

Welcome to the Crystal Structure Analysis (CSA) documentation. CSA is a comprehensive Python framework for extracting, processing, and analyzing molecular crystal structures from the Cambridge Structural Database (CSD).

🚀 Key Features
---------------

* **High-Performance Pipeline**: GPU-accelerated batch processing with PyTorch
* **CSD Integration**: Direct interface to Cambridge Structural Database
* **Advanced Analytics**: Fragment analysis, intermolecular contacts, and geometric descriptors
* **Efficient Storage**: HDF5-based data management with variable-length datasets
* **Scalable Architecture**: Parallel processing for large datasets

📖 Quick Navigation
-------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🏃 Getting Started
      :link: getting_started/index
      :link-type: doc

      New to CSA? Start here for installation, configuration, and your first analysis.

   .. grid-item-card:: 📚 User Guide
      :link: user_guide/index
      :link-type: doc

      Learn the core concepts and workflow of the CSA pipeline.

   .. grid-item-card:: 🎯 Tutorials
      :link: tutorials/index
      :link-type: doc

      Step-by-step tutorials for common analysis scenarios.

   .. grid-item-card:: 🔧 API Reference
      :link: api_reference/index
      :link-type: doc

      Complete API documentation for all modules and classes.

   .. grid-item-card:: 💡 Examples
      :link: examples/index
      :link-type: doc

      Ready-to-run code examples for various use cases.

   .. grid-item-card:: ⚙️ Technical Details
      :link: technical_details/index
      :link-type: doc

      Deep dive into algorithms, architecture, and performance.

🔬 What CSA Does
----------------

CSA transforms raw crystallographic data into rich, analysis-ready datasets through a five-stage pipeline:

1. **Family Extraction** - Query and organize CSD structures by chemical families
2. **Similarity Clustering** - Group structures by 3D packing similarity
3. **Representative Selection** - Choose optimal structures using statistical metrics
4. **Data Extraction** - Extract atomic coordinates, bonds, and intermolecular contacts
5. **Feature Engineering** - Compute advanced geometric and topological descriptors

.. note::
   CSA requires a valid Cambridge Crystallographic Data Centre (CCDC) license for full functionality.

📋 Table of Contents
--------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/index
   getting_started/installation
   getting_started/quickstart
   getting_started/configuration

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/index

.. toctree::
   :maxdepth: 2
   :caption: Technical Details

   technical_details/index

🤝 Community & Support
-----------------------

* **Issues**: Report bugs and request features on GitHub
* **Discussions**: Join the community forum for questions and ideas
* **Contributing**: Read our contribution guidelines to get involved

📄 License
----------

This project is licensed under the MIT License - see the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`