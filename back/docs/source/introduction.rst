Introduction
============

Welcome to **Crystal_Math**, an open-source Python pipeline for automated extraction, clustering, and analysis of molecular-crystal structures from the Cambridge Structural Database (CSD).

Crystal_Math provides:
- **Refcode family extraction**: automatically group all entries sharing a base CSD refcode.
- **Packing-similarity clustering**: detect polymorphs and cluster together by packing similarity.
- **Unique-structure selection**: pick one representative per cluster using the van der Waals Fingerprint Variance (vdWFV) metric.
- **Raw data extraction**: parse CIFs into HDF5 format (coordinates, bonds, hydrogen-bond networks).
- **Post-extraction processing**: GPU-accelerated computation of fragment properties, contact analyses, tensor/quaternion metrics, bond angles, planarity, and Steinhardt order parameters.

----

Repository
----------

:GitHub: https://github.com/nigalanakis/Crystal_Math

----

Prerequisites
-------------

Before you begin, ensure you have:

- **Python 3.9** interpreter  
- A valid **CSD licence** and the **CCDC Python API** (`ccdc` package) installed  
- Standard scientific libraries:
  - **NumPy** (array operations)  
  - **Pandas** (data handling)  
  - **h5py** (HDF5 read/write)  
  - **NetworkX** (graph-based clustering)  
  - **PyTorch** (GPU acceleration)  

----

Next Steps
----------

Proceed to :doc:`installation` for step-by-step setup, then to :doc:`usage` for command-line examples, followed by :doc:`config_reference` and :doc:`api_reference` for detailed configuration and module documentation.
