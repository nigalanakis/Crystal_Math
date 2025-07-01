User Guide
==========

Welcome to the Crystal Structure Analysis (CSA) User Guide. This comprehensive guide covers all aspects of using CSA for molecular crystal analysis, from basic concepts to advanced workflows.

Overview
--------

CSA transforms raw crystallographic data from the Cambridge Structural Database into rich, analysis-ready datasets through a sophisticated five-stage pipeline. This guide will help you understand each component, master the workflows, and optimize your analyses for maximum insight.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔄 Pipeline Overview
      :link: pipeline_overview
      :link-type: doc

      Understand CSA's five-stage workflow and how data flows through the system.

   .. grid-item-card:: 📊 Data Model
      :link: data_model
      :link-type: doc

      Learn how CSA represents crystal structures and molecular properties.

   .. grid-item-card:: ⚙️ Configuration
      :link: configuration
      :link-type: doc

      Master the configuration system for customizing your analyses.

   .. grid-item-card:: 🔍 Basic Analysis
      :link: basic_analysis
      :link-type: doc

      Step-by-step guide to running your first complete CSA analysis.

Core Concepts
-------------

Essential Concepts
~~~~~~~~~~~~~~~~~~

**Crystal Structure Analysis Pipeline**: CSA transforms raw crystallographic data through five sequential stages, each building upon the previous to create increasingly refined and feature-rich datasets.

**Fragment-Based Analysis**: CSA identifies rigid molecular fragments and computes their geometric, topological, and interaction properties, enabling detailed study of molecular packing and intermolecular forces.

**GPU-Accelerated Processing**: Leveraging PyTorch's tensor operations, CSA processes thousands of structures simultaneously on modern GPUs, dramatically reducing analysis time.

**Variable-Length Data Storage**: Using HDF5's variable-length datasets, CSA efficiently stores crystal structures with varying numbers of atoms, bonds, and contacts without padding overhead.

Key Features at a Glance
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - Description
     - Use Cases
   * - **CSD Integration**
     - Direct access to Cambridge Structural Database
     - Large-scale crystal mining
   * - **Similarity Clustering**
     - 3D packing similarity using CCDC algorithms
     - Polymorph identification
   * - **Fragment Analysis**
     - Automatic rigid fragment detection
     - Conformational analysis
   * - **Contact Mapping**
     - Intermolecular contact and H-bond detection
     - Interaction studies
   * - **Geometric Descriptors**
     - Bond angles, torsions, planarity metrics
     - Structure-property relationships
   * - **Batch Processing**
     - GPU-accelerated tensor operations
     - High-throughput screening

Common Workflows
~~~~~~~~~~~~~~~~

**1. Basic Crystal Survey**

.. code-block:: json

    {
      "filters": {
        "target_species": ["C", "H", "N", "O"],
        "molecule_weight_limit": 300.0,
        "crystal_type": ["homomolecular"]
      }
    }

*Use case*: Survey organic crystal structures for statistical analysis

**2. Pharmaceutical Polymorph Analysis**

.. code-block:: json

    {
      "filters": {
        "target_species": ["C", "H", "N", "O", "S", "Cl", "F"],
        "molecule_weight_limit": 800.0,
        "crystal_type": ["homomolecular", "hydrate"]
      },
      "actions": {
        "cluster_refcode_families": true,
        "get_unique_structures": false
      }
    }

*Use case*: Study polymorphic relationships in drug compounds

**3. Intermolecular Interaction Study**

.. code-block:: json

    {
      "filters": {
        "target_species": ["C", "H", "N", "O"],
        "crystal_type": ["co-crystal"]
      },
      "post_extraction_batch_size": 8
    }

*Use case*: Analyze hydrogen bonding patterns in co-crystals

**4. Fragment Conformational Analysis**

.. code-block:: json

    {
      "filters": {
        "molecule_weight_limit": 150.0,
        "target_z_prime_values": [1]
      },
      "extraction_batch_size": 64
    }

*Use case*: Study conformational preferences of small organic molecules

Understanding CSA's Approach
-----------------------------

Scientific Foundation
~~~~~~~~~~~~~~~~~~~~~

CSA is built on several key scientific principles:

**Packing Similarity**: Crystal structures with similar molecular arrangements often exhibit related physical properties. CSA uses CCDC's packing similarity algorithms to identify these relationships.

**Fragment Rigidity**: Molecular fragments connected by non-rotatable bonds behave as rigid bodies in crystal packing. CSA automatically identifies these fragments and treats them as fundamental units.

**Contact Analysis**: Intermolecular contacts shorter than the sum of van der Waals radii indicate significant interactions. CSA systematically identifies and characterizes these contacts.

**Symmetry Expansion**: Crystal symmetry operations generate multiple contact instances from a single asymmetric unit contact. CSA expands contacts using space group symmetry for complete interaction maps.

Technical Architecture
~~~~~~~~~~~~~~~~~~~~~~

**Modular Design**: CSA's five-stage pipeline allows users to run individual stages independently, enabling iterative analysis and custom workflows.

**Data Flow Optimization**: Each stage writes intermediate results to disk, allowing pipeline restart from any point and memory-efficient processing of large datasets.

**GPU Memory Management**: CSA automatically manages GPU memory allocation, batching operations to maximize throughput while preventing out-of-memory errors.

**Validation Framework**: Every stage includes comprehensive validation to ensure data quality and catch potential issues early in the pipeline.

Analysis Workflows
------------------

.. toctree::
   :maxdepth: 2

   basic_analysis
   advanced_features
   batch_processing
   custom_workflows

Data Management
---------------

.. toctree::
   :maxdepth: 2

   input_data
   output_formats
   data_storage
   data_export

Performance and Optimization
-----------------------------

.. toctree::
   :maxdepth: 2

   gpu_acceleration
   memory_management
   parallel_processing

Getting the Most from CSA
--------------------------

Best Practices
~~~~~~~~~~~~~~

1. **Start Small**: Begin with restrictive filters to understand CSA's behavior before scaling to large datasets
2. **Monitor Resources**: Use system monitoring tools to optimize batch sizes for your hardware
3. **Validate Results**: Always inspect output files and run sample analyses to verify data quality
4. **Document Workflows**: Keep detailed records of filter settings and analysis parameters for reproducibility

Performance Tips
~~~~~~~~~~~~~~~~~

1. **GPU Utilization**: Ensure CUDA drivers and PyTorch are properly configured for maximum GPU performance
2. **Storage Optimization**: Use SSDs for HDF5 files and ensure sufficient free space for intermediate files
3. **Memory Planning**: Account for peak memory usage during post-extraction processing
4. **Batch Size Tuning**: Optimize batch sizes based on available GPU memory and structure complexity

Common Pitfalls
~~~~~~~~~~~~~~~~

1. **Over-restrictive Filters**: Too many constraints can result in empty datasets
2. **Insufficient Resources**: Large datasets require substantial computational resources
3. **Incomplete Processing**: Pipeline interruptions can leave datasets in inconsistent states
4. **Version Compatibility**: Ensure CCDC software versions are compatible with your CSD database

Next Steps
----------

Choose your path based on your experience level and goals:

**New Users**: Start with :doc:`pipeline_overview` to understand CSA's workflow, then work through :doc:`basic_analysis`.

**Experienced Users**: Jump to :doc:`advanced_features` or :doc:`custom_workflows` for specialized techniques.

**Developers**: Review :doc:`../technical_details/index` for implementation details and extension points.

**Performance Focus**: Begin with :doc:`gpu_acceleration` and :doc:`memory_management` for optimization strategies.

The User Guide is designed to be read in sequence for comprehensive understanding, but each section is self-contained for quick reference during analysis work.
