Pipeline Overview
=================

CSA transforms raw crystallographic data from the Cambridge Structural Database into rich, analysis-ready datasets through a sophisticated five-stage pipeline. This guide explains each stage, data flow, and how to customize the pipeline for your research needs.

.. note::
   
   This overview focuses on understanding the pipeline workflow. For hands-on analysis, see :doc:`basic_analysis`.

The Five-Stage Pipeline
------------------------

CSA's pipeline processes crystal structures through five sequential stages, each building upon the previous to create increasingly refined datasets.

Stage 1: Family Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Query the CSD and group structures into chemical families

**Input**: Configuration filters specifying target molecules
**Output**: ``refcode_families.csv`` - List of structures grouped by chemical family
**Duration**: 5-30 minutes (depending on filter scope)

This stage searches the Cambridge Structural Database using your filter criteria and groups the resulting structures into chemical families based on molecular connectivity.

**Key Operations**:
- Execute CSD queries based on filter parameters
- Apply chemical filters (species, molecular weight, charges, etc.)
- Group structures by molecular formula and connectivity
- Apply quality filters (resolution, disorder, temperature, etc.)

**Example Output Structure**:

.. code-block:: text

    refcode,family,molecular_formula,molecular_weight,space_group
    AABHTZ,family_001,C8H10N2O,150.18,P21/c
    AABHTZ01,family_001,C8H10N2O,150.18,P-1
    AABHTZ02,family_001,C8H10N2O,150.18,Pna21

**Configuration Control**:

.. code-block:: json

    "actions": {
      "get_refcode_families": true  // Enable/disable this stage
    }

Stage 2: Similarity Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Group similar crystal packings within chemical families

**Input**: ``refcode_families.csv``
**Output**: ``clustered_families.csv`` - Structures grouped by packing similarity
**Duration**: 10-60 minutes (depending on dataset size)

This stage uses CCDC's packing similarity algorithms to identify structures with similar 3D arrangements, enabling polymorph identification and packing motif analysis.

**Key Operations**:
- Calculate 3D packing similarity using CCDC algorithms
- Cluster structures within each chemical family
- Assign cluster identifiers based on similarity thresholds
- Rank structures within clusters by quality metrics

**Example Output Structure**:

.. code-block:: text

    refcode,family,cluster,similarity_score,cluster_size,is_representative
    AABHTZ,family_001,cluster_001,1.000,3,True
    AABHTZ01,family_001,cluster_001,0.856,3,False
    AABHTZ02,family_001,cluster_002,1.000,1,True

**Configuration Control**:

.. code-block:: json

    "actions": {
      "cluster_refcode_families": true  // Enable/disable clustering
    }

Stage 3: Representative Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Select one representative structure per cluster to reduce redundancy

**Input**: ``clustered_families.csv``
**Output**: ``unique_structures.csv`` - Selected representative structures
**Duration**: 1-5 minutes

This stage selects the highest-quality representative from each cluster, dramatically reducing dataset size while preserving chemical and packing diversity.

**Key Operations**:
- Apply selection criteria (resolution, completeness, temperature)
- Score structures based on quality metrics
- Select best representative per cluster
- Maintain diversity across chemical families

**Selection Criteria (in priority order)**:
1. **Resolution**: Prefer higher resolution structures
2. **Completeness**: Favor complete datasets
3. **R-factor**: Select lower R-factor structures
4. **Temperature**: Prefer standard temperature measurements
5. **Publication date**: Use more recent determinations as tiebreakers

**Configuration Control**:

.. code-block:: json

    "actions": {
      "get_unique_structures": true  // Enable/disable selection
    }

Stage 4: Structure Data Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Extract detailed structural data from CSD entries

**Input**: ``unique_structures.csv``
**Output**: ``structures.h5`` - Raw HDF5 dataset with coordinates and properties
**Duration**: 30 minutes - 4 hours (depending on dataset size)

This stage retrieves complete structural information from the CSD, including atomic coordinates, bond connectivity, unit cell parameters, and crystallographic metadata.

**Extracted Data Categories**:

**Crystal-Level Properties**:
- Unit cell parameters (a, b, c, α, β, γ)
- Space group and symmetry operations
- Crystal density and volume
- Temperature and experimental conditions

**Molecular Properties**:
- Atomic coordinates and labels
- Bond connectivity and types
- Molecular fragments and formulas
- Formal charges and oxidation states

**Quality Metrics**:
- Resolution and R-factors
- Data completeness
- Disorder flags and quality indicators

**Configuration Control**:

.. code-block:: json

    "actions": {
      "get_structure_data": true  // Enable/disable extraction
    },
    "extraction_batch_size": 32  // Batch size for GPU processing

Stage 5: Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Compute advanced geometric and topological descriptors

**Input**: ``structures.h5``
**Output**: ``structures_processed.h5`` - Analysis-ready dataset with computed features
**Duration**: 1-8 hours (depending on dataset size and complexity)

This stage performs intensive computational analysis to extract geometric descriptors, fragment properties, and intermolecular interactions using GPU-accelerated tensor operations.

**Computed Features**:

**Fragment Analysis**:
- Rigid fragment identification and isolation
- Centers of mass and inertia tensors
- Shape descriptors (asphericity, acylindricity)
- Conformational descriptors

**Geometric Descriptors**:
- Bond lengths, angles, and torsions
- Planarity and linearity metrics
- Ring conformations and puckering
- Molecular volume and surface area

**Intermolecular Interactions**:
- Contact identification and classification
- Hydrogen bond detection and geometry
- π-π stacking interactions
- van der Waals contact analysis

**Topological Descriptors**:
- Connectivity indices
- Graph-based molecular descriptors
- Packing efficiency metrics

**Configuration Control**:

.. code-block:: json

    "actions": {
      "post_extraction_process": true  // Enable/disable feature engineering
    },
    "post_extraction_batch_size": 16  // Batch size for intensive computations

Pipeline Workflow Control
-------------------------

Customizing Pipeline Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pipeline is designed for flexibility, allowing you to:

**Run Complete Pipeline**:

.. code-block:: json

    "actions": {
      "get_refcode_families": true,
      "cluster_refcode_families": true,
      "get_unique_structures": true,
      "get_structure_data": true,
      "post_extraction_process": true
    }

**Skip Clustering (for polymorphism studies)**:

.. code-block:: json

    "actions": {
      "get_refcode_families": true,
      "cluster_refcode_families": false,
      "get_unique_structures": false,
      "get_structure_data": true,
      "post_extraction_process": true
    }

**Resume from Extraction** (if you have existing CSV files):

.. code-block:: json

    "actions": {
      "get_refcode_families": false,
      "cluster_refcode_families": false,
      "get_unique_structures": false,
      "get_structure_data": true,
      "post_extraction_process": true
    }

**Feature Engineering Only** (for existing raw datasets):

.. code-block:: json

    "actions": {
      "get_refcode_families": false,
      "cluster_refcode_families": false,
      "get_unique_structures": false,
      "get_structure_data": false,
      "post_extraction_process": true
    }

Data Flow and Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Understanding pipeline dependencies helps with troubleshooting and custom workflows:

.. code-block:: text

    Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5
       ↓        ↓        ↓        ↓        ↓
    families.csv → clustered.csv → unique.csv → structures.h5 → processed.h5

**Restart Capabilities**:
- Each stage can be restarted independently if outputs exist
- Failed stages automatically resume from last checkpoint
- Intermediate files enable iterative development

Performance Characteristics
---------------------------

Understanding Computational Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Usage Patterns**:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Stage
     - Memory Usage
     - Bottleneck
     - Optimization Strategy
   * - Family Extraction
     - Low (< 2 GB)
     - CSD I/O
     - Use local CSD installation
   * - Clustering
     - Medium (2-8 GB)
     - CCDC algorithms
     - Limit family sizes
   * - Selection
     - Low (< 1 GB)
     - CPU processing
     - Minimal optimization needed
   * - Data Extraction
     - High (4-32 GB)
     - GPU memory
     - Optimize batch sizes
   * - Feature Engineering
     - Very High (8-64 GB)
     - GPU compute
     - Balance batch size and memory

**Time Scaling**:

.. code-block:: python

    # Approximate timing estimates
    def estimate_pipeline_time(n_structures):
        """Estimate total pipeline time in hours."""
        
        family_time = 0.5  # Relatively constant
        cluster_time = n_structures * 0.001  # Linear with structures
        selection_time = 0.1  # Minimal
        extraction_time = n_structures * 0.05  # Linear with batch efficiency
        processing_time = n_structures * 0.1  # Most intensive stage
        
        total_hours = (family_time + cluster_time + selection_time + 
                      extraction_time + processing_time)
        
        return total_hours

**Scaling Recommendations**:

.. code-block:: text

    Dataset Size     | Recommended Resources      | Expected Time
    ------------------|---------------------------|---------------
    < 1,000 structures | 16 GB RAM, GTX 1660      | 2-6 hours
    1,000-10,000      | 32 GB RAM, RTX 3070      | 6-24 hours  
    10,000-50,000     | 64 GB RAM, RTX 4080      | 1-5 days
    > 50,000          | 128 GB RAM, A100/H100    | 3-14 days

Quality Control and Validation
------------------------------

Pipeline Validation Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSA includes comprehensive validation at each stage:

**Stage 1 Validation**:
- Verify filter syntax and parameters
- Check CSD connectivity and licensing
- Validate output file formats

**Stage 2 Validation**:
- Confirm clustering algorithm convergence
- Verify similarity score distributions
- Check for degenerate clusters

**Stage 3 Validation**:
- Validate selection criteria application
- Ensure representative diversity
- Check for missing families

**Stage 4 Validation**:
- Verify structural data completeness
- Check coordinate system consistency
- Validate bond connectivity

**Stage 5 Validation**:
- Confirm feature calculation accuracy
- Check for computation failures
- Validate output data integrity

**Automated Quality Checks**:

.. code-block:: python

    # Example validation workflow
    def validate_pipeline_output(data_directory, data_prefix):
        """Comprehensive pipeline output validation."""
        
        checks = []
        
        # Check file existence
        required_files = [
            f"{data_prefix}_refcode_families.csv",
            f"{data_prefix}_structures_processed.h5"
        ]
        
        for filename in required_files:
            if not Path(data_directory) / "csv" / filename.exists():
                checks.append(f"Missing file: {filename}")
        
        # Validate data integrity
        with h5py.File(f"{data_directory}/structures/{data_prefix}_structures_processed.h5") as f:
            n_structures = len(f['refcode_list'])
            
            # Check for complete feature computation
            required_datasets = [
                'fragment_formula', 'fragment_com_coords', 
                'inter_cc_length', 'bond_length'
            ]
            
            for dataset in required_datasets:
                if dataset not in f:
                    checks.append(f"Missing dataset: {dataset}")
        
        return checks

Troubleshooting Common Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Stage 1 Problems**:

.. code-block:: text

    Error: No structures found matching filters
    Solution: Relax filter criteria gradually

    Error: CSD connection timeout
    Solution: Check CCDC licensing and network connectivity

**Stage 2 Problems**:

.. code-block:: text

    Error: Clustering failed for large families  
    Solution: Enable family size limits in configuration

    Warning: Many singleton clusters
    Solution: Adjust similarity thresholds

**Stage 4-5 Problems**:

.. code-block:: text

    Error: CUDA out of memory
    Solution: Reduce batch sizes

    Error: Slow processing on CPU
    Solution: Enable GPU acceleration or reduce dataset size

Best Practices
--------------

Pipeline Optimization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start Small**: Begin with restrictive filters to understand pipeline behavior
2. **Profile Performance**: Monitor resource usage to optimize batch sizes
3. **Checkpoint Frequently**: Use intermediate outputs for iterative development
4. **Validate Early**: Check results at each stage before proceeding
5. **Document Workflows**: Maintain detailed records of successful configurations

**Development Workflow**:

.. code-block:: bash

    # 1. Test with small dataset
    python csa_main.py --config prototype.json
    
    # 2. Validate results
    python validate_output.py prototype_output/
    
    # 3. Scale up gradually
    python csa_main.py --config medium_scale.json
    
    # 4. Production run
    python csa_main.py --config full_dataset.json

**Production Considerations**:

1. **Resource Planning**: Estimate requirements before large runs
2. **Backup Strategy**: Protect intermediate and final results
3. **Monitoring**: Track progress and resource utilization
4. **Recovery Planning**: Prepare for interruptions and failures
5. **Result Validation**: Verify output quality and completeness

Next Steps
----------

After understanding the pipeline architecture:

**New Users**: Proceed to :doc:`basic_analysis` for hands-on pipeline execution
**Intermediate Users**: Explore :doc:`configuration` for advanced customization
**Advanced Users**: Review :doc:`../technical_details/performance` for optimization strategies

See Also
--------

:doc:`basic_analysis` : Step-by-step pipeline execution guide
:doc:`configuration` : Advanced configuration strategies  
:doc:`data_model` : Understanding CSA's data organization
:doc:`../technical_details/architecture` : Technical implementation details