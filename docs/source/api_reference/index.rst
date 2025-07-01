API Reference
=============

Complete reference documentation for all CSA modules, classes, and functions.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   core/crystal_analyzer
   core/csa_config
   core/csa_main

.. toctree::
   :maxdepth: 2
   :caption: Extraction Pipeline

   extraction/csd_operations
   extraction/structure_data_extractor
   extraction/structure_post_extraction_processor

.. toctree::
   :maxdepth: 2
   :caption: Processing Utilities

   processing/geometry_utils
   processing/fragment_utils
   processing/contact_utils
   processing/cell_utils
   processing/symmetry_utils

.. toctree::
   :maxdepth: 2
   :caption: Input/Output

   io/data_reader
   io/data_writer
   io/hdf5_utils
   io/dimension_scanner

.. toctree::
   :maxdepth: 2
   :caption: Validation

   validation/csd_structure_validator
   validation/dataset_initializer

Module Overview
---------------

Core Modules
~~~~~~~~~~~~

:doc:`core/crystal_analyzer`
    Main orchestration class for the complete CSA pipeline

:doc:`core/csa_config` 
    Configuration management and validation

:doc:`core/csa_main`
    Command-line interface and entry point

Extraction Pipeline
~~~~~~~~~~~~~~~~~~~

:doc:`extraction/csd_operations`
    High-level CSD operations: family extraction, clustering, selection

:doc:`extraction/structure_data_extractor`
    Raw data extraction from CSD structures into HDF5

:doc:`extraction/structure_post_extraction_processor`
    Feature engineering and advanced descriptor computation

Processing Utilities
~~~~~~~~~~~~~~~~~~~~

:doc:`processing/geometry_utils`
    Geometric calculations: bond angles, planarity, order parameters

:doc:`processing/fragment_utils`
    Fragment identification and property computation

:doc:`processing/contact_utils`
    Intermolecular contact analysis and symmetry expansion

:doc:`processing/cell_utils`
    Unit cell transformations and matrix operations

:doc:`processing/symmetry_utils`
    Crystallographic symmetry operations and parsing

Input/Output
~~~~~~~~~~~~

:doc:`io/data_reader`
    Reading raw data from HDF5 files for batch processing

:doc:`io/data_writer`
    Writing processed data to HDF5 with proper formatting

:doc:`io/hdf5_utils`
    HDF5 file initialization and management utilities

:doc:`io/dimension_scanner`
    Scanning datasets to determine optimal array dimensions

Validation
~~~~~~~~~~

:doc:`validation/csd_structure_validator`
    Structure quality validation against filter criteria

:doc:`validation/dataset_initializer`
    HDF5 dataset creation and schema management

Quick Reference
---------------

Most Common Classes
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   crystal_analyzer.CrystalAnalyzer
   csa_config.ExtractionConfig
   csd_operations.CSDOperations
   structure_data_extractor.StructureDataExtractor
   structure_post_extraction_processor.StructurePostExtractionProcessor

Key Functions
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   csa_config.load_config
   geometry_utils.compute_bond_angles_batch
   fragment_utils.identify_rigid_fragments_batch
   contact_utils.compute_symmetric_contacts_batch
   cell_utils.compute_cell_matrix_batch

Data Structures
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   csd_structure_validator.StructureValidationResult
   structure_post_extraction_processor.CrystalParams
   structure_post_extraction_processor.AtomParams
   structure_post_extraction_processor.BondParams

Usage Patterns
--------------

Basic Pipeline Setup
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from crystal_analyzer import CrystalAnalyzer
    from csa_config import load_config
    
    # Load configuration
    config = load_config('my_analysis.json')
    
    # Initialize analyzer
    analyzer = CrystalAnalyzer(config)
    
    # Run pipeline
    analyzer.extract_data()

Custom Processing
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from structure_post_extraction_processor import StructurePostExtractionProcessor
    import torch
    
    # Initialize processor
    processor = StructurePostExtractionProcessor(
        hdf5_path='data.h5',
        batch_size=32,
        device=torch.device('cuda')
    )
    
    # Run processing
    processor.run()

Batch Geometric Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from geometry_utils import compute_bond_angles_batch
    import torch
    
    # Compute bond angles for batch
    angle_ids, angles, mask, indices = compute_bond_angles_batch(
        atom_labels=labels,
        atom_coords=coords,
        atom_mask=atom_mask,
        bond_atom1_idx=bond1_idx,
        bond_atom2_idx=bond2_idx,
        bond_mask=bond_mask,
        device=torch.device('cuda')
    )

Data Access
~~~~~~~~~~~

.. code-block:: python

    from data_reader import RawDataReader
    import h5py
    
    # Open HDF5 file
    with h5py.File('structures.h5', 'r') as f:
        reader = RawDataReader(f)
        
        # Read batch of structures
        crystal_params = reader.read_crystal_parameters(refcodes)
        atom_data = reader.read_atoms(refcodes, max_atoms=100)

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from csa_config import ExtractionConfig
    
    # Load from JSON
    config = ExtractionConfig.from_json('config.json')
    
    # Access configuration
    batch_size = config.extraction_batch_size
    filters = config.filters
    actions = config.actions

Type Hints and Documentation
----------------------------

All CSA modules include comprehensive type hints and docstrings following NumPy documentation standards:

.. code-block:: python

    def compute_center_of_mass_batch(
            atom_coords: torch.Tensor,        # (B, N, 3)
            atom_frac_coords: torch.Tensor,   # (B, N, 3)
            atom_weights: torch.Tensor,       # (B, N)
            atom_mask: torch.BoolTensor,      # (B, N)
            device: torch.device
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Cartesian and fractional centers of mass for each fragment.

        Parameters
        ----------
        atom_coords : torch.Tensor of shape (B, N, 3)
            Cartesian coordinates, padded to N atoms.
        atom_frac_coords : torch.Tensor of shape (B, N, 3)
            Fractional coordinates, padded similarly.
        atom_weights : torch.Tensor of shape (B, N)
            Atomic weights (zero for padding).
        atom_mask : torch.BoolTensor of shape (B, N)
            True for real atoms, False for padding.
        device : torch.device
            Device to perform computation on.

        Returns
        -------
        com_coords : torch.Tensor of shape (B, 3)
            Cartesian center of mass per fragment.
        com_frac_coords : torch.Tensor of shape (B, 3)
            Fractional center of mass per fragment.
        """