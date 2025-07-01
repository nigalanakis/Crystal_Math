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
        atom_mask=mask,
        device=torch.device('cuda')
    )

Data Access Patterns
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import h5py
    import pandas as pd
    
    # Load structure summary data
    def load_structure_summary(hdf5_path):
        """Load crystal properties for all structures."""
        with h5py.File(hdf5_path, 'r') as f:
            data = {
                'refcode': [s.decode() for s in f['refcode_list'][:]],
                'space_group': [sg.decode() for sg in f['space_groups'][:]],
                'cell_volume': f['cell_volumes'][:],
                'cell_density': f['cell_densities'][:]
            }
        return pd.DataFrame(data)

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from csa_config import ExtractionConfig, load_config
    
    # Load from JSON
    config = load_config('analysis.json')
    
    # Create programmatically
    config = ExtractionConfig(
        data_directory='./output',
        data_prefix='my_analysis',
        actions={
            'get_refcode_families': True,
            'cluster_refcode_families': True,
            'get_unique_structures': True,
            'get_structure_data': True,
            'post_extraction_process': True
        },
        filters={
            'target_z_prime_values': [1],
            'molecule_weight_limit': 500.0,
            'target_species': ['C', 'H', 'N', 'O']
        },
        extraction_batch_size=32,
        post_extraction_batch_size=16
    )

Error Handling Patterns
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import logging
    from pathlib import Path
    
    def robust_analysis(config_path, max_retries=3):
        """Execute CSA with error recovery."""
        
        for attempt in range(max_retries):
            try:
                config = load_config(config_path)
                analyzer = CrystalAnalyzer(config)
                analyzer.extract_data()
                return True
                
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logging.info("Retrying...")
                else:
                    logging.error("All attempts failed")
                    raise

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # GPU memory optimization
    torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Batch size optimization
    def optimize_batch_size(available_memory_gb, avg_atoms_per_structure=50):
        memory_per_structure_mb = avg_atoms_per_structure * 2  # Rough estimate
        optimal_batch = int((available_memory_gb * 1024 * 0.5) / memory_per_structure_mb)
        return max(8, min(optimal_batch, 128))
    
    # Use optimized settings
    batch_size = optimize_batch_size(16)  # 16GB GPU
    processor = StructurePostExtractionProcessor(
        hdf5_path='data.h5',
        batch_size=batch_size,
        device=torch.device('cuda')
    )

Module Dependencies
-------------------

**Core Dependencies**

.. code-block:: text

    crystal_analyzer
    ├── csa_config
    ├── csd_operations
    ├── structure_data_extractor
    └── structure_post_extraction_processor

**Processing Pipeline**

.. code-block:: text

    structure_data_extractor
    ├── io.data_writer
    ├── io.hdf5_utils
    └── validation.csd_structure_validator

    structure_post_extraction_processor  
    ├── io.data_reader
    ├── io.data_writer
    ├── processing.geometry_utils
    ├── processing.fragment_utils
    └── processing.contact_utils

**Utility Modules**

.. code-block:: text

    processing/
    ├── geometry_utils
    ├── fragment_utils  
    ├── contact_utils
    ├── cell_utils
    └── symmetry_utils

    io/
    ├── data_reader
    ├── data_writer
    ├── hdf5_utils
    └── dimension_scanner

    validation/
    ├── csd_structure_validator
    └── dataset_initializer

Import Patterns
---------------

**Standard Import Patterns**

.. code-block:: python

    # Main pipeline components
    from crystal_analyzer import CrystalAnalyzer
    from csa_config import load_config, ExtractionConfig
    from csd_operations import CSDOperations
    from structure_data_extractor import StructureDataExtractor
    from structure_post_extraction_processor import StructurePostExtractionProcessor

    # Processing utilities
    from processing.geometry_utils import compute_bond_angles_batch
    from processing.fragment_utils import identify_rigid_fragments_batch
    from processing.contact_utils import compute_intermolecular_contacts_batch
    from processing.cell_utils import compute_cell_matrix_batch

    # I/O utilities
    from io.data_reader import DataReader
    from io.data_writer import DataWriter
    from io.hdf5_utils import initialize_hdf5_file

    # Validation
    from validation.csd_structure_validator import StructureValidator

**Conditional Imports**

.. code-block:: python

    # GPU processing (optional)
    try:
        import torch
        GPU_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        GPU_AVAILABLE = False

    # CCDC API (requires license)
    try:
        from ccdc import io, crystal
        CSD_AVAILABLE = True
    except ImportError:
        CSD_AVAILABLE = False
        
    # Optional visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        PLOTTING_AVAILABLE = True
    except ImportError:
        PLOTTING_AVAILABLE = False

Common Workflows
----------------

**Complete Analysis Pipeline**

.. code-block:: python

    def run_complete_analysis(config_path):
        """Execute full CSA pipeline from configuration to results."""
        
        # 1. Load and validate configuration
        config = load_config(config_path)
        
        # 2. Initialize main analyzer
        analyzer = CrystalAnalyzer(extraction_config=config)
        
        # 3. Execute all pipeline stages
        analyzer.extract_data()
        
        # 4. Verify output files
        output_dir = Path(config.data_directory)
        raw_file = output_dir / f"{config.data_prefix}.h5"
        processed_file = output_dir / f"{config.data_prefix}_processed.h5"
        
        assert raw_file.exists(), "Raw data file not created"
        assert processed_file.exists(), "Processed data file not created"
        
        return raw_file, processed_file

**Stage-by-Stage Processing**

.. code-block:: python

    def run_staged_analysis(config_path):
        """Execute CSA pipeline with intermediate validation."""
        
        config = load_config(config_path)
        
        # Stage 1-3: CSD Operations
        csd_ops = CSDOperations(
            data_directory=config.data_directory,
            data_prefix=config.data_prefix
        )
        
        if config.actions.get('get_refcode_families'):
            families = csd_ops.get_refcode_families_df()
            csd_ops.save_refcode_families_csv(families)
            print(f"Extracted {len(families)} structures")
        
        if config.actions.get('cluster_refcode_families'):
            clustered = csd_ops.cluster_families(config.filters)
            print(f"Created {clustered['cluster_id'].nunique()} clusters")
        
        if config.actions.get('get_unique_structures'):
            unique = csd_ops.get_unique_structures()
            print(f"Selected {len(unique)} representative structures")
        
        # Stage 4: Raw Data Extraction
        if config.actions.get('get_structure_data'):
            extractor = StructureDataExtractor(
                hdf5_path=config.data_directory / f"{config.data_prefix}.h5",
                filters=config.filters,
                batch_size=config.extraction_batch_size
            )
            extractor.run()
            print("Raw data extraction complete")
        
        # Stage 5: Feature Engineering
        if config.actions.get('post_extraction_process'):
            processor = StructurePostExtractionProcessor(
                hdf5_path=config.data_directory / f"{config.data_prefix}.h5",
                batch_size=config.post_extraction_batch_size,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            processor.run()
            print("Feature engineering complete")

**Custom Processing Pipeline**

.. code-block:: python

    def custom_processing_workflow(raw_hdf5_path, custom_config):
        """Example of custom post-processing workflow."""
        
        # Load raw data
        reader = DataReader(raw_hdf5_path)
        
        # Initialize custom processor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Custom processing loop
        for batch_start in range(0, reader.n_structures, custom_config['batch_size']):
            batch_end = min(batch_start + custom_config['batch_size'], reader.n_structures)
            
            # Load batch data
            crystal_data = reader.load_crystal_batch(batch_start, batch_end)
            atom_data = reader.load_atom_batch(batch_start, batch_end)
            
            # Custom feature computation
            custom_features = compute_custom_descriptors(
                crystal_data, atom_data, custom_config
            )
            
            # Save results
            save_custom_features(custom_features, batch_start, custom_config['output_path'])

**Parallel Processing Workflow**

.. code-block:: python

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    def parallel_analysis_workflow(config_list, max_workers=None):
        """Process multiple configurations in parallel."""
        
        if max_workers is None:
            max_workers = min(len(config_list), mp.cpu_count() - 1)
        
        def process_single_config(config_path):
            try:
                return run_complete_analysis(config_path)
            except Exception as e:
                return f"Failed: {config_path} - {str(e)}"
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_config, config_list))
        
        # Summarize results
        successful = [r for r in results if not isinstance(r, str)]
        failed = [r for r in results if isinstance(r, str)]
        
        print(f"Completed: {len(successful)} successful, {len(failed)} failed")
        return successful, failed

Advanced Usage Examples
-----------------------

**Memory-Efficient Large Dataset Processing**

.. code-block:: python

    def memory_efficient_processing(hdf5_path, small_batch_size=8):
        """Process large datasets with limited memory."""
        
        import gc
        
        processor = StructurePostExtractionProcessor(
            hdf5_path=hdf5_path,
            batch_size=small_batch_size,
            device=torch.device('cuda')
        )
        
        # Enable memory optimizations
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        # Process with periodic cleanup
        original_run = processor.run
        
        def memory_managed_run():
            try:
                return original_run()
            finally:
                torch.cuda.empty_cache()
                gc.collect()
        
        processor.run = memory_managed_run
        processor.run()

**Quality Control Pipeline**

.. code-block:: python

    def quality_control_pipeline(config_path):
        """Enhanced pipeline with comprehensive quality checks."""
        
        config = load_config(config_path)
        
        # Pre-analysis validation
        assert CSD_AVAILABLE, "CCDC API not available"
        assert GPU_AVAILABLE or config.post_extraction_batch_size <= 16, "GPU required for large batches"
        
        # Run analysis with monitoring
        start_time = time.time()
        
        try:
            analyzer = CrystalAnalyzer(extraction_config=config)
            analyzer.extract_data()
            
        except Exception as e:
            logging.error(f"Analysis failed: {e}")
            raise
        
        duration = time.time() - start_time
        
        # Post-analysis validation
        output_dir = Path(config.data_directory)
        validate_output_files(output_dir, config.data_prefix)
        
        # Performance reporting
        logging.info(f"Analysis completed in {duration:.1f} seconds")
        
        return duration

**Integration with External Tools**

.. code-block:: python

    def export_for_external_analysis(hdf5_path, output_formats=['csv', 'json']):
        """Export CSA results for external analysis tools."""
        
        import pandas as pd
        import json
        
        # Load processed data
        reader = DataReader(hdf5_path)
        
        # Export crystal properties
        if 'csv' in output_formats:
            crystal_props = reader.load_all_crystal_properties()
            crystal_df = pd.DataFrame(crystal_props)
            crystal_df.to_csv(hdf5_path.with_suffix('_crystal_properties.csv'), index=False)
        
        # Export fragment data
        if 'json' in output_formats:
            fragment_data = reader.load_all_fragment_data()
            with open(hdf5_path.with_suffix('_fragments.json'), 'w') as f:
                json.dump(fragment_data, f, indent=2)
        
        # Export contact networks
        contact_networks = reader.load_contact_networks()
        export_contact_networks_to_gephi(contact_networks, hdf5_path.with_suffix('_networks.gexf'))

Troubleshooting
---------------

**Common Issues and Solutions**

.. code-block:: python

    def diagnose_csa_issues():
        """Diagnostic function for common CSA problems."""
        
        issues_found = []
        
        # Check CCDC installation
        try:
            from ccdc import io
            io.EntryReader('CSD')
        except ImportError:
            issues_found.append("CCDC Python API not installed")
        except Exception as e:
            issues_found.append(f"CCDC connection failed: {e}")
        
        # Check GPU availability
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                test_tensor = torch.randn(1000, 1000, device='cuda')
                del test_tensor
            except Exception as e:
                issues_found.append(f"GPU test failed: {e}")
        else:
            issues_found.append("CUDA not available - will use CPU (slower)")
        
        # Check disk space
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        if free_space_gb < 10:
            issues_found.append(f"Low disk space: {free_space_gb:.1f}GB available")
        
        # Report findings
        if issues_found:
            print("Issues detected:")
            for issue in issues_found:
                print(f"  - {issue}")
        else:
            print("All systems check passed!")
        
        return len(issues_found) == 0

**Performance Monitoring**

.. code-block:: python

    import psutil
    import time

    class CSAPerformanceMonitor:
        """Monitor CSA performance and resource usage."""
        
        def __init__(self):
            self.start_time = None
            self.process = psutil.Process()
        
        def start_monitoring(self):
            self.start_time = time.time()
            self.initial_memory = self.process.memory_info().rss / 1024**2
            
        def get_current_stats(self):
            current_time = time.time()
            current_memory = self.process.memory_info().rss / 1024**2
            
            stats = {
                'elapsed_time': current_time - self.start_time,
                'memory_usage_mb': current_memory,
                'memory_increase_mb': current_memory - self.initial_memory,
                'cpu_percent': self.process.cpu_percent()
            }
            
            if torch.cuda.is_available():
                stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
                stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            
            return stats
        
        def log_stats(self):
            stats = self.get_current_stats()
            logging.info(f"Performance: {stats}")

**Configuration Validation**

.. code-block:: python

    def validate_csa_configuration(config_path):
        """Comprehensive configuration validation."""
        
        try:
            config = load_config(config_path)
        except Exception as e:
            return False, f"Configuration loading failed: {e}"
        
        validation_errors = []
        
        # Check required directories
        data_dir = Path(config.data_directory)
        if not data_dir.exists():
            try:
                data_dir.mkdir(parents=True)
            except Exception as e:
                validation_errors.append(f"Cannot create data directory: {e}")
        
        # Check batch sizes
        if config.extraction_batch_size > 256:
            validation_errors.append("Extraction batch size too large (>256)")
        
        if config.post_extraction_batch_size > 128:
            validation_errors.append("Post-extraction batch size too large (>128)")
        
        # Check filter consistency
        if 'target_species' in config.filters:
            valid_elements = {'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
                             'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca'}
            invalid_elements = set(config.filters['target_species']) - valid_elements
            if invalid_elements:
                validation_errors.append(f"Invalid elements: {invalid_elements}")
        
        # Return results
        is_valid = len(validation_errors) == 0
        return is_valid, validation_errors

See Also
--------

* :doc:`../getting_started/quickstart` - Getting started with CSA
* :doc:`../user_guide/index` - Comprehensive user guide
* :doc:`../tutorials/index` - Step-by-step tutorials
* :doc:`../examples/index` - Ready-to-run examples
* :doc:`../technical_details/index` - Implementation details