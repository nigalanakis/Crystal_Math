csa_main module
===============

.. automodule:: csa_main
   :members:
   :undoc-members:
   :show-inheritance:

Command-Line Interface for CSA Pipeline
----------------------------------------

The ``csa_main`` module provides the primary command-line interface for executing the Crystal Structure Analysis pipeline. It handles argument parsing, logging configuration, and orchestrates the complete extraction workflow.

Main Functions
--------------

.. autofunction:: csa_main.main

   **Entry Point for CSA Execution**

   Main entry point that orchestrates the complete CSA pipeline execution. This function coordinates logging setup, argument parsing, and pipeline execution with comprehensive error handling.

   **Execution Workflow:**

   1. **Logging Configuration** - Sets up structured console logging
   2. **Argument Parsing** - Processes command-line arguments and validates inputs  
   3. **Pipeline Execution** - Loads configuration and runs extraction pipeline
   4. **Error Handling** - Captures and logs any execution failures
   5. **Success Reporting** - Confirms successful completion

   **Usage from Command Line:**

   .. code-block:: bash

      # Basic execution with default configuration
      python csa_main.py --config analysis.json
      
      # Using alternative configuration path
      python csa_main.py --config /path/to/custom/config.json
      
      # With output redirection and logging
      python csa_main.py --config analysis.json 2>&1 | tee analysis.log

   **Usage from Python:**

   .. code-block:: python

      # Direct function call
      import sys
      sys.argv = ['csa_main.py', '--config', 'my_analysis.json']
      
      from csa_main import main
      main()

   **Exit Codes:**
      * **0** - Successful completion
      * **1** - Configuration or parsing error  
      * **2** - Pipeline execution failure

   **Raises:**
      * :obj:`SystemExit` - On argument parsing failures
      * :obj:`Exception` - On pipeline execution errors (logged and re-raised)

.. autofunction:: csa_main.run_extraction

   **Execute CSA Data Extraction Pipeline**

   Coordinates the complete CSA extraction workflow by loading configuration, initializing the analyzer, and executing all enabled pipeline stages.

   **Pipeline Stages Executed:**

   1. **Refcode Family Extraction** - Query CSD for structure families
   2. **Similarity Clustering** - Group similar crystal packings
   3. **Representative Selection** - Choose optimal structures per cluster  
   4. **Raw Data Extraction** - Extract atomic coordinates and contacts
   5. **Feature Engineering** - Compute advanced structural descriptors

   **Parameters:**
      * **config_path** (:obj:`Path`) - Path to JSON configuration file

   **Configuration Loading:**

   .. code-block:: python

      # Configuration is loaded and validated
      extraction_cfg = load_config(config_path)
      
      # CrystalAnalyzer is initialized with config
      analyzer = CrystalAnalyzer(extraction_config=extraction_cfg)
      
      # Complete pipeline is executed
      analyzer.extract_data()

   **Progress Monitoring:**

   The function provides detailed logging of pipeline progress:

   .. code-block:: text

      INFO - Loading configuration from /path/to/config.json
      INFO - Starting extraction step...
      INFO - Extracting refcode families into DataFrame...
      INFO - Extracted 45,823 structures across 12,456 families
      INFO - Clustering refcode families...
      INFO - Refcode families clustered into 8,234 groups
      INFO - Selecting unique structures...
      INFO - Unique structures selected: 8,234 structures
      INFO - Extracting detailed structure data...
      INFO - Raw data extraction complete
      INFO - Starting post-extraction processing...
      INFO - Post-extraction processing complete

   **Error Recovery:**

   Errors are logged with full context and re-raised:

   .. code-block:: python

      try:
          analyzer.extract_data()
      except Exception as e:
          logging.exception("Data extraction failed with an error.")
          raise  # Re-raise for calling code

   **Output Files Generated:**

   * ``{prefix}_refcode_families.csv`` - Initial family assignments
   * ``{prefix}_refcode_families_clustered.csv`` - Clustered families
   * ``{prefix}_refcode_families_unique.csv`` - Selected representatives
   * ``{prefix}.h5`` - Raw structural data in HDF5 format
   * ``{prefix}_processed.h5`` - Computed features and descriptors

   **Raises:**
      * :obj:`Exception` - Any error during extraction (logged and re-raised)

Configuration and Logging
--------------------------

.. autofunction:: csa_main.setup_logging

   **Configure Structured Console Logging**

   Establishes standardized logging configuration for the CSA pipeline with appropriate formatting and output handling.

   **Logging Configuration:**

   * **Handler Cleanup** - Removes any existing root logger handlers
   * **Stream Handler** - Directs output to stderr for proper console display
   * **Formatter** - Structured format with timestamps and module identification
   * **Log Level** - Set to INFO for operational visibility

   **Log Format:**

   .. code-block:: text

      %(asctime)s - %(name)s - %(levelname)s - %(message)s

   **Example Output:**

   .. code-block:: text

      2024-01-15 10:30:15 - crystal_analyzer - INFO - Starting data extraction pipeline
      2024-01-15 10:30:16 - csd_operations - INFO - Extracting refcode families
      2024-01-15 10:31:45 - structure_data_extractor - INFO - Processing batch 1/256

   **Usage in Scripts:**

   .. code-block:: python

      from csa_main import setup_logging
      
      # Configure logging for custom scripts
      setup_logging()
      
      import logging
      logger = logging.getLogger(__name__)
      logger.info("Custom script starting...")

.. autofunction:: csa_main.parse_args

   **Parse and Validate Command-Line Arguments**

   Processes command-line arguments for the CSA pipeline with validation and default value handling.

   **Supported Arguments:**

   .. option:: -c, --config

      Path to JSON configuration file

      * **Type:** Path object
      * **Default:** ``../config/csa_config.json`` (relative to script location)
      * **Required:** No (uses default if not specified)

   **Argument Processing:**

   .. code-block:: python

      parser = argparse.ArgumentParser(
          description="Run CSD data extraction pipeline"
      )
      parser.add_argument(
          '-c', '--config',
          type=Path,
          default=Path('../config/csa_config.json').expanduser(),
          help="Path to the JSON configuration file"
      )

   **Usage Examples:**

   .. code-block:: bash

      # Use default configuration
      python csa_main.py
      
      # Specify custom configuration
      python csa_main.py --config my_analysis.json
      
      # Full path specification
      python csa_main.py --config /home/user/projects/config.json

   **Validation:**

   * Path objects are automatically created and expanded
   * User home directory expansion (``~``) is supported
   * Relative paths are resolved from script location

   **Returns:**
      :obj:`argparse.Namespace` with validated ``config`` attribute

   **Raises:**
      :obj:`SystemExit` - On argument parsing failures or help requests

Command-Line Usage
------------------

**Basic Execution Patterns**

.. code-block:: bash

   # Standard execution
   python src/csa_main.py --config analysis.json
   
   # With logging to file
   python src/csa_main.py --config analysis.json 2>&1 | tee analysis.log
   
   # Background execution
   nohup python src/csa_main.py --config analysis.json > analysis.log 2>&1 &

**Configuration File Management**

.. code-block:: bash

   # Validate configuration before running
   python -c "from csa_config import load_config; load_config('analysis.json')"
   
   # Copy and modify template
   cp templates/pharmaceutical.json my_analysis.json
   nano my_analysis.json
   
   # Run with custom configuration
   python src/csa_main.py --config my_analysis.json

**Resource Monitoring**

.. code-block:: bash

   # Monitor resource usage during execution
   python src/csa_main.py --config analysis.json &
   watch -n 5 'ps aux | grep csa_main'
   
   # Track GPU usage if applicable
   watch -n 2 nvidia-smi

**Error Handling and Recovery**

.. code-block:: bash

   # Capture full error traces
   python src/csa_main.py --config analysis.json 2>&1 | tee full_log.txt
   
   # Resume from checkpoint (if implemented)
   python src/csa_main.py --config analysis.json --resume
   
   # Debug with verbose output
   python -u src/csa_main.py --config analysis.json

Integration Examples
--------------------

**Batch Processing Scripts**

.. code-block:: python

   #!/usr/bin/env python3
   """Batch process multiple configurations."""
   
   import subprocess
   import sys
   from pathlib import Path
   
   def run_csa_batch(config_files):
       """Execute CSA for multiple configurations."""
       for config_file in config_files:
           print(f"Processing {config_file}...")
           
           result = subprocess.run([
               sys.executable, 'src/csa_main.py',
               '--config', str(config_file)
           ], capture_output=True, text=True)
           
           if result.returncode == 0:
               print(f"✓ {config_file} completed successfully")
           else:
               print(f"✗ {config_file} failed:")
               print(result.stderr)
   
   # Process all configurations in directory
   config_dir = Path('configurations')
   configs = list(config_dir.glob('*.json'))
   run_csa_batch(configs)

**Workflow Integration**

.. code-block:: python

   """Integrate CSA into larger analysis workflow."""
   
   from csa_main import run_extraction
   from pathlib import Path
   import logging
   
   def complete_analysis_workflow(base_config):
       """Execute complete analysis with pre/post processing."""
       
       # Pre-processing steps
       logging.info("Starting pre-processing...")
       prepare_analysis_environment()
       
       # CSA execution
       logging.info("Running CSA extraction...")
       config_path = Path(base_config)
       run_extraction(config_path)
       
       # Post-processing steps  
       logging.info("Starting post-processing...")
       analyze_extracted_data()
       generate_reports()
       
       logging.info("Complete workflow finished")

**Error Recovery and Monitoring**

.. code-block:: python

   """Monitor and recover from CSA execution errors."""
   
   import time
   import subprocess
   from pathlib import Path
   
   def monitored_csa_execution(config_path, max_retries=3):
       """Execute CSA with automatic retry on failure."""
       
       for attempt in range(max_retries):
           try:
               result = subprocess.run([
                   'python', 'src/csa_main.py',
                   '--config', str(config_path)
               ], check=True, capture_output=True, text=True)
               
               print("CSA execution completed successfully")
               return result
               
           except subprocess.CalledProcessError as e:
               print(f"Attempt {attempt + 1} failed: {e}")
               if attempt < max_retries - 1:
                   print(f"Retrying in 60 seconds...")
                   time.sleep(60)
               else:
                   print("All retry attempts exhausted")
                   raise

Performance Considerations
--------------------------

**Memory Management**

The command-line interface provides monitoring for memory usage:

.. code-block:: python

   # Memory usage is logged during execution
   INFO - Peak memory usage: 8.2 GB
   INFO - GPU memory allocated: 3.1 GB
   INFO - Processing batch 45/128 (35% complete)

**Execution Time Estimation**

.. code-block:: text

   INFO - Stage 1 (Family Extraction): 2.3 minutes
   INFO - Stage 2 (Clustering): 15.7 minutes  
   INFO - Stage 3 (Selection): 1.1 minutes
   INFO - Stage 4 (Data Extraction): 45.2 minutes
   INFO - Stage 5 (Feature Engineering): 23.8 minutes
   INFO - Total execution time: 88.1 minutes

**Resource Requirements**

* **CPU**: Multi-core recommended for parallel processing
* **Memory**: 8GB+ RAM for typical datasets  
* **GPU**: CUDA-compatible GPU recommended for stages 4-5
* **Storage**: 10GB+ free space for intermediate files

See Also
--------

:doc:`../core/crystal_analyzer` : Main pipeline orchestration
:doc:`../core/csa_config` : Configuration management
:doc:`../getting_started/quickstart` : Getting started guide