Installation
============

This guide will help you install the Crystal Structure Analysis (CSA) framework and its dependencies.

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~~

- **Python**: 3.8 or higher
- **Operating System**: Linux (recommended), macOS, or Windows
- **Memory**: 16 GB RAM minimum, 32 GB+ recommended for large datasets
- **GPU**: CUDA-compatible GPU recommended for optimal performance
- **Storage**: SSD recommended for HDF5 file operations

Required Licenses
~~~~~~~~~~~~~~~~~~

CSA requires access to the Cambridge Structural Database (CSD):

- **CCDC License**: Valid license for CCDC software and Python API
- **CSD Database**: Access to the Cambridge Structural Database

Contact the `Cambridge Crystallographic Data Centre <https://www.ccdc.cam.ac.uk/>`_ for licensing information.

Installation Methods
--------------------

Method 1: Standard Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone the repository**::

    git clone https://github.com/your-org/crystal-structure-analysis.git
    cd crystal-structure-analysis

2. **Create a virtual environment**::

    python -m venv csa_env
    source csa_env/bin/activate  # On Windows: csa_env\Scripts\activate

3. **Install dependencies**::

    pip install -r requirements.txt

Method 2: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For contributors and developers:

1. **Clone with development dependencies**::

    git clone https://github.com/your-org/crystal-structure-analysis.git
    cd crystal-structure-analysis

2. **Install in development mode**::

    pip install -e .
    pip install -r requirements-dev.txt

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

CSA relies on several key packages:

**Scientific Computing**
  - **PyTorch** (≥1.12.0): GPU-accelerated tensor operations
  - **NumPy** (≥1.21.0): Numerical computations
  - **SciPy** (≥1.7.0): Scientific algorithms

**Data Management**
  - **HDF5** (≥1.12.0): High-performance data storage
  - **h5py** (≥3.6.0): Python interface to HDF5
  - **pandas** (≥1.3.0): Data manipulation and analysis

**Crystallography**
  - **CCDC Python API**: Cambridge Structural Database access
  - **NetworkX** (≥2.6): Graph operations for molecular networks

**Parallel Processing**
  - **multiprocessing**: Built-in Python parallel processing
  - **concurrent.futures**: Asynchronous execution

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

**Visualization**
  - **matplotlib** (≥3.5.0): Plotting and visualization
  - **plotly** (≥5.0.0): Interactive plots

**Development Tools**
  - **pytest** (≥6.0.0): Testing framework
  - **black**: Code formatting
  - **flake8**: Code linting
  - **sphinx**: Documentation generation

CCDC Software Setup
-------------------

Installing CCDC Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Download CCDC Software**: Obtain the CCDC software suite from the CCDC website
2. **Install CSD Python API**: Follow CCDC's installation instructions for your platform
3. **Verify Installation**::

    python -c "from ccdc import io; print('CCDC API successfully imported')"

Database Configuration
~~~~~~~~~~~~~~~~~~~~~~

Configure your CSD database path:

1. **Set Environment Variables**::

    export CCDC_CSD_DIRECTORY="/path/to/your/csd/database"

2. **Verify Database Access**::

    python -c "from ccdc import io; reader = io.EntryReader('CSD'); print(f'CSD contains {len(reader)} entries')"

GPU Setup (Optional but Recommended)
-------------------------------------

CUDA Installation
~~~~~~~~~~~~~~~~~

For GPU acceleration:

1. **Install CUDA Toolkit** (≥11.6):
   
   - Download from `NVIDIA CUDA <https://developer.nvidia.com/cuda-toolkit>`_
   - Follow platform-specific installation instructions

2. **Install PyTorch with CUDA**::

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3. **Verify GPU Setup**::

    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

Verification
------------

Quick Installation Test
~~~~~~~~~~~~~~~~~~~~~~~

Run this script to verify your installation:

.. code-block:: python

    #!/usr/bin/env python3
    """Installation verification script for CSA."""

    def test_imports():
        """Test all critical imports."""
        try:
            # Core scientific libraries
            import numpy as np
            import torch
            import h5py
            import pandas as pd
            print("✓ Core scientific libraries imported successfully")
            
            # CCDC API
            from ccdc import io
            print("✓ CCDC API imported successfully")
            
            # CSA modules
            import sys
            sys.path.append('src')
            from csa_config import load_config
            from crystal_analyzer import CrystalAnalyzer
            print("✓ CSA modules imported successfully")
            
            return True
        except ImportError as e:
            print(f"✗ Import error: {e}")
            return False

    def test_gpu():
        """Test GPU availability."""
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {device}")
            return True
        else:
            print("! GPU not available (CPU-only mode)")
            return False

    def test_csd_access():
        """Test CSD database access."""
        try:
            from ccdc import io
            reader = io.EntryReader('CSD')
            entry_count = len(reader)
            print(f"✓ CSD database accessible ({entry_count:,} entries)")
            return True
        except Exception as e:
            print(f"✗ CSD access error: {e}")
            return False

    if __name__ == "__main__":
        print("CSA Installation Verification")
        print("=" * 40)
        
        success = True
        success &= test_imports()
        success &= test_gpu()
        success &= test_csd_access()
        
        print("=" * 40)
        if success:
            print("🎉 Installation verification successful!")
        else:
            print("❌ Installation issues detected. Check error messages above.")

Save this as ``verify_installation.py`` and run::

    python verify_installation.py

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**CCDC API Not Found**

*Problem*: ``ImportError: No module named 'ccdc'``

*Solutions*:
  1. Ensure CCDC software is properly installed
  2. Check that the CCDC Python API is in your Python path
  3. Verify your CCDC license is valid

**CUDA Not Available**

*Problem*: ``torch.cuda.is_available()`` returns ``False``

*Solutions*:
  1. Install NVIDIA GPU drivers
  2. Install CUDA toolkit
  3. Reinstall PyTorch with CUDA support
  4. Check GPU compatibility

**HDF5 Library Issues**

*Problem*: HDF5 library errors during h5py operations

*Solutions*:
  1. Update h5py: ``pip install --upgrade h5py``
  2. Install HDF5 development libraries (Linux): ``sudo apt-get install libhdf5-dev``
  3. Use conda for HDF5 management: ``conda install h5py``

**Memory Issues**

*Problem*: Out of memory errors during processing

*Solutions*:
  1. Reduce batch sizes in configuration
  2. Enable memory-mapped HDF5 datasets
  3. Use CPU processing for large datasets
  4. Monitor memory usage with system tools

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. **Check the logs**: CSA provides detailed logging information
2. **Review configuration**: Ensure your config files are properly formatted
3. **Test with small datasets**: Verify functionality with minimal examples
4. **Consult documentation**: Check the troubleshooting section
5. **Report issues**: Submit bug reports with full error traces

Next Steps
----------

After successful installation:

1. **Configure CSA**: Set up your configuration files
2. **Run quickstart**: Follow the quickstart tutorial
3. **Explore examples**: Try the provided example workflows
4. **Read the user guide**: Learn about CSA's capabilities

Continue to the :doc:`quickstart` to set up your first CSA project.