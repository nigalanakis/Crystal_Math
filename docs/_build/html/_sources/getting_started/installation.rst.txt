Installation
============

This guide will help you install the Crystal Structure Analysis (CSA) framework and its dependencies step by step.

System Requirements
------------------

**Minimum Requirements**
  * **Python**: 3.9 (required for CSD Python API)
  * **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
  * **Memory**: 16 GB RAM minimum
  * **Storage**: 10 GB free space (more for large analyses)

**Recommended Requirements**
  * **Memory**: 32+ GB RAM for large datasets
  * **GPU**: CUDA-compatible GPU for optimal performance
  * **Storage**: SSD for HDF5 file operations

**Required Licenses**
  * **CCDC License**: Valid license for CCDC software and Python API
  * **CSD Database**: Access to the Cambridge Structural Database

.. note::
   Contact the `Cambridge Crystallographic Data Centre <https://www.ccdc.cam.ac.uk/>`_ for licensing information.

Step 1: Install Python and Prerequisites
----------------------------------------

Python 3.9 Installation
~~~~~~~~~~~~~~~~~~~~~~~

**Windows:**

1. Download Python 3.9 from `python.org <https://www.python.org/downloads/>`_
2. Run the installer and check "Add Python to PATH"
3. Verify installation:

.. code-block:: bash

   python --version
   # Should output: Python 3.9.x

**macOS:**

.. code-block:: bash

   # Using Homebrew (recommended)
   brew install python@3.9
   
   # Or download from python.org

**Linux (Ubuntu/Debian):**

.. code-block:: bash

   sudo apt update
   sudo apt install python3.9 python3.9-venv python3.9-pip

Step 2: Install CCDC Software
-----------------------------

CSD Software Suite
~~~~~~~~~~~~~~~~~

1. **Download CCDC Software**:
   - Log into your CCDC account
   - Download the CSD System for your platform
   - Follow CCDC's installation instructions

2. **Install CSD Python API**:
   - Typically included with CSD installation
   - May require separate activation

3. **Verify CCDC Installation**:

.. code-block:: bash

   python -c "from ccdc import io; print('CCDC API successfully imported')"

.. note::
   If you encounter import errors, ensure the CCDC Python API is in your Python path. Check CCDC documentation for platform-specific setup.

Step 3: Install CSA
-------------------

Download CSA
~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/crystal-structure-analysis.git
   cd crystal-structure-analysis

Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create virtual environment
   python -m venv csa_env
   
   # Activate environment
   # Windows:
   csa_env\Scripts\activate
   
   # macOS/Linux:
   source csa_env/bin/activate

Install Dependencies
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Upgrade pip
   pip install --upgrade pip
   
   # Install CSA dependencies
   pip install -r requirements.txt

.. note::
   If you encounter compilation errors during installation, you may need to install development tools for your platform.

Step 4: Configure CSA
---------------------

Set Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~

**CSD Database Path:**

.. code-block:: bash

   # Windows (Command Prompt):
   set CCDC_CSD_DIRECTORY=C:\path\to\your\csd\database
   
   # Windows (PowerShell):
   $env:CCDC_CSD_DIRECTORY="C:\path\to\your\csd\database"
   
   # macOS/Linux:
   export CCDC_CSD_DIRECTORY="/path/to/your/csd/database"

**Make Environment Variables Permanent:**

**Windows:**
- Search for "Environment Variables" in Start Menu
- Add ``CCDC_CSD_DIRECTORY`` as a system variable

**macOS/Linux:**
Add to your shell configuration file (``~/.bashrc``, ``~/.zshrc``, etc.):

.. code-block:: bash

   echo 'export CCDC_CSD_DIRECTORY="/path/to/your/csd/database"' >> ~/.bashrc

Step 5: GPU Setup (Optional but Recommended)
--------------------------------------------

CUDA Installation
~~~~~~~~~~~~~~~~~

For GPU acceleration:

1. **Check GPU Compatibility**:

.. code-block:: bash

   nvidia-smi  # Should show your GPU info

2. **Install CUDA Toolkit** (version 11.6 or later):
   - Download from `NVIDIA CUDA <https://developer.nvidia.com/cuda-toolkit>`_
   - Follow platform-specific installation instructions

3. **Install PyTorch with CUDA**:

.. code-block:: bash

   pip uninstall torch torchvision torchaudio  # Remove CPU version if installed
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

4. **Verify GPU Setup**:

.. code-block:: bash

   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

Step 6: Verify Installation
---------------------------

Quick Verification Script
~~~~~~~~~~~~~~~~~~~~~~~~~

Save this script as ``verify_installation.py``:

.. code-block:: python

   #!/usr/bin/env python3
   """Quick installation verification for CSA."""

   def test_core_imports():
       """Test essential Python libraries."""
       try:
           import numpy
           import torch
           import h5py
           import pandas
           print("✓ Core scientific libraries installed")
           return True
       except ImportError as e:
           print(f"✗ Missing library: {e}")
           return False

   def test_ccdc_api():
       """Test CCDC API access."""
       try:
           from ccdc import io
           print("✓ CCDC API accessible")
           return True
       except ImportError:
           print("✗ CCDC API not found - check CCDC installation")
           return False

   def test_csd_database():
       """Test CSD database connection."""
       try:
           from ccdc import io
           reader = io.EntryReader('CSD')
           count = len(reader)
           print(f"✓ CSD database accessible ({count:,} entries)")
           return True
       except Exception as e:
           print(f"✗ CSD database error: {e}")
           return False

   def test_gpu():
       """Test GPU availability."""
       try:
           import torch
           if torch.cuda.is_available():
               device = torch.cuda.get_device_name(0)
               print(f"✓ GPU available: {device}")
           else:
               print("! No GPU available (CPU-only mode)")
           return True
       except Exception as e:
           print(f"✗ GPU test error: {e}")
           return False

   def test_csa_modules():
       """Test CSA module imports."""
       try:
           import sys
           import os
           
           # Add CSA source directory to path
           csa_src = os.path.join(os.path.dirname(__file__), 'src')
           if os.path.exists(csa_src):
               sys.path.insert(0, csa_src)
           
           from csa_config import ExtractionConfig
           from crystal_analyzer import CrystalAnalyzer
           print("✓ CSA modules importable")
           return True
       except ImportError as e:
           print(f"✗ CSA module error: {e}")
           return False

   if __name__ == "__main__":
       print("CSA Installation Verification")
       print("=" * 40)
       
       all_tests = [
           test_core_imports,
           test_ccdc_api,
           test_csd_database,
           test_gpu,
           test_csa_modules
       ]
       
       passed = 0
       for test in all_tests:
           if test():
               passed += 1
           print()
       
       print("=" * 40)
       print(f"Tests passed: {passed}/{len(all_tests)}")
       
       if passed == len(all_tests):
           print("🎉 Installation verification successful!")
           print("Ready to run CSA analyses!")
       elif passed >= 3:  # Core functionality works
           print("⚠️ Partial success - core functionality available")
           print("Some features may not work optimally")
       else:
           print("❌ Installation issues detected")
           print("Please review error messages above")

Run the verification:

.. code-block:: bash

   python verify_installation.py

Expected Output
~~~~~~~~~~~~~~

Successful installation should show:

.. code-block:: text

   CSA Installation Verification
   ========================================
   ✓ Core scientific libraries installed
   
   ✓ CCDC API accessible
   
   ✓ CSD database accessible (1,234,567 entries)
   
   ✓ GPU available: NVIDIA GeForce RTX 3080
   
   ✓ CSA modules importable
   
   ========================================
   Tests passed: 5/5
   🎉 Installation verification successful!
   Ready to run CSA analyses!

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**CCDC API Import Error**

.. code-block:: text

   ImportError: No module named 'ccdc'

**Solutions:**
1. Verify CCDC software is installed
2. Check CCDC Python API installation
3. Ensure Python path includes CCDC modules
4. Verify CCDC license is active

**CUDA/GPU Issues**

.. code-block:: text

   RuntimeError: CUDA not available

**Solutions:**
1. Install NVIDIA GPU drivers
2. Install CUDA toolkit (11.6+)
3. Reinstall PyTorch with CUDA support
4. Check GPU compatibility with CUDA

**CSD Database Access**

.. code-block:: text

   CSDError: Cannot open database

**Solutions:**
1. Check ``CCDC_CSD_DIRECTORY`` environment variable
2. Verify database files exist and are readable
3. Ensure proper file permissions
4. Check CCDC license status

**Memory/Performance Issues**

**If you encounter memory problems:**
1. Close other applications
2. Use smaller batch sizes in configurations
3. Consider upgrading system memory
4. Enable virtual memory/swap if needed

Getting Additional Help
~~~~~~~~~~~~~~~~~~~~~~

If installation problems persist:

1. **Check system requirements** - Ensure your system meets minimum requirements
2. **Review error messages** - Look for specific error codes or messages
3. **Consult CCDC documentation** - For CCDC-specific issues
4. **Try minimal installation** - Install only essential dependencies first
5. **Report installation issues** - Submit detailed bug reports with system information

Next Steps
----------

After successful installation:

1. **Test with quickstart** - Run your first analysis (:doc:`quickstart`)
2. **Learn configuration** - Understand CSA settings (:doc:`configuration`)
3. **Explore examples** - Try provided example workflows
4. **Join the community** - Get help and share experiences

.. note::
   Keep your virtual environment activated whenever using CSA:
   
   .. code-block:: bash
   
      # Activate before each CSA session
      source csa_env/bin/activate  # macOS/Linux
      csa_env\Scripts\activate     # Windows

Continue to :doc:`quickstart` to run your first CSA analysis!