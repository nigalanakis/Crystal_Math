# Crystal Structure Analysis (CSA)

[![Documentation Status](https://readthedocs.org/projects/crystal-math/badge/?version=latest)](https://crystal-math.readthedocs.io/en/latest/?badge=latest)
[![Python 3.9](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python framework for extracting, processing, and analyzing molecular crystal structures from the Cambridge Structural Database (CSD).

## 🚀 Key Features

- **High-Performance Pipeline**: GPU-accelerated batch processing with PyTorch
- **CSD Integration**: Direct interface to Cambridge Structural Database
- **Advanced Analytics**: Fragment analysis, intermolecular contacts, and geometric descriptors
- **Efficient Storage**: HDF5-based data management with variable-length datasets
- **Scalable Architecture**: Parallel processing for large datasets

## 📖 Documentation

**📚 [Complete Documentation](https://crystal-math.readthedocs.io/en/latest/index.html)**

The full documentation includes:
- [Getting Started Guide](https://crystal-math.readthedocs.io/en/latest/getting_started/index.html) - Installation and quickstart
- [User Guide](https://crystal-math.readthedocs.io/en/latest/user_guide/index.html) - Core concepts and workflows
- [API Reference](https://crystal-math.readthedocs.io/en/latest/api_reference/index.html) - Complete API documentation
- [Tutorials](https://crystal-math.readthedocs.io/en/latest/tutorials/index.html) - Step-by-step guides
- [Examples](https://crystal-math.readthedocs.io/en/latest/examples/index.html) - Ready-to-run code

## ⚡ Quick Start

```bash
# Install CSA
pip install -e .

# Run analysis
python src/csa_main.py --config your_config.json
```

For detailed installation instructions and requirements, see the [Installation Guide](https://crystal-math.readthedocs.io/en/latest/getting_started/installation.html).

## 🔬 What CSA Does

CSA transforms raw crystallographic data into analysis-ready datasets through a five-stage pipeline:

1. **Family Extraction** - Query and organize CSD structures by chemical families
2. **Similarity Clustering** - Group structures by 3D packing similarity
3. **Representative Selection** - Choose optimal structures using statistical metrics
4. **Data Extraction** - Extract atomic coordinates, bonds, and intermolecular contacts
5. **Feature Engineering** - Compute advanced geometric and topological descriptors

## 📋 Requirements

- Python 3.9 (Required for CSD Python API)
- PyTorch (GPU recommended)
- Valid CCDC license for CSD access
- HDF5 and related dependencies

See the [full requirements](https://crystal-math.readthedocs.io/en/latest/getting_started/installation.html#requirements) in the documentation.

## 🤝 Contributing

Contributions are welcome! Please see our [contributing guidelines](https://crystal-math.readthedocs.io/en/latest/technical_details/index.html#contributing) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: CSA requires a valid Cambridge Crystallographic Data Centre (CCDC) license for full functionality.
