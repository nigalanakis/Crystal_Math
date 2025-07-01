Configuration
=============

This guide covers advanced configuration strategies for CSA workflows. While the :doc:`../getting_started/configuration` guide covers basic setup, this section focuses on optimizing configurations for specific research scenarios and performance requirements.

.. note::
   
   This guide assumes familiarity with basic CSA configuration. Review the :doc:`../getting_started/configuration` guide first if you're new to CSA.

Research-Driven Configuration
-----------------------------

Configuring for Specific Research Questions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Polymorphism Studies**

Analyzing different crystal forms of the same molecule:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./polymorphism_study",
        "data_prefix": "polymorphs",
        "actions": {
          "get_refcode_families": true,
          "cluster_refcode_families": false,
          "get_unique_structures": false,
          "get_structure_data": true,
          "post_extraction_process": true
        },
        "filters": {
          "target_z_prime_values": [1, 2, 3, 4],
          "crystal_type": ["homomolecular"],
          "molecule_formal_charges": [0],
          "target_species": ["C", "H", "N", "O"],
          "structure_list": ["csd-unique"]
        }
      }
    }

**Hydrogen Bonding Analysis**

Focusing on structures with strong H-bond donors/acceptors:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./hbond_analysis",
        "data_prefix": "hydrogen_bonds",
        "filters": {
          "target_z_prime_values": [1],
          "crystal_type": ["homomolecular"],
          "molecule_formal_charges": [0],
          "molecule_weight_limit": 400.0,
          "target_species": ["C", "H", "N", "O"],
          "has_hbond_donors": true,
          "has_hbond_acceptors": true
        },
        "extraction_batch_size": 48,
        "post_extraction_batch_size": 24
      }
    }

**Conformational Flexibility**

Studying molecules with rotatable bonds:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./flexibility_study", 
        "data_prefix": "conformers",
        "filters": {
          "target_z_prime_values": [1],
          "crystal_type": ["homomolecular"],
          "molecule_formal_charges": [0],
          "min_rotatable_bonds": 2,
          "max_rotatable_bonds": 8,
          "molecule_weight_limit": 600.0
        }
      }
    }

**Metal-Organic Systems**

Including coordination compounds:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./metal_organic",
        "data_prefix": "coordination", 
        "filters": {
          "target_z_prime_values": [1, 2],
          "crystal_type": ["homomolecular", "organometallic"],
          "molecule_formal_charges": [0, 1, -1, 2, -2],
          "target_species": ["C", "H", "N", "O", "Fe", "Cu", "Zn", "Ni"],
          "exclude_disorder": true
        }
      }
    }

Performance-Driven Configuration
--------------------------------

Hardware-Optimized Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**High-Memory Systems (64GB+ RAM)**

Maximize throughput with large batches:

.. code-block:: json

    {
      "extraction": {
        "extraction_batch_size": 256,
        "post_extraction_batch_size": 128,
        "parallel_workers": 16,
        "cache_intermediate_results": true
      }
    }

**GPU-Accelerated Workstations**

Balance GPU memory and compute:

.. code-block:: json

    {
      "extraction": {
        "extraction_batch_size": 128,
        "post_extraction_batch_size": 64,
        "use_gpu_acceleration": true,
        "gpu_memory_fraction": 0.8
      }
    }

**Cluster/HPC Environments**

Optimize for distributed processing:

.. code-block:: json

    {
      "extraction": {
        "extraction_batch_size": 64,
        "post_extraction_batch_size": 32,
        "parallel_workers": 8,
        "checkpoint_frequency": 1000,
        "restart_on_failure": true
      }
    }

**Limited Resource Systems**

Conservative settings for laptops/workstations:

.. code-block:: json

    {
      "extraction": {
        "extraction_batch_size": 16,
        "post_extraction_batch_size": 8,
        "parallel_workers": 4,
        "memory_optimization": "aggressive"
      }
    }

Workflow-Specific Configurations
---------------------------------

Iterative Development
~~~~~~~~~~~~~~~~~~~~~

**Rapid Prototyping**

Quick testing with small datasets:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./prototype",
        "data_prefix": "test",
        "filters": {
          "target_z_prime_values": [1],
          "crystal_type": ["homomolecular"],
          "molecule_weight_limit": 200.0,
          "max_structures": 100
        },
        "extraction_batch_size": 32,
        "post_extraction_batch_size": 16
      }
    }

**Development Validation**

Testing pipeline changes:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./validation",
        "data_prefix": "dev_test",
        "actions": {
          "get_refcode_families": false,
          "cluster_refcode_families": false,
          "get_unique_structures": false,
          "get_structure_data": true,
          "post_extraction_process": true
        },
        "filters": {
          "structure_list": ["cif", "/path/to/test/structures"]
        },
        "debug_mode": true,
        "verbose_logging": true
      }
    }

Production Workflows
~~~~~~~~~~~~~~~~~~~~

**Large-Scale Screening**

High-throughput analysis of thousands of structures:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./large_scale_screening",
        "data_prefix": "hts_run_001",
        "filters": {
          "target_z_prime_values": [1],
          "crystal_type": ["homomolecular"],
          "molecule_weight_limit": 1000.0
        },
        "extraction_batch_size": 128,
        "post_extraction_batch_size": 64,
        "checkpoint_frequency": 500,
        "compression": "lz4",
        "backup_results": true
      }
    }

**Reproducible Research**

Ensuring consistent results across runs:

.. code-block:: json

    {
      "extraction": {
        "data_directory": "./reproducible_analysis",
        "data_prefix": "paper_dataset_v1",
        "random_seed": 42,
        "deterministic_clustering": true,
        "version_metadata": {
          "csa_version": "2.0.0",
          "csd_version": "2024.1",
          "analysis_date": "2025-01-15",
          "description": "Dataset for publication XYZ"
        }
      }
    }

Advanced Filter Strategies
---------------------------

Chemical Space Sampling
~~~~~~~~~~~~~~~~~~~~~~~~

**Diverse Chemical Sampling**

Ensuring broad coverage of chemical space:

.. code-block:: json

    {
      "extraction": {
        "filters": {
          "target_z_prime_values": [1],
          "crystal_type": ["homomolecular"],
          "molecule_formal_charges": [0],
          "sampling_strategy": "diverse",
          "molecular_descriptors": {
            "weight_range": [100, 800],
            "logp_range": [-2, 6],
            "hbd_range": [0, 5],
            "hba_range": [0, 10],
            "rotatable_bonds_range": [0, 15]
          }
        }
      }
    }

**Focused Chemical Series**

Analyzing structurally related compounds:

.. code-block:: json

    {
      "extraction": {
        "filters": {
          "target_z_prime_values": [1],
          "crystal_type": ["homomolecular"],
          "structural_patterns": [
            "benzene_ring",
            "carboxylic_acid",
            "amide_group"
          ],
          "similarity_threshold": 0.7,
          "scaffold_filtering": true
        }
      }
    }

Quality Control Filters
~~~~~~~~~~~~~~~~~~~~~~~

**High-Quality Structures Only**

Strict quality criteria for reliable analysis:

.. code-block:: json

    {
      "extraction": {
        "filters": {
          "min_resolution": 1.5,
          "max_r_factor": 0.05,
          "exclude_disorder": true,
          "exclude_polymers": true,
          "exclude_solvates": true,
          "min_temperature": 100,
          "max_temperature": 300,
          "quality_flags": ["high_precision", "complete_structure"]
        }
      }
    }

**Experimental Condition Filters**

Controlling for experimental variables:

.. code-block:: json

    {
      "extraction": {
        "filters": {
          "temperature_range": [90, 120],
          "pressure_range": ["ambient"],
          "radiation_type": ["Mo_Ka", "Cu_Ka"],
          "exclude_neutron": false,
          "min_completeness": 0.95,
          "min_observed_reflections": 1000
        }
      }
    }

Configuration Templates
-----------------------

Template Library
~~~~~~~~~~~~~~~~

CSA includes pre-defined configuration templates for common use cases:

**Pharmaceutical Template**

.. code-block:: bash

    # Copy pharmaceutical template
    cp templates/pharmaceutical.json my_pharma_config.json
    
    # Edit specific parameters
    nano my_pharma_config.json

**Materials Science Template**

.. code-block:: bash

    # Copy materials template  
    cp templates/materials.json my_materials_config.json

**Organic Chemistry Template**

.. code-block:: bash

    # Copy organic template
    cp templates/organic.json my_organic_config.json

Custom Template Creation
~~~~~~~~~~~~~~~~~~~~~~~~

**Creating Project Templates**

.. code-block:: python

    # create_template.py
    import json
    from pathlib import Path

    def create_project_template(project_name, base_template="organic"):
        """Create a custom configuration template."""
        
        template_dir = Path("templates")
        base_config = json.loads((template_dir / f"{base_template}.json").read_text())
        
        # Customize for project
        base_config["extraction"]["data_directory"] = f"./{project_name}"
        base_config["extraction"]["data_prefix"] = project_name
        
        # Save custom template
        output_path = template_dir / f"{project_name}.json"
        output_path.write_text(json.dumps(base_config, indent=2))
        
        return output_path

Configuration Validation
-------------------------

Pre-Flight Checks
~~~~~~~~~~~~~~~~~

**Validate Before Running**

.. code-block:: python

    from csa_config import ExtractionConfig, validate_configuration

    def validate_config_file(config_path):
        """Comprehensive configuration validation."""
        try:
            # Load and parse
            config = ExtractionConfig.from_json(config_path)
            
            # Check resource requirements
            estimated_memory = estimate_memory_usage(config)
            estimated_time = estimate_runtime(config)
            
            print(f"Configuration valid!")
            print(f"Estimated memory: {estimated_memory:.1f} GB")
            print(f"Estimated runtime: {estimated_time:.1f} hours")
            
            # Check for common issues
            warnings = check_common_issues(config)
            if warnings:
                print("Warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
                    
        except Exception as e:
            print(f"Configuration error: {e}")
            return False
            
        return True

**Resource Estimation**

.. code-block:: python

    def estimate_resources(config):
        """Estimate computational requirements."""
        
        # Estimate dataset size
        estimated_structures = estimate_structure_count(config.filters)
        
        # Memory requirements
        memory_per_structure = 2.5  # MB average
        peak_memory = (estimated_structures * memory_per_structure * 
                      config.extraction_batch_size / 1024)  # GB
        
        # Runtime estimation  
        structures_per_hour = 1000  # Typical throughput
        estimated_hours = estimated_structures / structures_per_hour
        
        return {
            'structures': estimated_structures,
            'peak_memory_gb': peak_memory,
            'estimated_hours': estimated_hours
        }

Configuration Management
------------------------

Version Control
~~~~~~~~~~~~~~~

**Track Configuration Changes**

.. code-block:: bash

    # Initialize git repository for configs
    mkdir csa_configurations
    cd csa_configurations
    git init
    
    # Add configurations
    cp ../my_analysis.json ./
    git add my_analysis.json
    git commit -m "Initial analysis configuration"
    
    # Track changes
    git log --oneline my_analysis.json

**Configuration Branches**

.. code-block:: bash

    # Create branches for different experiments
    git checkout -b experiment_1
    # Modify configuration...
    git commit -m "Experiment 1: increased batch size"
    
    git checkout -b experiment_2  
    # Different modifications...
    git commit -m "Experiment 2: additional filters"

Environment-Specific Configs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Development vs Production**

.. code-block:: python

    # config_manager.py
    import os
    import json
    from pathlib import Path

    def load_environment_config(base_config_path, environment="development"):
        """Load configuration with environment-specific overrides."""
        
        base_config = json.loads(Path(base_config_path).read_text())
        
        # Look for environment-specific overrides
        env_config_path = Path(base_config_path).with_suffix(f'.{environment}.json')
        
        if env_config_path.exists():
            env_overrides = json.loads(env_config_path.read_text())
            base_config = merge_configs(base_config, env_overrides)
            
        # Apply environment variables
        base_config = apply_env_vars(base_config)
        
        return base_config

    def merge_configs(base, overrides):
        """Deep merge configuration dictionaries."""
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base:
                base[key] = merge_configs(base[key], value)
            else:
                base[key] = value
        return base

Best Practices
--------------

Configuration Organization
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use descriptive naming**: Include project, date, and version in config names
2. **Document parameters**: Add comments explaining non-standard settings
3. **Version control**: Track configuration changes alongside code
4. **Environment separation**: Maintain different configs for dev/test/prod
5. **Template usage**: Start from validated templates rather than from scratch

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Profile first**: Measure before optimizing batch sizes
2. **Incremental scaling**: Gradually increase batch sizes to find optimal settings
3. **Monitor resources**: Watch memory and GPU utilization during runs
4. **Cache strategies**: Use appropriate caching for repeated analyses
5. **Checkpoint frequently**: Save progress for long-running analyses

Reproducibility
~~~~~~~~~~~~~~~

1. **Pin versions**: Document CSA, PyTorch, and CCDC versions
2. **Set random seeds**: Ensure deterministic behavior
3. **Save complete configs**: Store exact configuration with results
4. **Document environment**: Record hardware and software environment
5. **Validate consistency**: Test configurations across different systems

Troubleshooting
---------------

Common Configuration Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Problems**

.. code-block:: text

    ERROR: CUDA out of memory

*Solutions*:
- Reduce ``extraction_batch_size`` and ``post_extraction_batch_size``
- Enable memory optimization: ``"memory_optimization": "aggressive"``
- Use CPU processing: ``"use_gpu_acceleration": false``

**Performance Issues**

.. code-block:: text

    WARNING: Very slow processing detected

*Solutions*:
- Increase batch sizes if memory allows
- Enable GPU acceleration
- Reduce dataset size with more restrictive filters
- Use faster storage (SSD) for data directory

**Filter Problems**

.. code-block:: text

    ERROR: No structures match the specified filters

*Solutions*:
- Relax restrictive filters gradually
- Check filter syntax and valid values
- Use filter validation tools before running
- Start with broader filters and refine iteratively

**File System Issues**

.. code-block:: text

    ERROR: Permission denied writing to data directory

*Solutions*:
- Check directory permissions: ``chmod 755 /path/to/data``
- Ensure sufficient disk space
- Use absolute paths in configuration
- Verify write access: ``touch /path/to/data/test_file``

Next Steps
----------

With advanced configuration mastery:

1. **Optimize for your hardware**: Find the best performance settings
2. **Develop analysis templates**: Create reusable configurations  
3. **Automate workflows**: Script configuration generation and validation
4. **Share configurations**: Collaborate with standardized templates
5. **Monitor and improve**: Continuously optimize based on usage patterns

See Also
--------

:doc:`../getting_started/configuration` : Basic configuration guide
:doc:`basic_analysis` : Apply configurations in analysis workflows
:doc:`../technical_details/performance` : Performance optimization details  
:doc:`../api_reference/core/csa_config` : Configuration API reference