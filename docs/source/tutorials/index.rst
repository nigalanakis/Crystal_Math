Tutorials
=========

Step-by-step tutorials for common CSA analysis workflows. These hands-on guides walk you through complete analysis scenarios, from data extraction to publication-ready results.

.. note::
   
   All tutorials include downloadable configuration files and example datasets. Follow along with real data to master CSA workflows.

Getting Started Tutorials
-------------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Your First Analysis
      :link: first_analysis
      :link-type: doc

      Complete walkthrough of the five-stage CSA pipeline with a small organic dataset.

   .. grid-item-card:: ⚙️ Configuration Mastery
      :link: configuration_tutorial
      :link-type: doc

      Learn to optimize configurations for different research scenarios and hardware.

   .. grid-item-card:: 📊 Data Exploration
      :link: data_exploration
      :link-type: doc

      Master the art of exploring and understanding CSA datasets.

   .. grid-item-card:: 🔍 Quality Control
      :link: quality_control
      :link-type: doc

      Ensure reliable results through systematic data validation and filtering.

Domain-Specific Tutorials
-------------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 💊 Pharmaceutical Analysis
      :link: pharmaceutical_analysis
      :link-type: doc

      Analyze drug polymorphs, co-crystals, and pharmaceutical relevant crystal packings.

   .. grid-item-card:: 🧪 Organic Chemistry
      :link: organic_chemistry
      :link-type: doc

      Study molecular conformations, intermolecular interactions, and packing preferences.

   .. grid-item-card:: 🏗️ Materials Science
      :link: materials_science
      :link-type: doc

      Investigate porous materials, metal-organic frameworks, and solid-state properties.

   .. grid-item-card:: 🔬 Polymorphism Studies
      :link: polymorphism_studies
      :link-type: doc

      Comprehensive polymorphism analysis including energy landscapes and packing similarity.

Advanced Analysis Tutorials
---------------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🤖 Machine Learning
      :link: machine_learning
      :link-type: doc

      Apply ML techniques to crystal structure data for property prediction and clustering.

   .. grid-item-card:: 📈 Statistical Analysis
      :link: statistical_analysis
      :link-type: doc

      Advanced statistical methods for crystal structure datasets and hypothesis testing.

   .. grid-item-card:: 🌐 Comparative Studies
      :link: comparative_studies
      :link-type: doc

      Compare crystal structures across different chemical families and conditions.

   .. grid-item-card:: 📋 Custom Analysis
      :link: custom_analysis
      :link-type: doc

      Develop your own analysis workflows and integrate with external tools.

   .. grid-item-card:: ⚡ High-Performance Computing
      :link: hpc_workflows
      :link-type: doc

      Scale CSA analyses to HPC clusters and optimize for large datasets.

Specialized Workflows
--------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🎯 Fragment Analysis
      :link: fragment_analysis
      :link-type: doc

      Deep dive into molecular fragment properties and rigid body analysis.

   .. grid-item-card:: 🔗 Hydrogen Bonding
      :link: hydrogen_bonding
      :link-type: doc

      Comprehensive analysis of hydrogen bonding patterns and networks.

   .. grid-item-card:: 📏 Geometric Descriptors
      :link: geometric_descriptors
      :link-type: doc

      Compute and interpret geometric and topological molecular descriptors.

   .. grid-item-card:: 🧮 Contact Analysis
      :link: contact_analysis
      :link-type: doc

      Analyze intermolecular contacts, packing arrangements, and crystal engineering.

Tutorial Structure
-----------------

Each tutorial follows a consistent structure for maximum learning efficiency:

**Learning Objectives**
   Clear goals for what you'll achieve and learn

**Prerequisites**  
   Required knowledge and software setup

**Dataset Description**
   Information about the example dataset used

**Step-by-Step Instructions**
   Detailed walkthrough with explanations

**Code Examples**
   Complete, runnable code snippets

**Results Interpretation**
   How to understand and validate your results

**Extensions**
   Ideas for further exploration and customization

**Troubleshooting**
   Common issues and solutions

Quick Start Recommendations
---------------------------

**New to CSA?**
   Start with :doc:`first_analysis` to understand the basic workflow

**Specific Research Goal?**
   Jump to the relevant domain-specific tutorial

**Performance Issues?**
   Check :doc:`hpc_workflows` for optimization strategies

**Custom Requirements?**
   Begin with :doc:`custom_analysis` for workflow development

Tutorial Downloads
------------------

All tutorials include downloadable resources:

**Configuration Files**
   Pre-configured JSON files for each tutorial scenario

**Example Datasets**
   Small datasets for following along with examples

**Analysis Scripts**
   Complete Python scripts for tutorial workflows

**Expected Results**
   Reference outputs for validation

.. code-block:: bash

   # Download tutorial resources
   wget https://csa-tutorials.readthedocs.io/downloads/tutorial_resources.zip
   unzip tutorial_resources.zip
   
   # Or clone from repository
   git clone https://github.com/your-org/csa-tutorials.git
   cd csa-tutorials

Tutorial Support
----------------

**Community Forum**
   Ask questions and share insights with other users

**GitHub Issues**
   Report tutorial problems and suggest improvements

**Documentation Updates**
   Tutorials are regularly updated with new features and best practices

**Video Walkthroughs**
   Selected tutorials include video demonstrations

Contributing Tutorials
---------------------

Help expand the tutorial collection:

**Share Your Workflows**
   Submit successful analysis workflows as new tutorials

**Improve Existing Content**
   Suggest improvements and corrections via pull requests

**Domain Expertise**
   Contribute specialized tutorials for your research area

**Teaching Experience**
   Help improve tutorial pedagogy and clarity

.. code-block:: bash

   # Contribute a new tutorial
   git clone https://github.com/your-org/csa-documentation.git
   cd csa-documentation/docs/source/tutorials
   
   # Create new tutorial file
   cp template_tutorial.rst my_tutorial.rst
   
   # Edit and submit via pull request

Learning Paths
--------------

Structured learning sequences for different goals:

**Research Scientist Path**
   1. :doc:`first_analysis`
   2. :doc:`data_exploration` 
   3. Domain-specific tutorial (pharmaceutical/organic/materials)
   4. :doc:`statistical_analysis`
   5. :doc:`comparative_studies`

**Computational Crystallographer Path**
   1. :doc:`first_analysis`
   2. :doc:`configuration_tutorial`
   3. :doc:`geometric_descriptors`
   4. :doc:`fragment_analysis`
   5. :doc:`custom_analysis`

**Data Scientist Path**
   1. :doc:`first_analysis`
   2. :doc:`data_exploration`
   3. :doc:`machine_learning`
   4. :doc:`statistical_analysis`
   5. :doc:`hpc_workflows`

**Software Developer Path**
   1. :doc:`configuration_tutorial`
   2. :doc:`custom_analysis`
   3. :doc:`hpc_workflows`
   4. API reference sections
   5. Contributing guidelines

Advanced Topics
--------------

Beyond basic tutorials, explore advanced capabilities:

**Integration Workflows**
   - Combining CSA with external crystallographic software
   - Database integration and automated workflows
   - Cloud computing and containerization

**Method Development**
   - Implementing new analysis algorithms
   - Custom descriptor development
   - Performance optimization techniques

**Educational Applications**
   - Classroom exercises and assignments
   - Interactive notebooks and demonstrations
   - Assessment and evaluation methods

Feedback and Improvement
-----------------------

Tutorial quality depends on user feedback:

**Rate Tutorials**
   Help others find the most useful content

**Suggest Improvements**
   Identify unclear explanations or missing steps

**Request New Topics**
   Propose tutorials for your specific needs

**Share Success Stories**
   Let us know how tutorials helped your research

.. code-block:: python

   # Feedback form example
   tutorial_feedback = {
       "tutorial_name": "first_analysis",
       "difficulty_rating": 4,  # 1-5 scale
       "clarity_rating": 5,
       "usefulness_rating": 5,
       "suggestions": "Include more visualization examples",
       "would_recommend": True
   }

Getting Help
-----------

If you encounter issues with tutorials:

1. **Check Prerequisites**: Ensure all software requirements are met
2. **Verify Data**: Confirm tutorial datasets are correctly downloaded
3. **Review Error Messages**: Check CSA logs for specific error details
4. **Search Documentation**: Look for solutions in troubleshooting sections
5. **Ask the Community**: Post questions in forums or issue trackers
6. **Contact Support**: Reach out for direct assistance when needed

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   first_analysis
   configuration_tutorial
   data_exploration
   quality_control

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Domain-Specific

   pharmaceutical_analysis
   organic_chemistry
   materials_science
   polymorphism_studies

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced Analysis

   machine_learning
   statistical_analysis
   comparative_studies
   custom_analysis
   hpc_workflows

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Specialized Workflows

   fragment_analysis
   hydrogen_bonding
   geometric_descriptors
   contact_analysis