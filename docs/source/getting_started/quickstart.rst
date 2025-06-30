Quickstart
==========

Run your first analysis in 5 minutes!

Basic Example
-------------

.. code-block:: python

    from crystal_analyzer import CrystalAnalyzer
    from csa_config import load_config
    
    # Load configuration
    config = load_config('config.json')
    
    # Run analysis
    analyzer = CrystalAnalyzer(config)
    analyzer.extract_data()