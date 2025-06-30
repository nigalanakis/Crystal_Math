"""
Module: main.py 

Entry-point script for the Crystal Structure Analysis pipeline.

This module provides a command-line interface to execute the full CSD data
extraction pipeline via `CrystalAnalyzer.extract_data()`.

Examples
--------
python csa_main.py --config /path/to/config.json

Dependencies
------------
crystal_analyzer.CrystalAnalyzer
csa_config.load_config
argparse
logging
pathlib
sys
"""

import argparse
import logging
from logging import StreamHandler, Formatter
from pathlib import Path

from crystal_analyzer import CrystalAnalyzer
from csa_config import load_config

def setup_logging() -> None:
    """
    Configure the root logger for console output.

    This function performs the following actions:
    - Remove any existing handlers from the root logger.
    - Create a StreamHandler to stderr with format:
      "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    - Attach the handler to the root logger.
    - Set the root logger level to INFO.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(
        Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root.addHandler(stream_handler)
    root.setLevel(logging.INFO)

def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace object with the following attribute:
        - config (Path): Path to the JSON configuration file.

    Raises
    ------
    SystemExit
        If argument parsing fails.
    """
    parser = argparse.ArgumentParser(
        description="Run CSD data extraction pipeline"
    )
    parser.add_argument(
        '-c', '--config',
        type=Path,
        default=Path('../config/csa_config.json').expanduser(),
        help="Path to the JSON configuration file"
    )
    return parser.parse_args()

def run_extraction(config_path: Path) -> None:
    """
    Run the CSD data extraction pipeline.

    This function:
    - Loads the extraction configuration from the provided JSON file.
    - Instantiates a CrystalAnalyzer with the configuration.
    - Invokes `analyzer.extract_data()` to perform all extraction steps.

    Parameters
    ----------
    config_path : Path
        Path to the JSON configuration file. Must exist and be readable.

    Raises
    ------
    Exception
        If any error occurs during extraction, it is logged and re-raised.
    """
    logging.info(f"Loading configuration from {config_path}")
    extraction_cfg = load_config(config_path)

    analyzer = CrystalAnalyzer(extraction_config=extraction_cfg)

    logging.info("Starting extraction step...")
    analyzer.extract_data()

def main() -> None:
    """
    Entry point for the csa_main script.

    Workflow
    --------
    1. Configure logging by calling `setup_logging()`.
    2. Parse command-line arguments via `parse_args()`.
    3. Invoke `run_extraction()` with the resolved config path.
    4. Log success or catch and log exceptions before exiting.

    Raises
    ------
    SystemExit
        If argument parsing fails.
    Exception
        If `run_extraction()` throws an error; it is logged before propagation.
    """
    setup_logging()
    args = parse_args()

    try:
        run_extraction(config_path=args.config.resolve())
        logging.info("Data extraction completed successfully.")
    except Exception:
        logging.exception("Data extraction failed with an error.")
        raise

if __name__ == '__main__':
    main()