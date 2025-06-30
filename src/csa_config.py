"""
Module: csa_config.py 

Configuration objects and loader for the Crystal Structure Analysis pipeline.

This module defines:
- ExtractionConfig: dataclass controlling extraction parameters.
- load_config: utility to construct ExtractionConfig from a JSON file.
"""

from dataclasses import dataclass
from typing import Dict, Any, Union
import json
import logging
from pathlib import Path
from inspect import signature

logger = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """
    Configuration settings for the data-extraction pipeline.

    Parameters
    ----------
    data_directory : Path
        Directory under which all raw and intermediate extraction outputs will be stored.
        Subdirectories (e.g. “structures/”, “csv/”) are created automatically.
    data_prefix : str
        Prefix used when naming output files, for example
        ``"{data_prefix}_refcode_families.csv"``.
    actions : Dict[str, bool]
        Flags to enable or skip individual extraction substeps:
        - ``get_refcode_families``
        - ``cluster_refcode_families``
        - ``get_unique_structures``
        - ``get_structure_data``
        - ``post_extraction_process``
    filters : Dict[str, Any]
        Criteria for filtering CSD entries, for example:
        - ``elements`` (List[str]): only structures containing these elements
        - ``min_resolution`` (float): only structures with resolution ≤ this value
        - ``space_groups`` (List[str]): only structures in these space groups
    extraction_batch_size : int
        Number of structures or refcode families to process per batch during extraction
    post_extraction_batch_size : int
        Number of structures to process per batch during post-extraction

    Methods
    -------
    from_json(cls, json_path)
        Load and validate fields from the “extraction” section of a JSON file.
    """
    data_directory: Path
    data_prefix: str
    actions: Dict[str, bool]
    filters: Dict[str, Any]
    extraction_batch_size: int
    post_extraction_batch_size: int

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'ExtractionConfig':
        """
        Load an ExtractionConfig from a JSON file.

        Parameters
        ----------
        json_path : Union[str, Path]
            Path to the JSON configuration file.

        Returns
        -------
        ExtractionConfig
            Instance populated from the “extraction” section.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        KeyError
            If the “extraction” section is missing.
        json.JSONDecodeError
            If the file contains invalid JSON.
        """
        json_path = Path(json_path)
        if not json_path.is_file():
            logger.error(f"Config file not found: {json_path}")
            raise FileNotFoundError(f"Config file not found: {json_path}")

        raw = json.loads(json_path.read_text())
        try:
            config = raw['extraction']
        except KeyError:
            logger.error(f"'extraction' section missing in {json_path}")
            raise

        # Keep only keys valid for this dataclass
        valid_keys = set(signature(cls).parameters)
        cleaned = {k: v for k, v in config.items() if k in valid_keys}

        # Convert data_directory to Path
        if 'data_directory' in cleaned:
            cleaned['data_directory'] = Path(cleaned['data_directory'])

        return cls(**cleaned)


def load_config(config_path: Union[str, Path]) -> ExtractionConfig:
    """
    Read a JSON configuration file and return an ExtractionConfig instance.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the JSON configuration file.

    Returns
    -------
    ExtractionConfig
        Dataclass instance loaded from the “extraction” section.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the “extraction” section is missing.
    json.JSONDecodeError
        If the file contains invalid JSON.
    """
    config_path = Path(config_path)
    extraction_config = ExtractionConfig.from_json(config_path)
    return extraction_config
