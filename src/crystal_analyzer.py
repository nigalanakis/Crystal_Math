"""
Module: crystal_analyzer.py 

Main orchestration logic for extracting and processing molecular-crystal data
from the Cambridge Structural Database (CSD).

This module defines the CrystalAnalyzer class, which orchestrates the end-to-end
pipeline for:
- Extraction of refcode families
- Clustering of structures
- Extraction of structure-specific data
- Post-extraction processing (e.g., computing fragment properties)

Dependencies
------------
pandas
torch
csa_config
csd_operations
structure_data_extractor
structure_post_extraction_processor
"""

import logging
import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import Type

import torch

from csa_config import ExtractionConfig
from csd_operations import CSDOperations
from structure_data_extractor import StructureDataExtractor
from structure_post_extraction_processor import StructurePostExtractionProcessor

logger = logging.getLogger(__name__)


class CrystalAnalyzer:
    """
    Orchestrates the end-to-end extraction and processing pipeline for molecular-
    crystal data from the CSD.

    Attributes
    ----------
    extraction_config : ExtractionConfig
        Controls which extraction substeps to run, batch sizes, file paths, and
        CSD filtering criteria.
    csd_ops : CSDOperations
        Handles direct interactions with the CSD (refcode families, downloads, etc.).
    extractor : StructureDataExtractor
        Performs detailed per-structure data extraction and parsing into HDF5.
    data_dir : pathlib.Path
        Directory where intermediate and output data (CSV, HDF5) are stored.
    """

    def __init__(
            self,
            extraction_config: ExtractionConfig,
            csd_ops_cls: Type[CSDOperations] = CSDOperations,
            extractor_cls: Type[StructureDataExtractor] = StructureDataExtractor,
        ):
        """
        Initialize the CrystalAnalyzer pipeline with specified configurations.
    
        Parameters
        ----------
        extraction_config : ExtractionConfig
            Configuration object controlling which extraction substeps to run,
            batch sizes, file paths, and CSD filtering criteria.
        csd_ops_cls : Type[CSDOperations], optional
            Class implementing CSD operations. Default is CSDOperations.
        extractor_cls : Type[StructureDataExtractor], optional
            Class for extracting structure-specific data. Default is
            StructureDataExtractor.
    
        Raises
        ------
        RuntimeError
            If any batched computation fails (e.g., OOM) or if shape mismatches
            occur when writing back to HDF5.
        IOError
            If appending to or reading from the HDF5 file fails.
    
        Notes
        -----
        Data flows::
    
            raw_HDF5 -> (load into torch tensors on CPU) -> run batch computations
                     -> append new datasets to HDF5 -> log memory utilization and
                     batch progress
        """

        self.extraction_config = extraction_config

        # Resolve paths and create directories
        self.data_dir = Path(self.extraction_config.data_directory)
        self._setup_directories()

        # Instantiate low‐level CSD operations handler
        self.csd_ops = csd_ops_cls(
            data_directory=self.data_dir,
            data_prefix=self.extraction_config.data_prefix
        )

        # Instantiate the StructureDataExtractor for detailed per‐structure data
        h5_path = self.data_dir / f"{self.extraction_config.data_prefix}.h5"
        self.extractor = extractor_cls(
            hdf5_path=h5_path,
            filters={
                **self.extraction_config.filters,
                "data_directory": str(self.extraction_config.data_directory),
                "data_prefix":    self.extraction_config.data_prefix
            },
            batch_size=self.extraction_config.extraction_batch_size
        )

    def _setup_directories(self) -> None:
        """
        Verify and create (if needed) the output directory before any file I/O.

        This method checks:
          - extraction_config.data_directory / “raw” subfolder for downloaded CSD files
          - extraction_config.data_directory / “processed” subfolder for HDF5 outputs

        If any directory does not exist, it is created. Logs an INFO message for each
        new directory created.

        Raises
        ------
        OSError
            If directory creation fails due to permission issues or invalid paths.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def extract_data(self) -> None:
        """
        Execute all data-extraction substeps specified by extraction_config.actions.
    
        The sequence of substeps is:
        1. _extract_refcode_families     (if actions.get("get_refcode_families") is True)
        2. _cluster_refcode_families     (if actions.get("cluster_refcode_families") is True)
        3. _extract_unique_structures    (if actions.get("get_unique_structures") is True)
        4. _extract_structure_data       (if actions.get("get_structure_data") is True)
        5. _post_extraction_process      (if actions.get("post_extraction_process") is True)
    
        During each substep, corresponding CSV/HDF5 files are generated (refcode lists,
        clustered families, per-structure atom lists, fragment datasets, etc.). The
        elapsed time for the entire pipeline is logged at INFO level.
    
        Raises
        ------
        Exception
            If any substep fails (e.g., network error fetching from CSD, parsing error).
        """
        try:
            logger.info("Starting data extraction pipeline...")
            start = datetime.now()

            if self.extraction_config.actions.get("get_refcode_families"):
                self._extract_refcode_families()
            if self.extraction_config.actions.get("cluster_refcode_families"):
                self._cluster_refcode_families()
            if self.extraction_config.actions.get("get_unique_structures"):
                self._extract_unique_structures()
            if self.extraction_config.actions.get("get_structure_data"):
                self._extract_structure_data()
            if self.extraction_config.actions.get("post_extraction_process"):
                self._post_extraction_process()

            duration = datetime.now() - start
            logger.info(f"Data extraction completed in {duration}")
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise

    def _extract_refcode_families(self) -> pd.DataFrame:
        """
        Query CSD to retrieve all refcode families, save to disk, and return.
    
        This method performs the following steps:
        - Invoke self.csd_ops.get_refcode_families_df()
        - Receive a DataFrame with columns ['family_id', 'refcode']
        - Write the DataFrame to disk at:
          extraction_config.data_directory /
          f"{extraction_config.data_prefix}_refcode_families.csv"
        - Log the number of families retrieved at INFO level
    
        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
            - family_id : Unique integer or string ID for each refcode family
            - refcode    : CSD refcode belonging to that family
        """
        logger.info("Extracting refcode families into DataFrame...")
        df = self.csd_ops.get_refcode_families_df()
        self.csd_ops.save_refcode_families_csv()
        n_structures = len(df)
        n_families = df['family_id'].nunique()
        logger.info(f"Extracted {n_structures} structures across {n_families} families")
        return df

    def _cluster_refcode_families(self) -> None:
        """
        Group structures within each refcode family according to packing similarity.
    
        This method performs the following steps:
        - Read the CSV produced by _extract_refcode_families()
        - For each family_id, call self.csd_ops.cluster_families(family_id,
          output_path) to perform clustering of atomic coordinates.
        - Save clustering results to:
          extraction_config.data_directory /
          f"{extraction_config.data_prefix}_clustered_families.csv"
        - Log the number of clusters and cluster sizes at INFO level.
    
        Raises
        ------
        RuntimeError
            If clustering fails for any family (e.g., insufficient data, corrupted CIF).
        """
        try:
            logger.info("Clustering refcode families...")
            clustered = self.csd_ops.cluster_families(self.extraction_config.filters)
            logger.info(f"Refcode families clustered into {len(clustered)} groups.")
        except Exception as e:
            logger.error(f"Clustering of refcode families failed. {e}")
            raise

    def _extract_unique_structures(self) -> None:
        """
        Retrieve unique crystal structures for each cluster representative.
    
        This method performs the following steps:
        - Read the clustered families CSV to identify one representative refcode per
          cluster.
        - For each representative refcode:
          - Use self.csd_ops.get_unique_structures() to fetch atomic coordinates,
            symmetry operators, and other metadata.
          - Save the raw CIF to:
            extraction_config.data_directory /
            f"{extraction_config.data_prefix}_structures/{refcode}.cif"
        - Update and log status (total structures fetched, failures, retries).
    
        Raises
        ------
        IOError
            If any CIF fails to download or write to disk.
        """
        logger.info("Selecting unique structures …")
        df_unique = self.csd_ops.get_unique_structures(
            self.extraction_config.filters,
            method="vdWFV"
        )
        logger.info(
            "Unique structures selected: %d structures across %d families",
            len(df_unique),
            df_unique['family_id'].nunique()
        )

    def _extract_structure_data(self) -> None:
        """
        Parse each downloaded CIF and extract fundamental structure data into HDF5.
    
        For each CIF in extraction_config.data_directory:
        - Use StructureDataExtractor to read atomic labels, fractional coordinates,
          symmetry operations, lattice parameters, and partial charges.
        - Organize the extracted data into a pandas DataFrame.
        - Batch-write the data to:
          extraction_config.data_directory /
          f"{extraction_config.data_prefix}_structure_data.h5"
        - Log the total number of structures processed and any parse errors.
    
        This method ensures that all per-structure numerics (coords, masks, labels)
        are stored in GPU-friendly formats for further GPU processing.
    
        Raises
        ------
        ValueError
            If CIF parsing yields inconsistent shapes (e.g., mismatched atom count vs.
            mask).
        IOError
            If HDF5 write fails due to disk space or file permissions.
        """
        h5_path = self.data_dir / f"{self.extraction_config.data_prefix}.h5"
        logger.info(f"Extracting detailed structure data into {h5_path} …")
        self.extractor.run()
        logger.info(f"Detailed structure data extracted and saved to {h5_path}")

    def _post_extraction_process(self) -> None:
        """
        Perform all post-extraction computations on the raw structure data.
    
        This step typically includes:
        - Fragment identification (rigid-fragment or molecular fragment detection)
        - Computation of fragment centers of mass (Cartesian & fractional)
        - Computation of fragment inertia tensors, eigenvalues, and quaternions
        - Computation of all intermolecular contacts and hydrogen-bond identification
        - Computation of distances/vectors from each contact atom to fragment COM
        - Augmentation of HDF5 datasets with new variable-length datasets for
          fragment-related properties
    
        Notes
        -----
        Data flows::
    
            raw_HDF5 -> (load into torch tensors on CPU) -> run batch computations
                     -> append new datasets to HDF5 -> log memory utilization and
                     batch progress
    
        Raises
        ------
        RuntimeError
            If any batched computation fails (e.g., OOM) or if shape mismatches occur
            when writing back to HDF5.
        IOError
            If appending to or reading from the HDF5 file fails.
        """
        h5_path = self.data_dir / f"{self.extraction_config.data_prefix}.h5"
        proc = StructurePostExtractionProcessor(
            hdf5_path=h5_path,
            batch_size=self.extraction_config.post_extraction_batch_size,
            device=torch.device("cuda")
        )
        proc.run()
