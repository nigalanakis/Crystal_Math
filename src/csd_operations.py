"""
Module: csd_operations.py 

High-level interface for interacting with the Cambridge Structural Database (CSD).

This module provides functionality to:
- Extract and filter refcode families
- Cluster structures by packing similarity
- Select representative structures using the vdWFV metric
- Save intermediate results to CSV

Dependencies
------------
pandas
networkx
ccdc
csd_structure_validator
"""

from pathlib import Path
from typing import Dict, Union, List, Tuple, Optional
import logging
import pandas as pd
import networkx as nx
import os
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from ccdc import io
from ccdc.crystal import PackingSimilarity
from csd_structure_validator import StructureValidator

logger = logging.getLogger(__name__)

@dataclass
class SimilaritySettings:
    """
    Configuration settings for packing similarity comparisons of crystal structures.

    Parameters
    ----------
    distance_tolerance : float, default=0.2
        Maximum allowed deviation in atomic distances (Å) when comparing packings.
    angle_tolerance : float, default=20.0
        Maximum allowed angular deviation (degrees) between molecular orientations.
    ignore_bond_types : bool, default=True
        If True, matching bond orders are not required for similarity.
    ignore_hydrogen_counts : bool, default=True
        If True, differences in hydrogen counts are ignored.
    ignore_hydrogen_positions : bool, default=True
        If True, explicit hydrogen coordinate differences are ignored.
    packing_shell_size : int, default=15
        Number of molecules considered in each packing-shell comparison.
    ignore_spacegroup : bool, default=True
        If True, space-group designations are not required to match.
    normalise_unit_cell : bool, default=True
        If True, unit cell parameters are normalized before comparison.
    """
    distance_tolerance: float = 0.2
    angle_tolerance: float = 20.0
    ignore_bond_types: bool = True
    ignore_hydrogen_counts: bool = True
    ignore_hydrogen_positions: bool = True
    packing_shell_size: int = 15
    ignore_spacegroup: bool = True
    normalise_unit_cell: bool = True


class CSDOperations:
    """
    High-level interface for querying, validating, clustering, and selecting
    crystal structures from the Cambridge Structural Database (CSD).

    Attributes
    ----------
    data_directory : Path
        Base directory for reading and writing CSD-related files.
    data_prefix : str
        Prefix used when naming output files.
    reader : io.EntryReader
        CCDC EntryReader instance connected to the "CSD" database.
    similarity_engine : PackingSimilarity
        Engine for computing pairwise packing similarity.
    """

    def __init__(self, data_directory: Union[str, Path], data_prefix: str):
        """
        Initialize CSDOperations with target directory and filename prefix.

        Parameters
        ----------
        data_directory : Union[str, Path]
            Directory under which all CSV outputs will be saved.
        data_prefix : str
            Prefix for generated CSV filenames (e.g., "<prefix>_refcode_families.csv").
        """
        self.data_directory = Path(data_directory)
        self.data_prefix = data_prefix
        self.reader = io.EntryReader("CSD")
        self._setup_similarity_engine()

    def _setup_similarity_engine(self, settings: Optional[SimilaritySettings] = None):
        """
        Instantiate and configure the packing similarity engine.

        Parameters
        ----------
        settings : SimilaritySettings, optional
            Custom threshold settings. If None, default settings are used.
        """
        settings = settings or SimilaritySettings()
        self.similarity_engine = PackingSimilarity()
        self.similarity_engine.settings.distance_tolerance = settings.distance_tolerance
        self.similarity_engine.settings.angle_tolerance = settings.angle_tolerance
        self.similarity_engine.settings.ignore_bond_types = settings.ignore_bond_types
        self.similarity_engine.settings.ignore_hydrogen_counts = settings.ignore_hydrogen_counts
        self.similarity_engine.settings.ignore_hydrogen_positions = settings.ignore_hydrogen_positions
        self.similarity_engine.settings.packing_shell_size = settings.packing_shell_size

    def get_refcode_families_df(self) -> pd.DataFrame:
        """
        Query the CSD and group entries by base refcode.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - family_id : str, first six characters of the refcode
            - refcode   : str, full CSD refcode
        """
        records = []
        for entry in self.reader:
            identifier = entry.identifier
            family_id = identifier[:6]
            records.append({"family_id": family_id, "refcode": identifier})
        return pd.DataFrame(records)

    def save_refcode_families_csv(self, df: Optional[pd.DataFrame] = None, filename: Optional[Union[str, Path]] = None) -> None:
        """
        Write the refcode-families DataFrame to a CSV file.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame to save. If None, uses get_refcode_families_df().
        filename : Union[str, Path], optional
            Full file path for output. If None, defaults to
            data_directory / f"{data_prefix}_refcode_families.csv".

        Raises
        ------
        OSError
            If writing to disk fails.
        """
        if df is None:
            df = self.get_refcode_families_df()
        output_path = Path(filename) if filename else self.data_directory / f"{self.data_prefix}_refcode_families.csv"
        self.data_directory.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def filter_families_by_size(self, df: pd.DataFrame, min_size: int = 2) -> pd.DataFrame:
        """
        Exclude families with fewer than a specified number of members.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ['family_id', 'refcode'].
        min_size : int, default=2
            Minimum number of members for a family to be retained.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.

        Raises
        ------
        KeyError
            If 'family_id' column is missing.
        """
        counts = df['family_id'].value_counts()
        valid = counts[counts >= min_size].index
        return df[df['family_id'].isin(valid)]

    def cluster_families(self, filters: Dict) -> Dict[str, List[List[str]]]:
        """
        Perform packing similarity clustering on each refcode family.

        Workflow
        --------
        1. Load initial refcode families CSV.
        2. Group refcodes by 'family_id'.
        3. For each group, validate entries and build a similarity graph.
        4. Identify connected components as clusters.
        5. Save clustered results to CSV.

        Parameters
        ----------
        filters : Dict[str, Any]
            Criteria for structure validation.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['family_id', 'refcode', 'cluster_id'].

        Raises
        ------
        FileNotFoundError
            If the initial CSV is missing.
        RuntimeError
            If clustering fails for any family.
        """
        df_path = self.data_directory / f"{self.data_prefix}_refcode_families.csv"
        if not df_path.exists():
            raise FileNotFoundError(f"Refcode families CSV not found: {df_path}")
        df = pd.read_csv(df_path)
        families = df.groupby("family_id")["refcode"].apply(list).to_dict()

        args = [(fam_id, refcodes, filters) for fam_id, refcodes in families.items()]
        max_workers = os.cpu_count() - 4 or 4
      
        records = []                                    # ← NEW
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for fam_id, clusters in executor.map(_process_single_family, args):
                for cluster_idx, cluster in enumerate(clusters, start=1):
                    for refcode in cluster:
                        records.append(
                            {"family_id": fam_id, "refcode": refcode, "cluster_id": cluster_idx}
                        )
        
        df_clusters = pd.DataFrame(records)             
        self._save_clustered_families(df_clusters)     
        return df_clusters

    def _check_structure(self, identifier: str, filters: Dict, entry: Optional[io.Entry] = None) -> bool:
        """
        Validate a CSD entry against filter criteria.

        Parameters
        ----------
        identifier : str
            CSD refcode.
        filters : Dict[str, Any]
            Validation criteria.
        entry : io.Entry, optional
            Preloaded CSD entry. If None, loaded internally.

        Returns
        -------
        bool
            True if the structure is valid, False otherwise.

        Raises
        ------
        Exception
            If validation fails unexpectedly.
        """
        if entry is None:
            entry = self.reader.entry(identifier)
        validator = StructureValidator(filters)
        result = validator.validate(entry.crystal, entry.molecule)
        return result.is_valid

    def _save_clustered_families(self, df: pd.DataFrame) -> None:
        """
        Save clustered families DataFrame to CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Must include ['family_id', 'refcode', 'cluster_id'].

        Raises
        ------
        OSError
            If file writing fails.
        """
        output_file = (
            self.data_directory / f"{self.data_prefix}_refcode_families_clustered.csv"
        )
        df.to_csv(output_file, index=False)
        logger.info(f"Saved clustered families to {output_file}")
        
    def get_unique_structures(self, filters: Dict, method: str = "vdWFV") -> pd.DataFrame:  # noqa: D401
        """
        Select one representative per cluster using the vdWFV metric.

        Workflow
        --------
        1. Load clustered families CSV.
        2. Group by ['family_id', 'cluster_id'].
        3. Compute vdWFV for each refcode; select the minimum.
        4. Save unique representatives to CSV.

        Parameters
        ----------
        filters : Dict[str, Any]
            Placeholder for revalidation filters.
        method : str, default="vdWFV"
            Only 'vdWFV' is supported.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['family_id', 'refcode'].

        Raises
        ------
        FileNotFoundError
            If clustered CSV is missing.
        NotImplementedError
            If method is not 'vdWFV'.
        """
        
        if method != "vdWFV":
            raise NotImplementedError("Only vdWFV method is supported.")

        csv_clusters = self.data_directory / f"{self.data_prefix}_refcode_families_clustered.csv"
        if not csv_clusters.exists():
            raise FileNotFoundError(csv_clusters)
        df_clusters = pd.read_csv(csv_clusters)

        if "cluster_id" in df_clusters.columns:
            group_cols = ["family_id", "cluster_id"]
        else:  # fall‑back – treat each family as single cluster
            group_cols = ["family_id"]
            df_clusters["cluster_id"] = 1  # dummy column for grouping

        # build argument list for executor
        args: List[Tuple[str, List[str]]] = [
            (fam_id, grp["refcode"].tolist())
            for (fam_id, _), grp in df_clusters.groupby(group_cols)
        ]

        reps: List[Tuple[str, str]] = []
        max_workers = os.cpu_count() - 4 or 4
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            for fam_id, rep_rc in exe.map(_representative_for_cluster, args):
                reps.append((fam_id, rep_rc))

        df_unique = pd.DataFrame(reps, columns=["family_id", "refcode"]).drop_duplicates()
        self._save_unique_structures(df_unique)
        return df_unique

    def _save_unique_structures(self, df: pd.DataFrame) -> None:  # unchanged except docstring tweak
        """
        Save unique structure representatives to CSV.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ['family_id', 'refcode'].

        Raises
        ------
        OSError
            If file writing fails.
        """
        out = self.data_directory / f"{self.data_prefix}_refcode_families_unique.csv"
        df.to_csv(out, index=False)
        logger.info("Saved unique structures to %s", out)

def _process_single_family(args: Tuple[str, List[str], Dict]) -> Tuple[str, List[List[str]]]:
    """
    Validate and cluster a single refcode family by packing similarity.

    Parameters
    ----------
    args : Tuple[str, List[str], Dict[str, Any]]
        - family_id : str
        - structures : List[str]
        - filters : Dict of validation criteria

    Returns
    -------
    Tuple[str, List[List[str]]]
        family_id and list of clusters (each a list of refcodes).

    Raises
    ------
    Exception
        If any error occurs during processing.
    """
    family_id, structures, filters = args
    reader = io.EntryReader("CSD")
    validator = StructureValidator(filters)
    valid = []
    for identifier in structures:
        try:
            entry = reader.entry(identifier)
            if validator.validate(entry.crystal, entry.molecule).is_valid:
                valid.append((identifier, entry.crystal))
        except Exception:
            continue

    sim = PackingSimilarity()
    G = nx.Graph()
    G.add_nodes_from(refcode for refcode, _ in valid)
    for (ref1, cry1), (ref2, cry2) in combinations(valid, 2):
        try:
            result = sim.compare(cry1, cry2)
            if result and result.nmatched_molecules == sim.settings.packing_shell_size and result.rmsd < 1.0:
                G.add_edge(ref1, ref2)
        except RuntimeError:
            continue

    return family_id, [sorted(group) for group in nx.connected_components(G)]

def _vdwfv_for_refcode(refcode: str) -> float:
    """
    Compute the vdWFV metric (1 − packing coefficient) for a refcode.

    Parameters
    ----------
    refcode : str
        CSD refcode.

    Returns
    -------
    float
        vdWFV value.

    Raises
    ------
    Exception
        If entry reading or coefficient retrieval fails.
    """
    reader = io.EntryReader("CSD")
    cry = reader.entry(refcode).crystal
    return 1.0 - cry.packing_coefficient

def _representative_for_cluster(args: Tuple[str, List[str]]) -> Tuple[str, str]:
    """
    Select the refcode with minimal vdWFV in a cluster.

    Parameters
    ----------
    args : Tuple[str, List[str]]
        - family_id : str
        - cluster : List[str]

    Returns
    -------
    Tuple[str, str]
        family_id and representative refcode.

    Raises
    ------
    Exception
        If any lookup fails.
    """
    family_id, cluster = args
    vdw_vals = {rc: _vdwfv_for_refcode(rc) for rc in cluster}
    lowest = min(vdw_vals.values())
    rep = sorted([rc for rc, v in vdw_vals.items() if v == lowest])[0]
    return family_id, rep

