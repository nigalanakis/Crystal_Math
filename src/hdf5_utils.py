"""
Module: hdf5_utils.py

Utilities to initialize and update HDF5 files for storing batched CSD structure data.

Dependencies
------------
pathlib
h5py
"""
from pathlib import Path
import h5py

def initialize_hdf5_file(
        hdf5_path: Path,
        compression: str = "gzip",
        chunk_size: int = 64
        ) -> h5py.File:
    """
    Create or open an HDF5 file and ensure the top-level '/structures' group exists.

    Parameters
    ----------
    hdf5_path : Path
        Path to the HDF5 file. If it does not exist, it will be created.
    compression : str, default="gzip"
        Compression algorithm to use for future datasets (not applied at group level).
    chunk_size : int, default=64
        Suggested chunk size for any future dataset creation (placeholder).

    Returns
    -------
    h5py.File
        An open HDF5 file handle in mode "a" (append).

    Side Effects
    ------------
    - Creates the file at hdf5_path if it does not exist.
    - Adds a group named '/structures' if it is missing.

    Raises
    ------
    OSError
        If the file cannot be opened or created.
    """
    f = h5py.File(str(hdf5_path), "a")
    if "structures" not in f:
        f.create_group("structures")
    return f

def write_structure_group(
        h5: h5py.File,
        refcode: str,
        raw_json: str
        ) -> None:
    """
    Store a raw JSON string under '/structures/<refcode>/raw_json', replacing any existing.

    Parameters
    ----------
    h5 : h5py.File
        An open HDF5 file, already structured with '/structures' group.
    refcode : str
        CSD refcode to identify the subgroup.
    raw_json : str
        Fully serialized JSON blob containing raw fields for this structure.

    Side Effects
    ------------
    - Creates or retrieves the group '/structures/<refcode>'.
    - Deletes any existing 'raw_json' dataset under that group.
    - Creates a new 'raw_json' dataset of type variable-length UTF-8 string.

    Raises
    ------
    KeyError
        If '/structures' group does not exist in h5 (should not happen if initialize_hdf5_file was used).
    OSError
        On dataset creation failure.
    """
    grp = h5["structures"].require_group(refcode)
    # raw JSON
    if "raw_json" in grp:
        del grp["raw_json"]
    grp.create_dataset(
        "raw_json",
        data=raw_json,
        dtype=h5py.string_dtype(encoding="utf-8")
    )
