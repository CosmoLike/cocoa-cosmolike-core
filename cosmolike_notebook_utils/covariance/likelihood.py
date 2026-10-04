"""Read a supplied likelihood covariance for notebook comparisons.

The forecast and the supplied matrix remain separate. These helpers read
files only when requested, select the same data-vector entries on both
axes, and never install a matrix in the compiled likelihood or invert it.
"""

from pathlib import Path

from getdist.inifile import IniFile
import numpy as np


def read_likelihood_covariance(dataset, size=None):
    """Read the total covariance and scale-cut mask named by a dataset.

    Arguments:
        dataset = path to the project's .dataset file. DEFAULT inheritance
            follows the same GetDist reader used by the likelihood.
        size = optional leading submatrix length. For DESxPlanck, 1500
            selects galaxy/shear entries without inventing CMB predictions.
    Returns:
        Dict with total [size,size], mask [size] (True retains an entry),
        original file_size, and dataset/covariance/mask paths as strings.
        Text columns follow the C reader: (i,j,C), (i,j,G,NG), or the
        ten-column CosmoCov layout whose last two columns sum to C.
        A .npy file contains the packed upper triangle, including diagonal.
        The units and ordering remain those of the supplied data vector.

    Missing text entries are zero, as in the likelihood reader. No split
    into SSC and cNG is inferred from a combined non-Gaussian column.
    No Hartlap correction, inversion or eigenvalue repair is performed.
    """
    dataset = Path(dataset).resolve()
    parameters = IniFile(settings=str(dataset))
    covariance_path = dataset.parent/parameters.string(name="cov_file")
    mask_path = dataset.parent/parameters.string(name="mask_file")

    # File indices identify physical entries. Checking them prevents a
    # shifted mask from silently applying a cut to the next measurement.
    mask_table = np.loadtxt(fname=mask_path, ndmin=2)
    file_size = len(mask_table)
    if (mask_table.shape[1] != 2
            or not np.array_equal(mask_table[:, 0], np.arange(file_size))
            or not np.all(np.isin(mask_table[:, 1], [0, 1]))):
        raise ValueError("mask needs consecutive zero-based indices and 0/1 cuts")
    if size is None:
        size = file_size
    if not isinstance(size, (int, np.integer)) or not 0 < size <= file_size:
        raise ValueError("size must select a nonempty leading part of the file")
    total = np.zeros(shape=(size, size))

    if covariance_path.suffix == ".npy":
        values = np.load(file=covariance_path, allow_pickle=False)
        if values.shape != (file_size*(file_size+1)//2,):
            raise ValueError("binary covariance must hold the packed upper triangle")
        first, second = np.triu_indices(n=file_size)
    else:
        # Inspect one data row before reading the large table. Loading only
        # the index and covariance columns avoids storing unused metadata.
        with covariance_path.open() as stream:
            ncolumn = 0
            for line in stream:
                content = line.split("#", 1)[0].strip()
                if content:
                    ncolumn = len(content.split())
                    break
        columns = {
            3: (0, 1, 2),
            4: (0, 1, 2, 3),
            10: (0, 1, 8, 9),
        }
        if ncolumn not in columns:
            raise ValueError("text covariance needs 3, 4 or 10 columns")
        table = np.loadtxt(fname=covariance_path,
                           usecols=columns[ncolumn], ndmin=2)
        indices = table[:, :2]
        if (not np.all(np.isfinite(indices))
                or np.any(indices != np.floor(indices))
                or np.any(indices < 0)
                or np.any(indices >= file_size)):
            raise ValueError("covariance indices must lie inside the mask layout")
        first = indices[:, 0].astype(np.intp)
        second = indices[:, 1].astype(np.intp)
        values = table[:, 2:].sum(axis=1)

    if not np.all(np.isfinite(values)):
        raise ValueError("supplied covariance contains nonfinite values")
    selected = (first < size) & (second < size)
    first = first[selected]
    second = second[selected]
    values = values[selected]
    total[first, second] = values
    total[second, first] = values
    return {
        "total": total,
        "mask": mask_table[:size, 1].astype(bool),
        "file_size": file_size,
        "dataset": str(dataset),
        "covariance_file": str(covariance_path),
        "mask_file": str(mask_path),
    }


def select_likelihood_entries(forecast, supplied, block_sizes, block_labels):
    """Apply one supplied scale-cut mask to every computed component.

    Arguments:
        forecast = computed dict with total, gaussian, ssc and cng matrices
            already in the supplied file's ordering and measurement units.
        supplied = read_likelihood_covariance result for that same layout.
        block_sizes, block_labels = contiguous physical probe groups before
            cuts. Groups with no retained entry are omitted from plot labels.
    Returns:
        Dict with cut components, supplied total, original indices and cut
        block sizes/labels. The original matrices are left unchanged.
    Raises:
        ValueError if the shapes disagree or a cluster mask retains a known
        Y null row. These errors need a corrected layout or physical cut;
        adding a positive diagonal would hide the underlying mismatch.
    """
    size = len(supplied["mask"])
    if forecast["total"].shape != (size, size):
        raise ValueError("forecast and supplied covariance layouts disagree")
    if (sum(block_sizes) != size or len(block_sizes) != len(block_labels)
            or any(count <= 0 for count in block_sizes)):
        raise ValueError("probe blocks must partition the uncut vector")
    indices = np.flatnonzero(supplied["mask"])
    if len(indices) == 0:
        raise ValueError("the supplied mask retains no measurements")
    if "valid_indices" in forecast:
        if not np.all(np.isin(indices, forecast["valid_indices"])):
            raise ValueError("the likelihood mask retains a defined Y null row")

    # A cut is a selection of measurements, so it acts on both covariance
    # axes. Applying it only to the diagonal would keep unwanted crosses.
    selection = np.ix_(indices, indices)
    result = {"indices": indices, "supplied": supplied["total"][selection]}
    for name in ("total", "gaussian", "ssc", "cng"):
        if forecast[name].shape != (size, size):
            raise ValueError(f"{name} does not have the full measurement layout")
        result[name] = forecast[name][selection]

    result["block_sizes"] = []
    result["block_labels"] = []
    start = 0
    for count, label in zip(block_sizes, block_labels):
        retained = int(np.count_nonzero(supplied["mask"][start:start+count]))
        if retained > 0:
            result["block_sizes"].append(retained)
            result["block_labels"].append(label)
        start += count
    return result
