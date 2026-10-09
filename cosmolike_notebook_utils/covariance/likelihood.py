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

    Raises:
        ValueError if the mask rows are not consecutive zero-based 0/1
        cuts, size is not in 1..file_size, a .npy file is not the packed
        upper triangle, a text file has other than 3, 4 or 10 columns, an
        index lies outside the mask layout, or a value is nonfinite.
    """
    # --- 1. THE DATASET FILE AND THE TWO FILES IT NAMES ---

    # Relative names inside the .dataset file refer to its own directory.
    dataset = Path(dataset).resolve()
    parameters = IniFile(settings=str(dataset))
    covariance_path = dataset.parent/parameters.string(name="cov_file")
    mask_path = dataset.parent/parameters.string(name="mask_file")

    # --- 2. THE SCALE-CUT MASK AND THE SELECTED LEADING SIZE ---

    # File indices identify physical entries. Checking them prevents a
    # shifted mask from silently applying a cut to the next measurement.
    # ndmin=2 keeps a one-line file as a [1,2] table, so the column slices
    # below work for any length.
    mask_table = np.loadtxt(fname=mask_path, ndmin=2)
    file_size = len(mask_table)
    if (mask_table.shape[1] != 2
            or not np.array_equal(mask_table[:, 0], np.arange(file_size))
            or not np.all(np.isin(mask_table[:, 1], [0, 1]))):
        raise ValueError("mask needs consecutive zero-based indices and 0/1 cuts")

    # The returned matrix covers only the leading size entries; file entries
    # beyond them are read and then dropped in section 4.
    if size is None:
        size = file_size
    if not isinstance(size, (int, np.integer)) or not 0 < size <= file_size:
        raise ValueError("size must select a nonempty leading part of the file")
    total = np.zeros(shape=(size, size))

    # --- 3. THE ENTRIES: A PACKED .npy TRIANGLE OR AN INDEXED TEXT TABLE ---

    # Both branches end with three flat arrays of equal length: row index
    # first, column index second, and the covariance value.
    if covariance_path.suffix == ".npy":
        values = np.load(file=covariance_path, allow_pickle=False)
        if values.shape != (file_size*(file_size+1)//2,):
            raise ValueError("binary covariance must hold the packed upper triangle")
        # np.triu_indices lists the (i,j) pairs with i <= j row by row, the
        # order of the packed triangle, so values[m] sits at (first[m], second[m]).
        first, second = np.triu_indices(n=file_size)
    else:
        # Inspect one data row before reading the large table. Loading only
        # the index and covariance columns avoids storing unused metadata.
        with covariance_path.open() as stream:
            ncolumn = 0
            for line in stream:
                # Text after "#" is a comment. The first line with content
                # fixes the column count for the whole file.
                content = line.split("#", 1)[0].strip()
                if content:
                    ncolumn = len(content.split())
                    break

        # Columns to read for each layout: the two indices, then the one or
        # two value columns whose sum is C. The ten-column CosmoCov layout
        # keeps its two covariance terms in zero-based columns 8 and 9.
        columns = {
            3: (0, 1, 2),
            4: (0, 1, 2, 3),
            10: (0, 1, 8, 9),
        }
        if ncolumn not in columns:
            raise ValueError("text covariance needs 3, 4 or 10 columns")

        # Indices are read as floats. They must be whole numbers inside the
        # mask layout before the integer cast, which would truncate 3.5 to 3.
        table = np.loadtxt(fname=covariance_path,
                           usecols=columns[ncolumn], ndmin=2)
        indices = table[:, :2]
        if (not np.all(np.isfinite(indices))
                or np.any(indices != np.floor(indices))
                or np.any(indices < 0)
                or np.any(indices >= file_size)):
            raise ValueError("covariance indices must lie inside the mask layout")
        # np.intp is the integer type numpy uses for array indexing.
        first = indices[:, 0].astype(np.intp)
        second = indices[:, 1].astype(np.intp)
        # One value column passes through unchanged; two columns are summed.
        values = table[:, 2:].sum(axis=1)

    # --- 4. THE SYMMETRIC MATRIX OF THE LEADING ENTRIES ---

    if not np.all(np.isfinite(values)):
        raise ValueError("supplied covariance contains nonfinite values")
    # Keep the entries of the leading size x size block. A file may list
    # one triangle only, so each value fills both symmetric positions.
    selected = (first < size) & (second < size)
    first = first[selected]
    second = second[selected]
    values = values[selected]

    # Two index arrays write values[m] into total[first[m], second[m]] for
    # every m at once; the second line mirrors it. Positions absent from a
    # text file keep the zero they were given above.
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
            A cluster forecast with the Y transform also supplies
            valid_indices: every entry except the last angular bin of each
            cluster-lensing row, where Y is identically zero.
        supplied = read_likelihood_covariance result for that same layout.
        block_sizes, block_labels = contiguous physical probe groups before
            cuts. Groups with no retained entry are omitted from plot labels.
    Returns:
        Dict with cut components, supplied total, original indices and cut
        block sizes/labels. The original matrices are left unchanged.
    Raises:
        ValueError if the shapes disagree, the blocks do not partition the
        uncut vector, the mask retains nothing, or a cluster mask retains a
        known Y null row. These errors need a corrected layout or physical
        cut; adding a positive diagonal would hide the underlying mismatch.
    """
    # Both covariances must describe the same uncut vector, and the probe
    # blocks must tile it exactly so the plot labels land on their entries.
    size = len(supplied["mask"])
    if forecast["total"].shape != (size, size):
        raise ValueError("forecast and supplied covariance layouts disagree")
    if (sum(block_sizes) != size or len(block_sizes) != len(block_labels)
            or any(count <= 0 for count in block_sizes)):
        raise ValueError("probe blocks must partition the uncut vector")

    # np.flatnonzero returns the positions of the True mask entries: the
    # original indices of the retained measurements, in increasing order.
    indices = np.flatnonzero(supplied["mask"])
    if len(indices) == 0:
        raise ValueError("the supplied mask retains no measurements")

    # A cluster Y row is identically zero in its last angular bin, so its
    # forecast covariance row is zero too; retaining it would make the cut
    # forecast singular.
    if "valid_indices" in forecast:
        if not np.all(np.isin(indices, forecast["valid_indices"])):
            raise ValueError("the likelihood mask retains a defined Y null row")

    # A cut is a selection of measurements, so it acts on both covariance
    # axes. Applying it only to the diagonal would keep unwanted crosses.
    # np.ix_(indices, indices) makes matrix[selection] the submatrix of the
    # retained rows and columns, not a list of entries along one diagonal.
    selection = np.ix_(indices, indices)
    result = {"indices": indices, "supplied": supplied["total"][selection]}
    for name in ("total", "gaussian", "ssc", "cng"):
        if forecast[name].shape != (size, size):
            raise ValueError(f"{name} does not have the full measurement layout")
        result[name] = forecast[name][selection]

    # Walk the uncut vector block by block, with start at each block's first
    # entry, and count what the mask retains there. A block that keeps
    # nothing gets no size and no plot label.
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
