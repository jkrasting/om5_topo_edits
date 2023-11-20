#!/usr/bin/env python

import argparse
import netCDF4 as nc


def read_netcdf(fname):
    """
    Reads a NetCDF file and returns a list of tuples containing (jEdit, iEdit, zEdit).

    :param fname: Filename of the input NetCDF file.
    :return: List of tuples with (jEdit, iEdit, zEdit).
    """

    with nc.Dataset(fname, "r") as dataset:
        # Extract the variables
        iEdit = dataset.variables["iEdit"][:]
        jEdit = dataset.variables["jEdit"][:]
        zEdit = dataset.variables["zEdit"][:]

        # Combine the variables into a list of tuples
        edit_list = list(zip(jEdit, iEdit, zEdit))

    return edit_list


def write_netcdf(edit_list, ni, nj, fname):
    # for edit in edit_list:
    #    print(f"New Depth at jEdit={edit[0]} iEdit={edit[1]} set to {edit[2]}")
    with nc.Dataset(fname, "w", format="NETCDF4") as dataset:
        dataset.createDimension("nEdits", len(edit_list))
        ni_var = dataset.createVariable("ni", "i4")
        nj_var = dataset.createVariable("nj", "i4")
        ni_var.long_name = (
            "The size of the i-dimension of the dataset these edits apply to"
        )
        nj_var.long_name = (
            "The size of the j-dimension of the dataset these edits apply to"
        )
        ni_var.assignValue(ni)
        nj_var.assignValue(nj)
        iEdit = dataset.createVariable("iEdit", "i4", ("nEdits",))
        jEdit = dataset.createVariable("jEdit", "i4", ("nEdits",))
        zEdit = dataset.createVariable("zEdit", "f4", ("nEdits",))
        iEdit.long_name = "i-index of edited data"
        jEdit.long_name = "j-index of edited data"
        zEdit.long_name = "New value of data"
        zEdit.units = "meters"
        for i, (j, i_, z) in enumerate(edit_list):
            iEdit[i] = i_
            jEdit[i] = j
            zEdit[i] = z
    print(f"NetCDF file '{fname}' written successfully.")


def get_ni_nj(filename):
    with nc.Dataset(filename, "r") as dataset:
        ni = dataset.variables["ni"][:]
        nj = dataset.variables["nj"][:]
    return ni, nj


def merge_edits(files):
    merged_edits = {}
    ni, nj = None, None

    for file in files:
        current_ni, current_nj = get_ni_nj(file)

        if ni is None and nj is None:
            ni, nj = current_ni, current_nj
        elif ni != current_ni or nj != current_nj:
            raise ValueError(
                f"ni and nj values are not consistent across files. Check file: {file}"
            )

        for jEdit, iEdit, zEdit in read_netcdf(file):
            if (iEdit, jEdit) in merged_edits and merged_edits[(iEdit, jEdit)] != zEdit:
                raise ValueError(
                    f"Conflicting edits found at point (iEdit={iEdit}, jEdit={jEdit})."
                )
            merged_edits[(iEdit, jEdit)] = zEdit

    return [(j, i, z) for (i, j), z in merged_edits.items()], ni, nj


def main():
    parser = argparse.ArgumentParser(
        description="Merge edits from multiple NetCDF files."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output NetCDF file name."
    )
    parser.add_argument("files", nargs="+", help="Input NetCDF files to merge.")
    args = parser.parse_args()

    try:
        edit_list, ni, nj = merge_edits(args.files)
        write_netcdf(edit_list, ni, nj, args.output)
        print(f"Merged edits written to {args.output}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
