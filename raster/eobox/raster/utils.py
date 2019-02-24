import numpy as np


dtype_ranges = {
    'uint8': (0, 255),
    'uint16': (0, 65535),
    'int16': (-32768, 32767),
    'uint32': (0, 4294967295),
    'int32': (-2147483648, 2147483647),
    'float32': (-3.4028235e+38, 3.4028235e+38),
    'float64': (-1.7976931348623157e+308, 1.7976931348623157e+308)}

def dtype_checker_df(df, dtype, return_=None):
    """Check if there are NaN values of values outside of a given datatype range.

    Arguments:
        df {dataframe} -- A dataframe.
        dtype {str} -- The datatype to check for.
    Keyword Arguments:
        return_ {str} -- Returns a boolean dataframe with the values not in the range of the dtype ('all'),
            the row ('rowsums') or column ('colsums') sums of that dataframe or an exit code 1 (None, default)
            if any of the values is not in the range.

    Returns:
        [int or DataFrame or Series] -- If no value is out of the range exit code 0 is returned, else depends on return_.
    """
    dtype_range = dtype_ranges[dtype]
    df_out_of_range = (df < dtype_range[0]) | (df > dtype_range[1]) | (~np.isfinite(df))
    if df_out_of_range.any().any():
        if return_== "colsums":
            df_out_of_range = df_out_of_range.apply(sum, axis=0) # column
        elif return_== "rowsums":
            df_out_of_range = df_out_of_range.apply(sum, axis=1) # row
        elif return_== "all":
            df_out_of_range = df_out_of_range
        else:
            df_out_of_range = 1
    else:
        df_out_of_range = 0
    return df_out_of_range

def cleanup_df_values_for_given_dtype(df, dtype, lower_as=None, higher_as=None, nan_as=None):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    if dtype not in dtype_ranges.keys():
        raise ValueError("'dtype' must be one of {dtype_ranges.keys()}")

    dtype_range = dtype_ranges[dtype]
    lut = {"lower_as": lower_as,
           "higher_as": higher_as,
           "nan_as": nan_as,
           }
    for key, value in lut.items():
        # the default is to assign the maximum number for unsigned dtypes and the min value for signed dtypes
        # this is a common convention with raster data
        # the real values will be derived below
        if value is None:
            if dtype[0] == "u":
                value = "max_range"
            else:
                value = "min_range"

        # derive or validate the values to be filled in
        if isinstance(value, str):
            if value not in ["min_range", "max_range"]:
                raise ValueError(f"'*_as' parameters must be numeric or in case of a string 'min_range' or 'max'range'.")
            else:
                if value == "min_range":
                    lut[key] = dtype_range[0]
                else:
                    lut[key] = dtype_range[1]
        elif is_number(value):
            if (value < dtype_range[0]) | (value > dtype_range[1]):
                raise ValueError("Numeric value of the '*_as' parameters must be in the data range of the given dtype.")
            else:
                lut[key] = value
        else:
            raise ValueError(f"'*_as' parameters must be numeric or in case of a string 'min_range' or 'max'range'.")

    # print(f"Replacing mapping: {lut}")
    df = df.where(~(df < dtype_range[0]), lut["lower_as"])
    df = df.where(~(df > dtype_range[1]), lut["higher_as"])
    df = df.where(~(df.isna()), lut["nan_as"])
    df = df.astype(dtype)

    return df
