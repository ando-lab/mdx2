import numpy as np
import pandas as pd


def _trim(series: "pd.Series", fraction: float = 0.25) -> "pd.Series[bool]":
    """
    Return a boolean mask (same index) that is True except at the indices
    of the N smallest and N largest values of `series`.
    """
    n = int(np.round(fraction * series.size))
    if n <= 0 or series.size < 20:  # if there are too few points, don't trim anything
        return pd.Series(True, index=series.index, dtype=bool)

    small_idx = series.nsmallest(n).index
    large_idx = series.nlargest(n).index

    mask = pd.Series(True, index=series.index, dtype=bool)
    mask.loc[small_idx] = False
    mask.loc[large_idx] = False
    return mask


def _calc_isoavg(df_in, bin_edges, trim_fraction=0.1):
    df = df_in.copy()
    df.dropna(subset=["intensity"], inplace=True)
    s_bins = pd.cut(df["s"], bins=bin_edges)

    df["mask"] = df.groupby(s_bins, observed=False)["intensity"].transform(lambda x: _trim(x, fraction=trim_fraction))
    df.dropna(subset=["mask"], inplace=True)

    df_isoavg = (
        df[df["mask"]]
        .groupby(s_bins, observed=False)
        .agg(
            {
                "s": ["mean", "count"],
                "intensity": ["mean", "std"],
                "intensity_error": "mean",
            }
        )
        .set_index(("s", "mean"))
    )
    df_isoavg["ioversigma"] = df_isoavg[("intensity", "mean")] / df_isoavg[("intensity_error", "mean")]

    # Treat non-finite I/sigma values as invalid so they are removed by later filters.
    df_isoavg.loc[~np.isfinite(df_isoavg["ioversigma"]), ("intensity", "mean")] = np.nan

    # first, drop any rows with NaN values in the index (i.e. ("s", "count") == 0)
    df_isoavg = df_isoavg[df_isoavg[("s", "count")] != 0]

    # apply some sane filters.
    # if count < 10, or intensity/intensity_error < 1, set intensity to NaN
    df_isoavg.loc[df_isoavg[("s", "count")] < 10, ("intensity", "mean")] = np.nan
    df_isoavg.loc[df_isoavg["ioversigma"] < 1, ("intensity", "mean")] = np.nan

    # add rows for s=0 and s=smax, with intensity = np.nan
    df_isoavg.loc[0] = np.nan
    df_isoavg.loc[bin_edges[-1]] = np.nan

    # sort by index
    df_isoavg = df_isoavg.sort_index()

    # fill the nans
    df_isoavg[("intensity", "mean")] = df_isoavg[("intensity", "mean")].interpolate(method="nearest").ffill().bfill()

    # get values to return as numpy arrays
    s_values = df_isoavg.index.values
    intensity_values = df_isoavg[("intensity", "mean")].values

    return s_values, intensity_values
