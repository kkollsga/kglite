"""Lossless pandas construction for already converted native result cells."""


def dataframe(data, columns=None, dtypes=None, native_dtypes=None):
    """Construct a DataFrame from exact native result cells.

    Integer/None columns use nullable Int64. Mixed columns containing integers
    use object, preserving numeric types and values. Other columns keep pandas
    inference. Unsupported recorded table dtypes retain the safe inferred form.
    pandas remains optional and is imported only when a caller requests a frame.

    A column may arrive as a numpy array instead of a list: native hands one
    over when every cell shares an unboxed numeric layout, which is the dtype
    pandas would have inferred from the boxed list anyway. Treat column values
    as a sequence — never as a truth value — so both forms flow through here.
    """
    import pandas as pd

    if isinstance(data, dict):
        raw = data
    else:
        names = columns if columns is not None else dict.fromkeys(k for row in data for k in row)
        raw = {name: [row.get(name) for row in data] for name in names}
        if not raw:
            return pd.DataFrame(index=range(len(data)), columns=columns)
    prepared = {}
    for name, values in raw.items():
        inferred = (
            native_dtypes[name] if native_dtypes is not None and name in native_dtypes else _integer_dtype(values)
        )
        target = (dtypes or {}).get(name, inferred)
        if target is None and len(values):
            prepared[name] = values
            continue
        try:
            prepared[name] = _column(pd, values, target)
        except (TypeError, ValueError):
            if target == inferred:
                raise
            prepared[name] = values if inferred is None else _column(pd, values, inferred)
    # Ordered/duplicate labels retain the existing constructor's semantics.
    return pd.DataFrame(prepared, columns=columns)


def _column(pd, values, dtype):
    """Build one column at the recorded dtype."""
    if dtype == "Int64":
        return pd.array(values, dtype=dtype)
    aware = _aware_datetime_dtype(pd, dtype)
    if aware is None:
        return pd.Series(values, dtype=dtype)
    # Aware cells are stored as UTC-naive instants, so constructing straight at
    # the aware dtype would read the UTC wall clock as zone-local time and shift
    # every value by the zone offset.
    naive = pd.Series(values, dtype=f"datetime64[{aware.unit}]")
    return naive.dt.tz_localize("UTC").dt.tz_convert(aware.tz)


def _aware_datetime_dtype(pd, dtype):
    """Return a tz-aware target as a dtype object; every other target is None."""
    if isinstance(dtype, str):
        if not dtype.startswith("datetime64[") or "," not in dtype:
            return None
        # An unusable zone raises here and reaches the caller's dtype fallback.
        dtype = pd.api.types.pandas_dtype(dtype)
    return dtype if isinstance(dtype, pd.DatetimeTZDtype) else None


def _integer_dtype(values):
    from pandas.api.types import infer_dtype

    # Native homogeneous numbers need no per-cell Python policy scan. Missing
    # and mixed labels retain the exact Python-type checks below.
    if infer_dtype(values, skipna=False) in {"integer", "floating"}:
        return None
    if not any(type(value) is int for value in values):
        return None
    if all(value is None or type(value) is int for value in values):
        return "Int64" if any(value is None for value in values) else None
    return object
