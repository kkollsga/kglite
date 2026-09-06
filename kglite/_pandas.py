"""Lossless pandas construction for already converted native result cells."""


def dataframe(data, columns=None, dtypes=None, native_dtypes=None):
    """Construct a DataFrame from exact native result cells.

    Integer/None columns use nullable Int64. Mixed columns containing integers
    use object, preserving numeric types and values. Other columns keep pandas
    inference. Unsupported recorded table dtypes retain the safe inferred form.
    pandas remains optional and is imported only when a caller requests a frame.
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
        if target is None and values:
            prepared[name] = values
            continue
        try:
            prepared[name] = pd.array(values, dtype=target) if target == "Int64" else pd.Series(values, dtype=target)
        except (TypeError, ValueError):
            if target == inferred:
                raise
            prepared[name] = (
                values
                if inferred is None
                else (pd.array(values, dtype=inferred) if inferred == "Int64" else pd.Series(values, dtype=inferred))
            )
    # Ordered/duplicate labels retain the existing constructor's semantics.
    return pd.DataFrame(prepared, columns=columns)


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
