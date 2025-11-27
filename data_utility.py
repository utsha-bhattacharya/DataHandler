#Utility module for flexible data loading, cleaning, transformation, and validation.
# =========================
# Imports
# =========================

import os
import re
import sqlite3

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)


# =========================
# Missing values handling
# =========================

def handle_missing(
    df: pd.DataFrame,
    cols=None,
    method: str = "mean",          # mean, median, mode, constant, drop_rows, drop_cols,
                                   # knn, iterative, interpolate, simple_* (simple_mean, ...)
    constant_value=None,           # used when method="constant" or simple_constant
    max_missing_ratio: float = 0.5,  # used when method="drop_cols"
    n_neighbors: int = 5,          # used when method="knn"
    iterative_estimator="bayesian_ridge",  # placeholder, can swap estimator object later
    interpolate_method: str = "linear",    # used when method="interpolate"
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Unified function to handle missing values in multiple ways.
    """
    if cols is None:
        cols = df.columns.tolist()

    work_df = df if inplace else df.copy()

    # 1. Deletion-based
    if method == "drop_rows":
        work_df = work_df.dropna(subset=cols)
        return work_df

    if method == "drop_cols":
        missing_ratio = work_df[cols].isna().mean()
        cols_to_drop = missing_ratio[missing_ratio > max_missing_ratio].index
        work_df = work_df.drop(columns=list(cols_to_drop))
        return work_df

    # 2. Simple pandas-based imputation
    if method in {"mean", "median", "mode", "constant"}:
        for col in cols:
            if method == "mean":
                if pd.api.types.is_numeric_dtype(work_df[col]):
                    work_df[col] = work_df[col].fillna(work_df[col].mean())
            elif method == "median":
                if pd.api.types.is_numeric_dtype(work_df[col]):
                    work_df[col] = work_df[col].fillna(work_df[col].median())
            elif method == "mode":
                if work_df[col].mode().shape[0] > 0:
                    work_df[col] = work_df[col].fillna(work_df[col].mode().iloc[0])
            elif method == "constant":
                value = constant_value if constant_value is not None else 0
                work_df[col] = work_df[col].fillna(value)
        return work_df

    # 3. Sklearn SimpleImputer (for more control)
    if method.startswith("simple_"):
        # simple_mean, simple_median, simple_most_frequent, simple_constant
        strategy = method.split("simple_")[-1]
        if strategy == "constant" and constant_value is None:
            raise ValueError(
                "constant_value must be provided for simple_constant strategy."
            )
        imputer = SimpleImputer(
            strategy="most_frequent" if strategy == "mode" else strategy,
            fill_value=constant_value,
        )
        work_df[cols] = imputer.fit_transform(work_df[cols])
        return work_df

    # 4. KNN imputation (multivariate)
    if method == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
        work_df[cols] = imputer.fit_transform(work_df[cols])
        return work_df

    # 5. Iterative (MICE-style) imputation (multivariate)
    if method == "iterative":
        # NOTE: iterative_estimator is currently unused here, but kept for future extension.
        imputer = IterativeImputer()
        work_df[cols] = imputer.fit_transform(work_df[cols])
        return work_df

    # 6. Time-series interpolation
    if method == "interpolate":
        for col in cols:
            work_df[col] = work_df[col].interpolate(method=interpolate_method)
        return work_df

    raise ValueError(f"Unknown method: {method}")


# =========================
# Flexible data loader
# =========================

def load_data(
    path: str,
    file_type: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Flexible loader for many common data formats.

    Parameters
    ----------
    path : str
        File path.
    file_type : str | None
        Explicit type override (e.g. 'csv', 'excel', 'json', 'sql', 'sqlite',
        'parquet', 'feather', 'tsv', 'txt', 'dat', 'xml', 'yaml', 'yml',
        'spss', 'stata', 'sas', 'avro', 'orc').
        If None, inferred from extension.
    kwargs : dict
        Extra arguments passed to the underlying pandas / reader function.

    Returns
    -------
    pd.DataFrame
    """
    if file_type is None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
    else:
        ext = file_type.lower()

    # 1. Tabular / spreadsheet
    if ext in {"csv"}:
        return pd.read_csv(path, **kwargs)
    if ext in {"tsv"}:
        return pd.read_csv(path, sep="\t", **kwargs)
    if ext in {"txt", "dat"}:
        # Default to comma; caller can override sep in kwargs.
        return pd.read_csv(path, **kwargs)
    if ext in {"xls", "xlsx"}:
        return pd.read_excel(path, **kwargs)

    # 2. Database-style
    if ext in {"sql"}:
        # Expect a SELECT query string in kwargs["sql"] and optionally a connection.
        sql = kwargs.pop("sql", None)
        conn = kwargs.pop("conn", None)
        if sql is None:
            raise ValueError(
                "For file_type='sql', provide sql='<SELECT ...>' and conn=<connection>."
            )
        if conn is None:
            raise ValueError(
                "For file_type='sql', provide an open DB connection in conn=..."
            )
        return pd.read_sql(sql, conn, **kwargs)

    if ext in {"sqlite", "db", "sqlite3"}:
        # Read all rows from a table: table name must be provided.
        table = kwargs.pop("table", None)
        if table is None:
            raise ValueError("For sqlite/db, provide table='<table_name>' in kwargs.")
        conn = sqlite3.connect(path)
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn, **kwargs)
        finally:
            conn.close()
        return df

    if ext in {"mdb", "accdb"}:
        raise NotImplementedError(
            "Access databases (.mdb/.accdb) need external tools/ODBC; not handled here."
        )

    # 3. Binary table formats
    if ext in {"parquet"}:
        return pd.read_parquet(path, **kwargs)
    if ext in {"feather"}:
        return pd.read_feather(path, **kwargs)

    # 4. Interchange formats
    if ext in {"json"}:
        return pd.read_json(path, **kwargs)
    if ext in {"xml"}:
        return pd.read_xml(path, **kwargs)
    if ext in {"yaml", "yml"}:
        import yaml

        with open(path, "r", encoding=kwargs.pop("encoding", "utf-8")) as f:
            data = yaml.safe_load(f)
        # Try to normalize list-of-records into DataFrame.
        return pd.json_normalize(data)

    # 5. Statistical formats
    if ext in {"sav", "zsav"}:
        return pd.read_spss(path, **kwargs)
    if ext in {"dta"}:
        return pd.read_stata(path, **kwargs)
    if ext in {"sas7bdat"}:
        return pd.read_sas(path, format="sas7bdat", **kwargs)
    if ext in {"xpt"}:
        return pd.read_sas(path, format="xpt", **kwargs)

    # 6. Big data / analytics
    if ext in {"avro"}:
        from fastavro import reader

        records = []
        with open(path, "rb") as f:
            for rec in reader(f):
                records.append(rec)
        return pd.DataFrame.from_records(records)

    if ext in {"orc"}:
        import pyarrow.orc as orc

        with open(path, "rb") as f:
            data = orc.ORCFile(f).read()
        return data.to_pandas()

    raise ValueError(f"Unsupported or unknown file type: {ext}")


# =========================
# Duplicate handling
# =========================

def handle_duplicates(
    df: pd.DataFrame,
    subset=None,
    method: str = "exact",          # "exact", "partial", "fuzzy"
    fuzzy_threshold: int = 90,
    keep: str | bool = "first",     # same as pandas drop_duplicates
    strip_spaces: bool = True,
    lower_case: bool = True,
    trim_columns: bool = True,
) -> pd.DataFrame:
    """
    Universal duplicate-handling function.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    subset : list or None
        Columns to consider for duplicate detection. None → entire row.
    method : str
        "exact" → exact duplicate removal
        "partial" → handles case/space/typo cleaning before duplicate check
        "fuzzy" → near-duplicate detection using fuzzy matching
    fuzzy_threshold : int
        Similarity threshold for fuzzy matching (0–100).
    keep : {"first", "last", False}
        How to keep duplicates (like pandas drop_duplicates).
    strip_spaces : bool
        Remove leading/trailing spaces from string columns.
    lower_case : bool
        Convert string columns to lowercase.
    trim_columns : bool
        Remove extra spaces within the string.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with duplicates handled.
    """
    df_clean = df.copy()

    # Step 1: Standard cleaning for strings
    for col in df_clean.select_dtypes(include=["object"]).columns:
        if strip_spaces:
            df_clean[col] = df_clean[col].str.strip()
        if lower_case:
            df_clean[col] = df_clean[col].str.lower()
        if trim_columns:
            df_clean[col] = df_clean[col].str.replace(r"\s+", " ", regex=True)

    # Method 1: Exact Duplicate Removal
    if method == "exact":
        return df_clean.drop_duplicates(subset=subset, keep=keep)

    # Method 2: Partial Duplicate Handling
    if method == "partial":
        # Apply normalization then treat as exact duplicates after cleaning
        return df_clean.drop_duplicates(subset=subset, keep=keep)

    # Method 3: Fuzzy (Near) Duplicate Handling
    if method == "fuzzy":
        if subset is None:
            subset = df_clean.columns.tolist()

        to_drop = set()
        df_clean["_checked"] = False

        for i in range(len(df_clean)):
            if df_clean.iloc[i]["_checked"]:
                continue

            for j in range(i + 1, len(df_clean)):
                if df_clean.iloc[j]["_checked"]:
                    continue

                row1 = " ".join(df_clean.iloc[i][subset].astype(str))
                row2 = " ".join(df_clean.iloc[j][subset].astype(str))

                similarity = fuzz.ratio(row1, row2)

                if similarity >= fuzzy_threshold:
                    to_drop.add(j)
                    df_clean.at[j, "_checked"] = True

        df_clean = df_clean.drop(index=to_drop)
        return df_clean.drop(columns="_checked")

    # Fallback (no duplicate handling)
    return df_clean


# =========================
# Structural error fixing
# =========================

def fix_structural_errors(
    df: pd.DataFrame,
    col_name_map: dict | None = None,
    boolean_cols: list[str] | None = None,
    numeric_str_cols: list[str] | None = None,
    date_cols: list[str] | None = None,
    date_format: str | None = None,
    categorical_cols: list[str] | None = None,
    category_replacements: dict | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Universal structural error fixer:
    - Standardizes column names.
    - Unifies boolean columns (yes/no, y/n, 1/0 to True/False).
    - Cleans numeric strings (removes commas, symbols).
    - Parses and standardizes date columns.
    - Standardizes category values with optional typo correction mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col_name_map : dict, optional
        Mapping to rename columns or correct typos in column names.
    boolean_cols : list of str, optional
        Columns to unify as booleans.
    numeric_str_cols : list of str, optional
        Columns storing numbers as strings needing cleaning.
    date_cols : list of str, optional
        Columns with dates to parse and standardize.
    date_format : str, optional
        Date format string, or None for automatic inference.
    categorical_cols : list of str, optional
        Columns with categorical data to standardize.
    category_replacements : dict, optional
        Mapping {column_name: {old_value: new_value}} or global {old_value: new_value}.
    inplace : bool, default False
        Whether to modify df in place.

    Returns
    -------
    pd.DataFrame
        Modified dataframe (same as input if inplace=True).
    """
    if not inplace:
        df = df.copy()

    # 1. Standardize column names
    def clean_col_name(name: str) -> str:
        name = name.strip().lower()
        name = name.replace(" ", "_").replace("-", "_")
        return name

    df.columns = [clean_col_name(col) for col in df.columns]

    # Apply explicit column rename if mapping provided
    if col_name_map:
        df.rename(columns=col_name_map, inplace=True)

    # 2. Unify boolean columns
    bool_map = {
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "1": True,
        "0": False,
        1: True,
        0: False,
        True: True,
        False: False,
        "true": True,
        "false": False,
    }
    if boolean_cols:
        for col in boolean_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map(bool_map).astype("boolean")

    # 3. Clean numeric strings and convert to float
    def clean_num_str(s):
        if pd.isna(s):
            return np.nan
        # Remove commas, currency symbols, spaces
        s = re.sub(r"[,\\$\\£\\€\\s]", "", str(s))
        try:
            return float(s)
        except Exception:
            return np.nan

    if numeric_str_cols:
        for col in numeric_str_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_num_str)

    # 4. Parse dates with optional format
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce")

    # 5. Standardize categorical columns with replacement mapping
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                if category_replacements and col in category_replacements:
                    df[col] = df[col].replace(category_replacements[col])
                elif category_replacements and not any(
                    col in key for key in category_replacements
                ):
                    # apply global replacements
                    df[col] = df[col].replace(category_replacements)
                # lower case and strip whitespace
                df[col] = df[col].astype(str).str.strip().str.lower()

    return df


# =========================
# Outlier handling
# =========================

def handle_outliers(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "iqr",  # 'iqr', 'zscore', 'quantile_cap', 'clip',
                          # 'replace_nan', 'replace_median',
                          # 'log_transform', 'boxcox_transform'
    iqr_factor: float = 1.5,
    zscore_threshold: float = 3.0,
    quantile_lower: float = 0.01,
    quantile_upper: float = 0.99,
    clip_min: dict | None = None,  # {col: min_val}
    clip_max: dict | None = None,  # {col: max_val}
    replace_with: str = "nan",     # kept for compatibility; method decides behavior
    boxcox_lmbda: float | None = None,
    return_outlier_indices: bool = False,
    inplace: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Universal outlier handler function.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.
    cols : list[str]
        List of numeric columns to handle.
    method : str
        Method to apply: 'iqr', 'zscore', 'quantile_cap', 'clip',
        'replace_nan', 'replace_median', 'log_transform', 'boxcox_transform'.
    iqr_factor : float
        Multiplier for IQR outlier bounds.
    zscore_threshold : float
        Z-score cutoff for outlier detection.
    quantile_lower : float
        Lower quantile bound for capping.
    quantile_upper : float
        Upper quantile bound for capping.
    clip_min : dict | None
        Dictionary of {col: min_val} for clipping extremes.
    clip_max : dict | None
        Dictionary of {col: max_val} for clipping extremes.
    replace_with : str
        Kept for backward compatibility; actual replacement determined by method.
    boxcox_lmbda : float | None
        Lambda parameter for Box-Cox transform, None to estimate.
    return_outlier_indices : bool
        If True, return a dict mapping col -> outlier row indices.
    inplace : bool
        Modify DataFrame in place or return a copy.

    Returns
    -------
    DataFrame or (DataFrame, outlier_indices dict) if return_outlier_indices=True.
    """
    if not inplace:
        df = df.copy()

    outlier_indices: dict[str, list] = {}

    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]

        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            mask_outliers = (series < lower_bound) | (series > upper_bound)
            outlier_indices[col] = df.index[mask_outliers].tolist()
            df.loc[mask_outliers, col] = np.nan

        elif method == "zscore":
            z_scores = np.abs(stats.zscore(series.dropna()))
            idx_non_na = series.dropna().index
            mask_outliers = z_scores > zscore_threshold
            outliers_idx = idx_non_na[mask_outliers]
            outlier_indices[col] = outliers_idx.tolist()
            df.loc[outliers_idx, col] = np.nan

        elif method == "quantile_cap":
            lower_cap = series.quantile(quantile_lower)
            upper_cap = series.quantile(quantile_upper)
            mask_lower = series < lower_cap
            mask_upper = series > upper_cap
            df.loc[mask_lower, col] = lower_cap
            df.loc[mask_upper, col] = upper_cap
            outlier_indices[col] = df.index[mask_lower | mask_upper].tolist()

        elif method == "clip":
            low = clip_min.get(col, None) if clip_min else None
            high = clip_max.get(col, None) if clip_max else None
            if low is not None or high is not None:
                df[col] = df[col].clip(lower=low, upper=high)
                outlier_indices[col] = df.index[
                    (df[col] == low) | (df[col] == high)
                ].tolist()
            else:
                outlier_indices[col] = []

        elif method in {"replace_nan", "replace_median"}:
            if method == "replace_nan":
                replacement = np.nan
            else:
                replacement = series.median()
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            mask_outliers = (series < lower_bound) | (series > upper_bound)
            outlier_indices[col] = df.index[mask_outliers].tolist()
            df.loc[mask_outliers, col] = replacement

        elif method == "log_transform":
            df[col] = np.log1p(series.clip(lower=0))
            outlier_indices[col] = []

        elif method == "boxcox_transform":
            from scipy.stats import boxcox

            # Box-Cox requires strictly positive data, clip minimum to small positive if needed
            clipped_series = series.clip(lower=1e-6)
            if boxcox_lmbda is None:
                transformed, lam = boxcox(clipped_series.dropna())
                full_transformed = pd.Series(index=clipped_series.index, dtype=float)
                full_transformed.loc[clipped_series.dropna().index] = transformed
                df[col] = full_transformed
            else:
                transformed = boxcox(clipped_series.dropna(), lmbda=boxcox_lmbda)
                full_transformed = pd.Series(index=clipped_series.index, dtype=float)
                full_transformed.loc[clipped_series.dropna().index] = transformed
                df[col] = full_transformed
            outlier_indices[col] = []

        else:
            raise ValueError(f"Unknown method '{method}' for outlier handling")

    if return_outlier_indices:
        return df, outlier_indices
    return df


# =========================
# Unit standardization
# =========================

def standardize_units(
    df: pd.DataFrame,
    config: dict,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Universal flexible unit standardizer.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    config : dict
        Dictionary describing how to convert units. Example:

        config = {
            "weight": {
                "value_col": "weight_value",
                "unit_col": "weight_unit",
                "target_unit": "kg",
                "conversion_map": {
                    "kg": 1.0,
                    "g": 0.001,
                    "lb": 0.453592,
                    "lbs": 0.453592,
                },
            },
            "height": {
                "value_col": "height_value",
                "unit_col": "height_unit",
                "target_unit": "cm",
                "conversion_map": {
                    "cm": 1.0,
                    "m": 100.0,
                    "inch": 2.54,
                    "in": 2.54,
                    "ft": 30.48,
                },
            },
            "temperature": {
                "value_col": "temp_value",
                "unit_col": "temp_unit",
                "target_unit": "C",  # 'C', 'F', or 'K'
            },
        }

    inplace : bool
        If True, modify df in place.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized units.
    """
    if not inplace:
        df = df.copy()

    # Helper for generic multiplicative conversions (kg<->lb, cm<->m, etc.)
    def _apply_generic_conversion(value_series, unit_series, conversion_map):
        factors = unit_series.map(conversion_map).astype(float)
        return value_series.astype(float) * factors

    # Helper for temperature conversion
    def _convert_temperature_to_target(series, unit_series, target_unit):
        target_unit = str(target_unit).upper()

        def convert_one(val, unit):
            if pd.isna(val) or pd.isna(unit):
                return pd.NA
            u = str(unit).strip().upper()

            # Normalize from C/F/K to Celsius first
            if u in {"C", "CELSIUS"}:
                c = float(val)
            elif u in {"F", "FAHRENHEIT"}:
                c = (float(val) - 32.0) * 5.0 / 9.0
            elif u in {"K", "KELVIN"}:
                c = float(val) - 273.15
            else:
                return pd.NA  # unknown unit

            # Then convert Celsius to target
            if target_unit == "C":
                return c
            elif target_unit == "F":
                return c * 9.0 / 5.0 + 32.0
            elif target_unit == "K":
                return c + 273.15
            else:
                return pd.NA

        return [convert_one(v, u) for v, u in zip(series, unit_series)]

    for name, spec in config.items():
        value_col = spec.get("value_col")
        unit_col = spec.get("unit_col")
        target_unit = spec.get("target_unit")

        if value_col not in df.columns or unit_col not in df.columns:
            continue

        # Temperature (special formulas)
        if name.lower().startswith("temp") or str(target_unit).upper() in {
            "C",
            "F",
            "K",
        }:
            df[value_col] = _convert_temperature_to_target(
                df[value_col], df[unit_col], target_unit
            )
            df[unit_col] = target_unit
            continue

        # Generic multiplicative conversions (weight, length, speed, currency, etc.)
        conversion_map = spec.get("conversion_map")
        if conversion_map is None:
            # no conversion map means can't standardize
            continue

        df[value_col] = _apply_generic_conversion(
            df[value_col],
            df[unit_col].astype(str).str.strip().str.lower(),
            {k.lower(): v for k, v in conversion_map.items()},
        )
        df[unit_col] = target_unit

    return df


# =========================
# Feature scaling
# =========================

def normalize_scale_features(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "standard",          # "standard", "minmax", "robust"
    feature_range: tuple[float, float] = (0.0, 1.0),  # for minmax
    with_centering: bool = True,       # for robust
    with_scaling: bool = True,         # for robust
    return_scaler: bool = False,
    inplace: bool = False,
):
    """
    Universal feature scaling function.

    - method="standard": z-score standardization (mean 0, std 1)
    - method="minmax": min–max scaling to feature_range
    - method="robust": robust scaling using median and IQR
    """
    if not inplace:
        df = df.copy()

    # Choose scaler
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == "robust":
        scaler = RobustScaler(
            with_centering=with_centering,
            with_scaling=with_scaling,
        )
    else:
        raise ValueError("method must be 'standard', 'minmax', or 'robust'")

    # Fit and transform only selected columns
    df[cols] = scaler.fit_transform(df[cols])

    if return_scaler:
        return df, scaler
    return df


# =========================
# Categorical encoding
# =========================

def encode_categoricals(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "label",        # "label", "onehot", "ordinal"
    ordinal_orders: dict | None = None,  # {col: [ordered_categories...]}
    drop_first: bool = False,     # for onehot
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Universal categorical encoder.

    - method="label": integer codes per category, per column
    - method="onehot": one-hot expand using pandas.get_dummies
    - method="ordinal": ordered integer codes using sklearn OrdinalEncoder
    """
    if not inplace:
        df = df.copy()

    # Ensure columns exist
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df

    if method == "label":
        # Per-column label-style encoding
        for col in cols:
            df[col] = df[col].astype("category").cat.codes

    elif method == "onehot":
        # One-hot encoding via pandas
        df = pd.get_dummies(df, columns=cols, drop_first=drop_first)

    elif method == "ordinal":
        # Build category orders per column if provided, otherwise infer
        if ordinal_orders:
            categories = []
            for col in cols:
                if col in ordinal_orders:
                    categories.append(ordinal_orders[col])
                else:
                    # infer order from sorted unique values
                    categories.append(
                        sorted(df[col].dropna().unique().tolist())
                    )
        else:
            categories = "auto"

        enc = OrdinalEncoder(
            categories=categories if categories != "auto" else "auto"
        )
        df[cols] = enc.fit_transform(df[cols])

    else:
        raise ValueError("method must be 'label', 'onehot', or 'ordinal'")

    return df


# =========================
# Feature engineering
# =========================

def feature_engineering(
    df: pd.DataFrame,
    new_feature_exprs: dict | None = None,
    log_transform_cols: list[str] | None = None,
    log_offset: float = 1.0,
    binning_config: dict | None = None,
    poly_config: dict | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Universal feature engineering function.

    Parameters
    ----------
    df : DataFrame
        Input data.
    new_feature_exprs : dict, optional
        Mapping {new_col_name: callable_or_str}
        - If callable: f(df) -> Series
        - If str: a pandas eval expression, using existing column names.
    log_transform_cols : list[str], optional
        Columns to log-transform with log1p-style: log(x + log_offset).
    log_offset : float
        Added before log to avoid log(0).
    binning_config : dict, optional
        {
          "col_name": {
              "bins": list or int,
              "labels": list or None,
              "strategy": "cut" or "qcut"
          },
          ...
        }
    poly_config : dict, optional
        {
          "cols": list[str],
          "degree": int,
          "include_bias": bool,
          "interaction_only": bool,
          "prefix": str
        }
    inplace : bool
        If True, modify df in place.

    Returns
    -------
    DataFrame
        Data with engineered features.
    """
    if not inplace:
        df = df.copy()

    # 1. Create new features
    if new_feature_exprs:
        for new_col, expr in new_feature_exprs.items():
            if callable(expr):
                df[new_col] = expr(df)
            elif isinstance(expr, str):
                df[new_col] = df.eval(expr)
            else:
                raise ValueError(
                    "new_feature_exprs values must be callables or strings."
                )

    # 2. Log transformation
    if log_transform_cols:
        for col in log_transform_cols:
            if col in df.columns:
                df[col + "_log"] = np.log(df[col] + log_offset)

    # 3. Binning
    if binning_config:
        for col, cfg in binning_config.items():
            if col not in df.columns:
                continue
            bins = cfg.get("bins")
            labels = cfg.get("labels", None)
            strategy = cfg.get("strategy", "cut")  # "cut" or "qcut"

            if strategy == "cut":
                df[col + "_bin"] = pd.cut(
                    df[col], bins=bins, labels=labels
                )
            elif strategy == "qcut":
                df[col + "_bin"] = pd.qcut(
                    df[col], q=bins, labels=labels
                )
            else:
                raise ValueError("strategy must be 'cut' or 'qcut'")

    # 4. Polynomial features
    if poly_config:
        cols = poly_config.get("cols", [])
        degree = poly_config.get("degree", 2)
        include_bias = poly_config.get("include_bias", False)
        interaction_only = poly_config.get("interaction_only", False)
        prefix = poly_config.get("prefix", "poly")

        cols = [c for c in cols if c in df.columns]
        if cols:
            poly = PolynomialFeatures(
                degree=degree,
                include_bias=include_bias,
                interaction_only=interaction_only,
            )
            X_poly = poly.fit_transform(df[cols].values)
            poly_names = poly.get_feature_names_out(cols)

            poly_df = pd.DataFrame(
                X_poly,
                columns=[f"{prefix}_{n}" for n in poly_names],
                index=df.index,
            )

            # Avoid duplicating original columns if desired; here we add only those not present
            for c in poly_df.columns:
                if c not in df.columns:
                    df[c] = poly_df[c]

    return df


# =========================
# Validation / inspection
# =========================

def validate_cleaned_data(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    allowed_missing_cols: list[str] | None = None,
    range_rules: dict | None = None,
    show_summary: bool = True,
) -> dict:
    """
    Universal validation for cleaned data:
    - Re-check summary stats.
    - Verify missing values.
    - Check numeric ranges.

    Parameters
    ----------
    df : DataFrame
        Cleaned dataframe to validate.
    numeric_cols : list[str], optional
        Numeric columns to check; if None, inferred.
    allowed_missing_cols : list[str], optional
        Columns where missing values are allowed.
    range_rules : dict, optional
        Per-column allowed ranges, e.g.:
        {
          "age": {"min": 0, "max": 120},
          "salary": {"min": 0},
        }
    show_summary : bool
        If True, prints summary-style info.

    Returns
    -------
    dict
        {
            "missing_issues": {col: count, ...},
            "range_issues": {col: {...}, ...},
            "summary": DataFrame (describe of numeric columns)
        }
    """
    report: dict = {
        "missing_issues": {},
        "range_issues": {},
        "summary": None,
    }

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if allowed_missing_cols is None:
        allowed_missing_cols = []

    # 1. Summary statistics
    if show_summary:
        print("=== Shape ===")
        print(df.shape)
        print("\n=== Dtypes ===")
        print(df.dtypes)
        print("\n=== Missing values (count) ===")
        print(df.isna().sum())
        print("\n=== Missing values (%) ===")
        print((df.isna().mean() * 100).round(2))
        print("\n=== Describe (numeric) ===")
        print(df[numeric_cols].describe())

    report["summary"] = df[numeric_cols].describe()

    # 2. Verify no unexpected missing values
    missing_counts = df.isna().sum()
    for col, cnt in missing_counts.items():
        if cnt > 0 and col not in allowed_missing_cols:
            report["missing_issues"][col] = int(cnt)

    # 3. Check ranges
    if range_rules:
        for col, rules in range_rules.items():
            if col not in df.columns:
                continue
            col_series = df[col]
            col_issues = {}

            min_allowed = rules.get("min", None)
            max_allowed = rules.get("max", None)

            if min_allowed is not None:
                below_mask = col_series < min_allowed
                n_below = int(below_mask.sum())
                if n_below > 0:
                    col_issues["below_min"] = {
                        "min_allowed": min_allowed,
                        "count": n_below,
                        "examples": col_series[below_mask].head(5).tolist(),
                    }

            if max_allowed is not None:
                above_mask = col_series > max_allowed
                n_above = int(above_mask.sum())
                if n_above > 0:
                    col_issues["above_max"] = {
                        "max_allowed": max_allowed,
                        "count": n_above,
                        "examples": col_series[above_mask].head(5).tolist(),
                    }

            if col_issues:
                report["range_issues"][col] = col_issues

    return report
