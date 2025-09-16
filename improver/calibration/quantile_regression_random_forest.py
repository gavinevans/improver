# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugins to perform quantile regression using random forests."""

from typing import Optional

import numpy as np
import pandas as pd

from improver import BasePlugin, PostProcessingPlugin
from improver.constants import DAYS_IN_YEAR, HOURS_IN_DAY

try:
    from quantile_forest import RandomForestQuantileRegressor
except ModuleNotFoundError:
    # Define empty class to avoid type hint errors.
    class RandomForestQuantileRegressor:
        pass


def quantile_forest_package_available():
    """Return True if quantile_forest package is available, False otherwise."""
    try:
        from quantile_forest import RandomForestQuantileRegressor  # noqa F401
    except ModuleNotFoundError:
        return False
    return True


def prep_feature(
    df: pd.DataFrame,
    variable_name: str,
    feature_name: str,
    transformation: Optional[str] = None,
    pre_transform_addition: np.float32 = 0,
) -> pd.DataFrame:
    """Prepare features that require computation from the input DataFrame. Options
    available are mean and standard deviation of the input feature, the
    day of year, sine of day of year, cosine of day of year, hour of day,
    sine of hour of day and cosine of hour of day. When computing the mean or standard
    deviation, these will be computed over either the percentile or realization column,
    depending upon which is available.

    Args:
        df: Input DataFrame.
        variable_name: Name of the variable to be used for the computation.
        feature_name: Feature to be computed. Options are "mean", "std", "day_of_year",
            "day_of_year_sin", "day_of_year_cos", "hour_of_day",
            "hour_of_day_sin" and "hour_of_day_cos".
    Returns:
        df: DataFrame with the computed feature added.
    """
    possible_features = [
        "mean",
        "std",
        "skewness",
        "kurtosis",
        "interquartile_range",
        "coefficient_of_variation",
        "range",
        "min",
        "max",
    ]
    if (
        feature_name in possible_features
        or feature_name.startswith("percentile_")
        or feature_name.startswith("members_below")
    ):
        representation_name = [
            n for n in ["percentile", "realization"] if n in df.columns
        ][0]
        groupby_cols = ["forecast_reference_time", "forecast_period", "wmo_id"]
        subset_cols = [*groupby_cols] + [
            representation_name,
            variable_name,
        ]
        # For a subset of the input DataFrame compute the mean or standard deviation
        # over the representation column, grouped by the groupby columns.
        if feature_name == "mean":
            subset_df = df[subset_cols].groupby(groupby_cols).mean()
        elif feature_name == "std":
            subset_df = df[subset_cols].groupby(groupby_cols).std()
        elif feature_name == "skewness":
            subset_df = df[subset_cols].groupby(groupby_cols).skew()
        elif feature_name == "kurtosis":
            subset_df = df[subset_cols].groupby(groupby_cols).apply(pd.DataFrame.kurt)
        elif feature_name == "interquartile_range":
            q75 = df[subset_cols].groupby(groupby_cols).quantile(0.75)
            q25 = df[subset_cols].groupby(groupby_cols).quantile(0.25)
            subset_df = q75 - q25
        elif feature_name == "coefficient_of_variation":
            mean = df[subset_cols].groupby(groupby_cols).mean()
            std = df[subset_cols].groupby(groupby_cols).std()
            subset_df = std / mean
        elif feature_name == "range":
            max_val = df[subset_cols].groupby(groupby_cols).max()
            min_val = df[subset_cols].groupby(groupby_cols).min()
            subset_df = max_val - min_val
        elif feature_name == "min":
            subset_df = df[subset_cols].groupby(groupby_cols).min()
        elif feature_name == "max":
            subset_df = df[subset_cols].groupby(groupby_cols).max()
        elif feature_name.startswith("members_below"):
            threshold = float(feature_name.split("_")[2])
            threshold = getattr(np, transformation)(
                np.array(threshold) + pre_transform_addition
            )
            subset_df = (
                df[subset_cols]
                .assign(below_threshold=lambda x: x[variable_name] < threshold)
                .groupby(groupby_cols)["below_threshold"]
                .sum()
            )
            subset_df.rename(variable_name, inplace=True)
        elif feature_name.startswith("percentile_"):
            perc = float(feature_name.split("_")[1])
            subset_df = df[subset_cols].groupby(groupby_cols).quantile(perc / 100.0)

        subset_df = subset_df.reset_index()
        # Rename the column to distinguish the computed feature from the original.
        subset_df.rename(
            columns={variable_name: f"{variable_name}_{feature_name}"}, inplace=True
        )
        # Merge the computed feature back into the original DataFrame.
        df = df.merge(
            subset_df[groupby_cols + [f"{variable_name}_{feature_name}"]],
            on=groupby_cols,
            how="left",
        )

    elif feature_name in ["day_of_year", "day_of_year_sin", "day_of_year_cos"]:
        # For a large DataFrame, the strftime("%j") computation can take a noticeable
        # amount of time, so this computation is done once for each unique time
        # and then merged back into the DataFrame.
        doy_df = pd.DataFrame({"time": df["time"].unique()})
        doy_df["day_of_year"] = np.array(doy_df["time"].dt.strftime("%j"), np.int32)

        if feature_name == "day_of_year_sin":
            doy_df[feature_name] = np.sin(
                2 * np.pi * doy_df["day_of_year"].values / (DAYS_IN_YEAR + 1)
            ).astype(np.float32)
        elif feature_name == "day_of_year_cos":
            doy_df[feature_name] = np.cos(
                2 * np.pi * doy_df["day_of_year"].values / (DAYS_IN_YEAR + 1)
            ).astype(np.float32)
        df = df.merge(doy_df[["time", feature_name]], on="time", how="left")
    elif feature_name in ["hour_of_day", "hour_of_day_sin", "hour_of_day_cos"]:
        # For hour_of_day, unlike day_of_year, the hour attribute doesn't require
        # computation, therefore there is no benefit to creating the separate DataFrame
        # and merging it back into the DataFrame.
        if df["time"].nunique() == 1:
            hour_of_day = np.int32(df["time"].iloc[0].hour)
        else:
            hour_of_day = np.array(df["time"].dt.hour, dtype=np.int32)
        if feature_name == "hour_of_day":
            feature_values = hour_of_day
        elif feature_name == "hour_of_day_sin":
            feature_values = np.sin(2 * np.pi * hour_of_day / HOURS_IN_DAY).astype(
                np.float32
            )
        elif feature_name == "hour_of_day_cos":
            feature_values = np.cos(2 * np.pi * hour_of_day / HOURS_IN_DAY).astype(
                np.float32
            )
        df[feature_name] = feature_values
    return df


def sanitise_forecast_dataframe(
    df: pd.DataFrame, feature_config: dict[str, list[str]]
) -> pd.DataFrame:
    """Sanitise the forecast DataFrame by removing columns that are no longer
    required. Following the computation of e.g. the mean or standard deviation,
    the original feature can be removed. The column over which the mean or
    standard deviation has been computed (e.g. the percentile or realization column)
    is also removed.

    Args:
        df: Input DataFrame, potentially including some computed features.
        feature_config: Feature configuration defining the features to be used for QRF.
    """
    representation_name = [n for n in ["percentile", "realization"] if n in df.columns][
        0
    ]
    collapsed_features = []
    possible_features = [
        "mean",
        "std",
        "skewness",
        "kurtosis",
        "interquartile_range",
        "coefficient_of_variation",
        "range",
        "min",
        "max"
    ]
    for key, values in feature_config.items():
        collapsed_features.extend(
            [
                key
                for v in values
                if v in possible_features
                or v.startswith("percentile_")
                or v.startswith("members_below")
            ]
        )
    collapsed_features = list(set(collapsed_features))
    # Subset the dataframe by the first value of the representation column
    # and drop the representation column and any features where the original variable
    # is no longer required. This reduces the size of the DataFrame e.g. if there are
    # 3 percentiles initially, the subsetted dataframe will be 1/3 of the size.
    df = df[df[representation_name] == df[representation_name].iloc[0]]
    df = df.drop(columns=[representation_name, *collapsed_features])
    return df


def get_required_column_names(
    df: pd.DataFrame, feature_config: dict[str, list[str]]
) -> list[str]:
    """Process the feature_config to return the expected column names that will be
    used as features with the QRF.

    Args:
        df: Input DataFrame.
        feature_config: Feature configuration defining the features to be used for QRF.
    Returns:
        List of expected column names that will be used as features with the QRF.
    Raises:
        ValueError: If a feature expected in the feature_config is not present in
        the DataFrame.
    """
    possible_features = [
        "mean",
        "std",
        "skewness",
        "kurtosis",
        "interquartile_range",
        "coefficient_of_variation",
        "range",
        "min",
        "max"
    ]
    feature_column_names = []
    for variable_name in feature_config.keys():
        for feature in feature_config[variable_name]:
            if (
                feature in possible_features
                or feature.startswith("percentile_")
                or feature.startswith("members_below")
            ):
                feature_column_names.append(f"{variable_name}_{feature}")
            elif feature in ["static"]:
                feature_column_names.append(variable_name)
            else:
                feature_column_names.append(feature)

    if len(list(set(feature_column_names) - set(df.columns))) > 0:
        msg = f"Feature '{feature}' is not supported."
        raise ValueError(msg)

    return feature_column_names


def _check_valid_transformation(transformation: str):
    """Check if the transformation is one of the supported types.
    Args:
        transformation: Transformation to be checked.
    Raises:
        ValueError: If the transformation is not one of the supported types.
    """
    if transformation not in ["log", "log10", "sqrt", "cbrt", None]:
        msg = (
            "Currently the only supported transformations are log, log10, sqrt "
            f"and cbrt. The transformation supplied was {transformation}."
        )
        raise ValueError(msg)


class TrainQuantileRegressionRandomForests(BasePlugin):
    """Plugin to train a model using quantile regression random forests."""

    def __init__(
        self,
        target_name: str,
        feature_config: dict[str, list[str]],
        n_estimators: int,
        max_depth: Optional[int] = None,
        max_samples: Optional[float] = None,
        random_state: Optional[int] = None,
        transformation: Optional[str] = None,
        pre_transform_addition: np.float32 = 0,
        oversampling_bins: Optional[list] = None,
        oversampling_weights: Optional[list] = None,
        sample_to_smallest: Optional[bool] = False,
        add_oversampling_noise: Optional[bool] = None,
        sampling_bins: Optional[list] = None,
        sampling_weights: Optional[list] = None,
        replicate_with_noise: Optional[bool] = False,
        n_replicates: Optional[int] = 1,
        imblearn_oversampling: Optional[str] = None,
        threshold_bands: Optional[list[float]] = None,
        threshold_band_decider: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialise the plugin.

        Args:
            target_name (str):
                Name of the target variable to be calibrated e.g. 'air_temperature'.
            feature_config (dict):
                Feature configuration defining the features to be used for quantile
                regression. The configuration is a dictionary of strings, where the
                keys are the names of the columns within the dataframe. Some
                features may be used as initially provided within the dataframe,
                whilst others may be computed from the data e.g. mean, std.
                If the key is the feature itself e.g. distance to water, then the value
                should state "static". In this case, the name of feature e.g.
                'distance_to_water' is expected to be a column name in the input
                dataframe. The config will have the structure:
                "DYNAMIC_VARIABLE_CF_NAME": ["FEATURE1", "FEATURE2"] e.g.
                {
                "air_temperature": ["mean", "std", "altitude"],
                "visibility_at_screen_level": ["mean", "std"]
                "distance_to_water": ["static"],
                }
            n_estimators (int):
                Number of trees in the forest.
            max_depth (int):
                Maximum depth of the tree.
            max_samples (float):
                If an int, then it is the number of samples to draw to train
                each tree. If a float, then it is the fraction of samples to draw
                to train each tree. If None, then each tree contains the same
                total number of samples as originally provided.
            random_state (int):
                Random seed for reproducibility.
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.
            kwargs:
                Additional keyword arguments for the quantile regression model.

        """

        self.target_name = target_name
        self.feature_config = feature_config
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.random_state = random_state
        self.transformation = transformation
        _check_valid_transformation(self.transformation)
        self.pre_transform_addition = pre_transform_addition
        self.oversampling_bins = oversampling_bins
        self.oversampling_weights = oversampling_weights
        self.sample_to_smallest = sample_to_smallest
        self.add_oversampling_noise = add_oversampling_noise
        self.sampling_bins = sampling_bins
        self.sampling_weights = sampling_weights
        self.replicate_with_noise = replicate_with_noise
        self.n_replicates = n_replicates
        self.imblearn_oversampling = imblearn_oversampling
        self.threshold_bands = threshold_bands
        self.threshold_band_decider = threshold_band_decider
        self.kwargs = kwargs
        self.expected_coordinate_order = ["forecast_reference_time", "forecast_period"]

    def fit_qrf(
        self,
        forecast_features: np.ndarray,
        target: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> RandomForestQuantileRegressor:
        """Fit the quantile regression random forest model.
        Args:
            forecast_features (numpy.ndarray):
                Array of forecast features.
            target (numpy.ndarray):
                Array of target values.
        Returns:
            qrf_model (RandomForestQuantileRegressor):
                Fitted quantile regression model.
        """
        qrf_model = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_samples=self.max_samples,
            random_state=self.random_state,
            oob_score=True,
            **self.kwargs,
        )
        qrf_model.fit(forecast_features, target, sample_weight=sample_weights)
        return qrf_model

    def process(
        self,
        forecast_df: pd.DataFrame,
        truth_df: pd.DataFrame,
    ) -> None:
        """Train a quantile regression random forests model.

        Args:
            forecast_df:
                DataFrame containing the forecast information and features.
            truth_df:
                Cube containing the truths. The truths should have the same validity
                times as the forecasts.

        References:
            Johnson. (2024). quantile-forest: A Python Package for Quantile
            Regression Forests. Journal of Open Source Software, 9(93), 5976.
            https://doi.org/10.21105/joss.05976.
            Meinshausen, N. (2006). Quantile regression forests.
            Journal of Machine Learning Research,
            7(35), 983–999. http://jmlr.org/papers/v7/meinshausen06a.html
            Taillardat, M., O. Mestre, M. Zamo, and P. Naveau, 2016: Calibrated
            Ensemble Forecasts Using Quantile Regression Forests and Ensemble Model
            Output Statistics. Mon. Wea. Rev., 144, 2375–2393,
            https://doi.org/10.1175/MWR-D-15-0260.1.
            Taillardat, M. and Mestre, O.: From research to applications – examples of
            operational ensemble post-processing in France using machine learning,
            Nonlin. Processes Geophys., 27, 329–347,
            https://doi.org/10.5194/npg-27-329-2020, 2020.

        """
        if self.transformation:
            forecast_df[self.target_name] = getattr(np, self.transformation)(
                forecast_df[self.target_name] + self.pre_transform_addition
            )
            truth_df["ob_value"] = getattr(np, self.transformation)(
                truth_df["ob_value"] + self.pre_transform_addition
            )

        for variable_name in self.feature_config.keys():
            if variable_name not in forecast_df.columns:
                msg = (
                    f"Feature '{variable_name}' is not present in the "
                    "forecast DataFrame."
                )
                raise ValueError(msg)
            for feature_name in self.feature_config[variable_name]:
                forecast_df = prep_feature(
                    forecast_df,
                    variable_name,
                    feature_name,
                    transformation=self.transformation,
                    pre_transform_addition=self.pre_transform_addition,
                )
        # forecast_df.to_parquet("/data/scratch/gavin.evans/temp7/forecast_df.parquet")

        forecast_df = sanitise_forecast_dataframe(forecast_df, self.feature_config)

        feature_column_names = get_required_column_names(
            forecast_df, self.feature_config
        )
        merge_columns = ["wmo_id", "time"]
        combined_df = forecast_df.merge(
            truth_df[merge_columns + ["ob_value"]], on=merge_columns, how="inner"
        )

        if self.threshold_bands:
            qrf_models = []
            for index, band in enumerate(self.threshold_bands[:-1]):
                print("bands = ", band, self.threshold_bands[index + 1])
                if self.transformation is not None:
                    self.threshold_bands[index] = getattr(np, self.transformation)(
                        np.array(band) + self.pre_transform_addition
                    )
                combined_banded_df = combined_df[
                    (
                        combined_df[self.threshold_band_decider]
                        >= self.threshold_bands[index]
                    )
                    & (
                        combined_df[self.threshold_band_decider]
                        <= self.threshold_bands[index + 1]
                    )
                ]
                feature_values = np.array(combined_banded_df[feature_column_names])
                target_values = combined_banded_df["ob_value"].values
                qrf_models.append(
                    self.fit_qrf(feature_values, target_values, sample_weights=None)
                )
            return qrf_models

        if self.oversampling_bins is not None:
            bin_edges = getattr(np, self.transformation)(
                np.array(self.oversampling_bins) + self.pre_transform_addition
            )

            combined_df["ob_value_group"] = pd.cut(
                combined_df["ob_value"],
                bins=bin_edges,
                labels=False,
                include_lowest=True,
            )
            n_bins = len(bin_edges) - 1
            rows = len(combined_df.index)
            n_samples_per_bin = int(np.ceil(rows / n_bins))

            def sampler(df, n_samples_per_bin, weights, random_state):
                return df.sample(
                    n_samples_per_bin,
                    replace=True,
                    weights=weights,
                    random_state=random_state,
                )

            if self.oversampling_weights is None:
                weights = None
            else:
                combined_df["ob_value_group_weight"] = combined_df["ob_value_group"]
                combined_df["ob_value_group_weight"] = combined_df[
                    "ob_value_group_weight"
                ].astype(np.float32)
                for index, ob_value_group in enumerate(
                    sorted(combined_df["ob_value_group"].unique())
                ):
                    combined_df.loc[
                        combined_df["ob_value_group"] == ob_value_group,
                        "ob_value_group_weight",
                    ] = self.oversampling_weights[index]
                weights = combined_df["ob_value_group_weight"]

            combined_df = (
                combined_df.groupby("ob_value_group", group_keys=False)
                .apply(
                    sampler,
                    n_samples_per_bin=n_samples_per_bin,
                    random_state=self.random_state,
                    weights=weights,
                    include_groups=False,
                )
                .reset_index(drop=True)
            )

            if self.add_oversampling_noise is not None:
                combined_df["duplicates"] = combined_df.duplicated()
                noise = np.random.default_rng(self.random_state).normal(
                    loc=0,
                    scale=combined_df["ob_value"] * 0.01,
                    size=len(combined_df.index),
                )
                combined_df.loc[combined_df["duplicates"], "ob_value"] += noise[
                    combined_df["duplicates"]
                ]
                bound_min = getattr(np, self.transformation)(
                    self.pre_transform_addition
                )
                combined_df["ob_value"] = combined_df["ob_value"].clip(lower=bound_min)
                combined_df = combined_df.drop(columns=["duplicates"])

        if self.imblearn_oversampling in ["smote", "adasyn", "kmeans_smote"]:
            from imblearn.over_sampling import ADASYN, SMOTE, KMeansSMOTE

            bin_edges = getattr(np, self.transformation)(
                np.array(self.oversampling_bins) + self.pre_transform_addition
            )
            combined_df["ob_value_group"] = pd.cut(
                combined_df["ob_value"],
                bins=bin_edges,
                labels=False,
                include_lowest=True,
            )
            if self.imblearn_oversampling == "adasyn":
                sampler = ADASYN(random_state=self.random_state)
            elif self.imblearn_oversampling == "smote":
                sampler = SMOTE(random_state=self.random_state)
            elif self.imblearn_oversampling == "kmeans_smote":
                sampler = KMeansSMOTE(random_state=self.random_state)
            X_res, y_res = sampler.fit_resample(
                combined_df[feature_column_names], combined_df["ob_value_group"]
            )
            y_res = combined_df.loc[y_res.index, "ob_value"]
            combined_df = pd.concat([X_res, y_res], axis=1)

        if self.sampling_weights is None:
            sample_weights = None
        else:
            if self.sampling_bins:
                bin_edges = getattr(np, self.transformation)(
                    np.array(self.sampling_bins) + self.pre_transform_addition
                )

                combined_df["ob_value_group"] = pd.cut(
                    combined_df["ob_value"],
                    bins=bin_edges,
                    labels=False,
                    include_lowest=True,
                )
                combined_df["ob_value_group_weight"] = combined_df["ob_value_group"]
                combined_df["ob_value_group_weight"] = combined_df[
                    "ob_value_group_weight"
                ].astype(np.float32)
                for index, ob_value_group in enumerate(
                    sorted(combined_df["ob_value_group"].unique())
                ):
                    combined_df.loc[
                        combined_df["ob_value_group"] == ob_value_group,
                        "ob_value_group_weight",
                    ] = self.sampling_weights[index]
                sample_weights = combined_df["ob_value_group_weight"]
            else:
                if self.sampling_weights == "low_values_get_priority":
                    combined_df = combined_df.sort_values(
                        by="ob_value", ascending=False
                    )
                    sample_weights = np.arange(1, len(combined_df["ob_value"]) + 1)

        if self.replicate_with_noise:
            combined_df = pd.concat(
                [combined_df] * self.n_replicates, ignore_index=True
            )
            combined_df["duplicates"] = combined_df.duplicated()
            noise = np.random.default_rng(self.random_state).normal(
                loc=0,
                scale=combined_df["ob_value"] * 0.01,
                size=len(combined_df.index),
            )
            combined_df.loc[combined_df["duplicates"], "ob_value"] += noise[
                combined_df["duplicates"]
            ]
            bound_min = getattr(np, self.transformation)(self.pre_transform_addition)
            bound_max = getattr(np, self.transformation)(combined_df["ob_value"].max())
            combined_df["ob_value"] = combined_df["ob_value"].clip(
                lower=bound_min, upper=bound_max
            )
            combined_df = combined_df.drop(columns=["duplicates"])

        combined_df.to_parquet(
            "/data/scratch/gavin.evans/temp7/combined_df_trial_127_20251230T0000Z.parquet"
        )
        import pdb

        pdb.set_trace()
        feature_values = np.array(combined_df[feature_column_names])
        target_values = combined_df["ob_value"].values

        # Fit the quantile regression model
        return self.fit_qrf(
            feature_values, target_values, sample_weights=sample_weights
        )


class ApplyQuantileRegressionRandomForests(PostProcessingPlugin):
    """Plugin to apply a trained model using quantile regression random forests."""

    def __init__(
        self,
        target_name: str,
        feature_config: dict[str, list[str]],
        quantiles: list[np.float32],
        transformation: str = None,
        pre_transform_addition: np.float32 = 0,
        threshold_bands: Optional[list[float]] = None,
        threshold_band_decider: Optional[str] = None,
    ) -> None:
        """Initialise the plugin.

        Args:
            target_name (str):
                Name of the target variable to be calibrated.
            feature_config (dict):
                Feature configuration defining the features to be used for quantile
                regression. The configuration is a dictionary of strings, where the
                keys are the names of the columns within the dataframe. Some
                features may be used as initially provided within the dataframe,
                whilst others may be computed from the data e.g. mean, std.
                If the key is the feature itself e.g. distance to water, then the value
                should state "static". In this case, the name of feature e.g.
                'distance_to_water' is expected to be a column name in the input
                dataframe. The config will have the structure:
                "DYNAMIC_VARIABLE_CF_NAME": ["FEATURE1", "FEATURE2"] e.g.
                {
                "air_temperature": ["mean", "std", "altitude"],
                "visibility_at_screen_level": ["mean", "std"]
                "distance_to_water": ["static"],
                }
            quantiles (float):
                Quantiles used for prediction (values ranging from 0 to 1).
            transformation (str):
                Transformation to be applied to the data before fitting.
            pre_transform_addition (float):
                Value to be added before transformation.

        Raises:
            ValueError: If the transformation is not one of the supported types.

        """
        self.target_name = target_name
        self.feature_config = feature_config
        self.quantiles = quantiles
        self.transformation = transformation
        _check_valid_transformation(self.transformation)
        self.pre_transform_addition = pre_transform_addition
        self.threshold_bands = threshold_bands
        self.threshold_band_decider = threshold_band_decider

    def _reverse_transformation(self, forecast: np.ndarray) -> np.ndarray:
        """Reverse the transformation applied to the data prior to fitting the QRF.

        Args:
            forecast: Calibrated forecast.
        Returns:
            forecast: Forecast with the transformation reversed.
        """
        if self.transformation:
            if self.transformation == "log":
                forecast = np.exp(forecast) - self.pre_transform_addition
            elif self.transformation == "log10":
                forecast = 10 ** (forecast) - self.pre_transform_addition
            elif self.transformation == "sqrt":
                forecast = forecast**2 - self.pre_transform_addition
            elif self.transformation == "cbrt":
                forecast = forecast**3 - self.pre_transform_addition
        return forecast

    def process(
        self,
        qrf_model: RandomForestQuantileRegressor,
        forecast_df: pd.DataFrame,
    ) -> np.ndarray:
        """Apply a quantile regression random forests model.

        Args:
            qrf_model: A trained QRF model.
            forecast_df: DataFrame containing the forecast information and features.

        Returns:
            Calibrated forecast as a numpy array.

        """
        feature_values = []

        for variable_name in self.feature_config.keys():
            # Transform the feature cube data if a transformation is specified.
            if (
                self.transformation
                and set(["mean", "std"]).intersection(
                    self.feature_config[variable_name]
                )
                and self.target_name in forecast_df.columns
            ):
                forecast_df[self.target_name] = getattr(np, self.transformation)(
                    forecast_df[self.target_name] + self.pre_transform_addition
                )

            for feature_name in self.feature_config[variable_name]:
                forecast_df = prep_feature(
                    forecast_df,
                    variable_name,
                    feature_name,
                    transformation=self.transformation,
                    pre_transform_addition=self.pre_transform_addition,
                )

        forecast_df = sanitise_forecast_dataframe(forecast_df, self.feature_config)
        feature_column_names = get_required_column_names(
            forecast_df, self.feature_config
        )

        if self.threshold_bands:
            if self.transformation is not None:
                calibrated_forecast1 = np.zeros(
                    (len(forecast_df.index), len(self.quantiles)), dtype=np.float32
                )
                for index, (band, qrf_model1) in enumerate(
                    zip(self.threshold_bands[:-1], qrf_model)
                ):
                    self.threshold_bands[index] = getattr(np, self.transformation)(
                        np.array(band) + self.pre_transform_addition
                    )
                    forecast_banded_df = forecast_df[
                        (
                            forecast_df[self.threshold_band_decider]
                            >= self.threshold_bands[index]
                        )
                        & (
                            forecast_df[self.threshold_band_decider]
                            <= self.threshold_bands[index + 1]
                        )
                    ]
                    if len(forecast_banded_df.index) == 0:
                        continue
                    feature_values = np.array(forecast_banded_df[feature_column_names])
                    calibrated_forecast = qrf_model1.predict(
                        feature_values, quantiles=self.quantiles
                    )
                    calibrated_forecast = np.float32(calibrated_forecast)

                    calibrated_forecast = self._reverse_transformation(
                        calibrated_forecast
                    )
                    calibrated_forecast1[forecast_banded_df.index] = calibrated_forecast
                return calibrated_forecast1

        feature_values = np.array(forecast_df[feature_column_names])

        calibrated_forecast = qrf_model.predict(
            feature_values, quantiles=self.quantiles
        )
        calibrated_forecast = np.float32(calibrated_forecast)

        calibrated_forecast = self._reverse_transformation(calibrated_forecast)
        return calibrated_forecast
