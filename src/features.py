from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import polars as pl
from category_encoders import OrdinalEncoder, TargetEncoder


TARGET_COL = "PitNextLap"
ID_COL = "id"
COMPOUND_MAPPING = [{"col": "Compound", "mapping": {"SOFT": 1, "MEDIUM": 2, "HARD": 3}}]


class FeatureArtifacts(TypedDict):
    ordinal_encoder: OrdinalEncoder
    target_encoder: TargetEncoder
    max_tyre_life_context: pl.DataFrame
    expected_stints_context: pl.DataFrame
    tyre_life_pitstop_context: pl.DataFrame
    feature_columns: list[str]


def load_frame(
    path: str | Path,
    sample_size: int | None = None,
    sample_seed: int = 42,
) -> pl.DataFrame:
    df = pl.read_csv(path)
    if sample_size is None:
        return df
    return df.sample(n=min(sample_size, df.height), seed=sample_seed, shuffle=True)


def prepare_base(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort(["Year", "Race", "Driver", "LapNumber"])
        .with_columns(
            [
                pl.col("LapNumber").diff().over(["Year", "Race", "Driver"]).alias("lap_gap"),
                pl.col("LapTime (s)").diff().over(["Year", "Race", "Driver"]).alias("lap_time_diff"),
            ]
        )
        .with_columns(
            [
                (pl.col("lap_time_diff") / pl.col("lap_gap")).alias("normalized_pace_decay"),
                (
                    pl.col("Cumulative_Degradation")
                    / pl.col("TyreLife").fill_null(1).replace(0, 1)
                ).alias("deg_intensity"),
            ]
        )
    )


def build_contexts(train_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    max_tyre_life_context = train_df.group_by(["Race", "Compound"]).agg(
        pl.col("TyreLife").max().alias("max_tyre_life_context")
    )
    expected_stints_context = train_df.group_by(["Race", "Year"]).agg(
        pl.col("Stint").quantile(0.9).alias("ExpectedStints_p90")
    )
    tyre_life_pitstop_context = (
        train_df.filter(pl.col(TARGET_COL) == 1)
        .group_by(["Race", "Year", "Compound"])
        .agg(pl.col("TyreLife").quantile(0.9).alias("TyreLife_PitStop_p90"))
    )
    return max_tyre_life_context, expected_stints_context, tyre_life_pitstop_context


def add_context_features(
    df: pl.DataFrame,
    max_tyre_life_context: pl.DataFrame,
    expected_stints_context: pl.DataFrame,
    tyre_life_pitstop_context: pl.DataFrame,
) -> pl.DataFrame:
    return (
        df.join(max_tyre_life_context, on=["Race", "Compound"], how="left")
        .with_columns(
            (pl.col("TyreLife") / pl.col("max_tyre_life_context")).alias("tyre_exhaustion_ratio")
        )
        .join(expected_stints_context, on=["Race", "Year"], how="left")
        .join(tyre_life_pitstop_context, on=["Race", "Year", "Compound"], how="left")
        .with_columns(
            (pl.col("TyreLife") / pl.col("TyreLife_PitStop_p90")).alias("tyreLife_pitstop_usage")
        )
    )


def optimize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    casts = {
        "LapNumber": pl.Int16,
        "Stint": pl.Int8,
        "Position": pl.Int8,
        "Year": pl.Int16,
        "TyreLife": pl.Int16,
        "LapTime (s)": pl.Float32,
        "LapTime_Delta": pl.Float32,
        "Cumulative_Degradation": pl.Float32,
        "RaceProgress": pl.Float32,
        "TyreLife_PitStop_p90": pl.Float32,
        "tyreLife_pitstop_usage": pl.Float32,
    }
    return df.with_columns([pl.col(col).cast(dtype) for col, dtype in casts.items() if col in df.columns])


def engineer_features(df: pl.DataFrame, artifacts: FeatureArtifacts) -> pl.DataFrame:
    return (
        df.pipe(prepare_base)
        .pipe(
            add_context_features,
            artifacts["max_tyre_life_context"],
            artifacts["expected_stints_context"],
            artifacts["tyre_life_pitstop_context"],
        )
        .fill_null(0)
        .fill_nan(0)
        .pipe(optimize_dtypes)
    )


def fit_feature_artifacts(train_df: pl.DataFrame) -> FeatureArtifacts:
    if TARGET_COL not in train_df.columns:
        raise ValueError(f"O dataframe de treino precisa conter a coluna {TARGET_COL!r}.")

    max_tyre_life_context, expected_stints_context, tyre_life_pitstop_context = build_contexts(train_df)
    provisional_artifacts: FeatureArtifacts = {
        "ordinal_encoder": None,  # type: ignore[typeddict-item]
        "target_encoder": None,  # type: ignore[typeddict-item]
        "max_tyre_life_context": max_tyre_life_context,
        "expected_stints_context": expected_stints_context,
        "tyre_life_pitstop_context": tyre_life_pitstop_context,
        "feature_columns": [],
    }

    engineered = engineer_features(train_df, provisional_artifacts)
    train_pd = engineered.to_pandas()
    feature_columns = [col for col in train_pd.columns if col not in {TARGET_COL, ID_COL}]

    ordinal_encoder = OrdinalEncoder(cols=["Compound"], mapping=COMPOUND_MAPPING)
    ordinal_encoded = ordinal_encoder.fit_transform(train_pd[feature_columns])

    target_encoder = TargetEncoder(cols=["Driver", "Race"], smoothing=10)
    target_encoder.fit(ordinal_encoded, train_pd[TARGET_COL])

    return {
        "ordinal_encoder": ordinal_encoder,
        "target_encoder": target_encoder,
        "max_tyre_life_context": max_tyre_life_context,
        "expected_stints_context": expected_stints_context,
        "tyre_life_pitstop_context": tyre_life_pitstop_context,
        "feature_columns": feature_columns,
    }


def transform_features(df: pl.DataFrame, artifacts: FeatureArtifacts) -> pl.DataFrame:
    engineered = engineer_features(df, artifacts)
    frame_pd = engineered.to_pandas()
    encoded = artifacts["ordinal_encoder"].transform(frame_pd[artifacts["feature_columns"]])
    encoded = artifacts["target_encoder"].transform(encoded)
    if TARGET_COL in frame_pd.columns:
        encoded[TARGET_COL] = frame_pd[TARGET_COL].to_numpy()
    return pl.from_pandas(encoded)


def fit_transform_features(train_df: pl.DataFrame) -> tuple[pl.DataFrame, FeatureArtifacts]:
    artifacts = fit_feature_artifacts(train_df)
    return transform_features(train_df, artifacts), artifacts
