from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features import TARGET_COL, fit_transform_features, load_frame, transform_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = ROOT / "data" / "in" / "train.csv"
DEFAULT_MODELS_DIR = ROOT / "models"


@dataclass
class TrainingResult:
    name: str
    auc: float
    elapsed_seconds: float
    model_path: Path


def _save_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def train_models(
    train_path: Path = DEFAULT_TRAIN_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    test_size: float = 0.33,
    random_state: int = 42,
    sample_size: int | None = None,
    sample_seed: int = 42,
) -> list[TrainingResult]:
    raw_train = load_frame(train_path, sample_size=sample_size, sample_seed=sample_seed)
    train_df, valid_df = train_test_split(
        raw_train.to_pandas(),
        test_size=test_size,
        random_state=random_state,
        stratify=raw_train[TARGET_COL].to_pandas(),
    )

    import polars as pl

    train_pl = pl.from_pandas(train_df)
    valid_pl = pl.from_pandas(valid_df)

    encoded_train, artifacts = fit_transform_features(train_pl)
    encoded_valid = transform_features(valid_pl, artifacts)

    X_train = encoded_train.drop(TARGET_COL).to_pandas()
    y_train = encoded_train[TARGET_COL].to_pandas()
    X_valid = encoded_valid.drop(TARGET_COL).to_pandas()
    y_valid = encoded_valid[TARGET_COL].to_pandas()

    smoke_test = sample_size is not None
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=20 if smoke_test else 100,
            n_jobs=-1,
            random_state=random_state,
        ),
        "ebm": ExplainableBoostingClassifier(
            outer_bags=4 if smoke_test else 32,
            interactions=5 if smoke_test else 15,
            n_jobs=1,
            random_state=random_state,
        ),
    }

    results: list[TrainingResult] = []
    for name, model in models.items():
        print(f"Iniciando modelo {name}")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
        model_path = models_dir / f"{name}.pkl"
        results.append(TrainingResult(name=name, auc=auc, elapsed_seconds=elapsed, model_path=model_path))
        print(f"{name}: ROC-AUC={auc:.4f} | tempo={elapsed:.2f}s")

    # Depois da validação, reajustamos o pipeline e os modelos com 100% do treino
    # para que os artefatos salvos usem toda a informação disponível.
    encoded_full_train, final_artifacts = fit_transform_features(raw_train)
    X_full_train = encoded_full_train.drop(TARGET_COL).to_pandas()
    y_full_train = encoded_full_train[TARGET_COL].to_pandas()

    for name, model in models.items():
        model.fit(X_full_train, y_full_train)
        _save_pickle(model, models_dir / f"{name}.pkl")

    _save_pickle(final_artifacts, models_dir / "feature_artifacts.pkl")

    rf = models["random_forest"]
    importances = pd.DataFrame(
        {"feature": X_full_train.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv(models_dir / "random_forest_feature_importances.csv", index=False)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelos para prever PitNextLap.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Seleciona aleatoriamente N linhas do treino. Útil para smoke tests locais.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed usada na amostragem aleatória quando --sample-size é informado.",
    )
    args = parser.parse_args()
    train_models(
        train_path=args.train_path,
        models_dir=args.models_dir,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )


if __name__ == "__main__":
    main()
