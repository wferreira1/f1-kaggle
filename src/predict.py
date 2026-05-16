from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from features import FeatureArtifacts, load_frame, transform_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_PATH = ROOT / "data" / "in" / "test.csv"
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "out" / "submission.csv"


def _load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def generate_submission(
    model_name: str = "ebm",
    test_path: Path = DEFAULT_TEST_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sample_size: int | None = None,
    sample_seed: int = 42,
) -> Path:
    artifacts: FeatureArtifacts = _load_pickle(models_dir / "feature_artifacts.pkl")
    model = _load_pickle(models_dir / f"{model_name}.pkl")

    raw_test = load_frame(test_path, sample_size=sample_size, sample_seed=sample_seed)
    encoded_test = transform_features(raw_test, artifacts)
    probabilities = model.predict_proba(encoded_test.to_pandas())[:, 1]

    # O pipeline ordena as linhas para gerar features temporais, então usamos os ids
    # ordenados do dataframe bruto transformado, embora `id` não entre no modelo.
    ordered_ids = raw_test.sort(["Year", "Race", "Driver", "LapNumber"])["id"].to_list()
    submission = pd.DataFrame({"id": ordered_ids, "PitNextLap": probabilities})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera uma submissão Kaggle a partir de um modelo salvo.")
    parser.add_argument("--model", default="ebm", choices=["ebm", "random_forest"])
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Seleciona aleatoriamente N linhas do teste. Útil para smoke tests locais.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed usada na amostragem aleatória quando --sample-size é informado.",
    )
    args = parser.parse_args()

    output_path = generate_submission(
        model_name=args.model,
        test_path=args.test_path,
        models_dir=args.models_dir,
        output_path=args.output_path,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    print(f"Submissão salva em: {output_path}")


if __name__ == "__main__":
    main()
