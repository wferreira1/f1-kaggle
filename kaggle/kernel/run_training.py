from pathlib import Path
import sys


PROJECT_DATASET = Path("/kaggle/input/f1-kaggle-code")
COMPETITION_DATASET = Path("/kaggle/input/playground-series-s6e5")
WORKING_DIR = Path("/kaggle/working")

sys.path.append(str(PROJECT_DATASET / "src"))

from predict import generate_submission
from train import train_models


def main() -> None:
    train_models(
        train_path=COMPETITION_DATASET / "train.csv",
        models_dir=WORKING_DIR / "models",
    )
    generate_submission(
        model_name="ebm",
        test_path=COMPETITION_DATASET / "test.csv",
        models_dir=WORKING_DIR / "models",
        output_path=WORKING_DIR / "submission.csv",
    )


if __name__ == "__main__":
    main()
