from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_BUILD_DIR = ROOT / ".kaggle-build" / "dataset"
KERNEL_DIR = ROOT / "kaggle" / "kernel"
DATASET_ID = "wfoliveira/f1-kaggle-code"
DATASET_TITLE = "F1 Kaggle Project Code"


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def build_dataset_package() -> Path:
    shutil.rmtree(DATASET_BUILD_DIR, ignore_errors=True)
    DATASET_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    required_files = ["README.md", "pyproject.toml", "uv.lock"]
    for file_name in required_files:
        shutil.copy2(ROOT / file_name, DATASET_BUILD_DIR / file_name)

    for path in (ROOT / "src").iterdir():
        if path.is_file() and path.suffix in {".py", ".sh"}:
            shutil.copy2(path, DATASET_BUILD_DIR / path.name)

    metadata = f'''{{
  "title": "{DATASET_TITLE}",
  "id": "{DATASET_ID}",
  "licenses": [
    {{
      "name": "other"
    }}
  ]
}}
'''
    (DATASET_BUILD_DIR / "dataset-metadata.json").write_text(metadata)
    return DATASET_BUILD_DIR


def deploy_dataset(create: bool, message: str) -> None:
    build_dir = build_dataset_package()
    if create:
        _run(["kaggle", "datasets", "create", "-p", str(build_dir)])
    else:
        _run(["kaggle", "datasets", "version", "-p", str(build_dir), "-m", message])


def push_kernel() -> None:
    metadata_path = KERNEL_DIR / "kernel-metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Arquivo ausente: {metadata_path}. "
            "Crie-o antes de enviar o kernel."
        )
    _run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publica o código no Kaggle e opcionalmente dispara o kernel remoto."
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Cria o dataset pela primeira vez. Sem essa flag, publica uma nova versão.",
    )
    parser.add_argument(
        "--message",
        default="Atualiza pipeline",
        help="Mensagem usada ao publicar uma nova versão do dataset.",
    )
    parser.add_argument(
        "--push-kernel",
        action="store_true",
        help="Depois de publicar o dataset, envia e executa o kernel remoto.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Apenas monta o pacote localmente, sem chamar o Kaggle CLI.",
    )
    args = parser.parse_args()

    if args.build_only:
        build_dir = build_dataset_package()
        print(f"Pacote preparado em: {build_dir}")
        return

    deploy_dataset(create=args.create, message=args.message)
    if args.push_kernel:
        push_kernel()


if __name__ == "__main__":
    main()
