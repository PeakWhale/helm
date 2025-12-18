from pathlib import Path
import shutil
import subprocess
import sys


def rm_file(path: Path):
    if path.exists() and path.is_file():
        path.unlink()


def rm_dir(path: Path):
    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def main():
    root = Path(__file__).resolve().parents[1]

    ledger = root / "data" / "ledger.duckdb"
    vault = root / "data" / "vault"
    chainlit_cache = root / ".chainlit"
    pycache_root = root / "__pycache__"
    pycache_src = root / "src" / "__pycache__"

    print("🌊 Peakwhale Helm demo reset starting")

    rm_file(ledger)

    if vault.exists():
        for pdf in vault.glob("*_loan_app.pdf"):
            rm_file(pdf)
    else:
        vault.mkdir(parents=True, exist_ok=True)

    rm_dir(chainlit_cache)
    rm_dir(pycache_root)
    rm_dir(pycache_src)

    print("✅ Cleaned generated artifacts")
    print("✅ Seeding ledger")
    subprocess.run([sys.executable, str(root / "src" / "seed_data.py")], check=True)

    print("✅ Generating loan PDFs")
    subprocess.run([sys.executable, str(root / "src" / "generate_docs.py")], check=True)

    print("✅ Demo reset complete")


if __name__ == "__main__":
    main()
