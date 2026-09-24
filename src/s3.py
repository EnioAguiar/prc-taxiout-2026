"""Acesso ao bucket da OpenSky: baixar dados e enviar submissões.

Uso:
    python src/s3.py ls                      # lista buckets e objetos visíveis
    python src/s3.py download                # baixa tudo para data/
    python src/s3.py submit submissions/x.parquet
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_BUCKET = "prc-2026-datasets"


def client() -> Minio:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("BUCKET_ACCESS_KEY")
    secret = os.environ.get("BUCKET_ACCESS_SECRET")
    if not key or not secret:
        sys.exit("Faltam BUCKET_ACCESS_KEY/BUCKET_ACCESS_SECRET no .env")
    endpoint = os.environ.get("S3_ENDPOINT") or "s3.opensky-network.org"
    return Minio(endpoint, access_key=key, secret_key=secret, secure=True)


def cmd_ls(s3: Minio) -> None:
    for bucket in s3.list_buckets():
        print(f"{bucket.name}/")
        for obj in s3.list_objects(bucket.name, recursive=True):
            print(f"  {obj.object_name}  {obj.size / 1e6:.1f} MB")


def cmd_download(s3: Minio) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for obj in s3.list_objects(DATA_BUCKET, recursive=True):
        dest = DATA_DIR / obj.object_name
        if dest.exists() and dest.stat().st_size == obj.size:
            print(f"ok      {obj.object_name}")
            continue
        print(f"baixando {obj.object_name} ({obj.size / 1e6:.1f} MB)")
        for attempt in range(1, 6):
            try:
                s3.fget_object(DATA_BUCKET, obj.object_name, str(dest))
                break
            except Exception as e:  # conexão da OSN cai com frequência
                print(f"  falhou ({e.__class__.__name__}), tentativa {attempt}/5")
        else:
            sys.exit(f"desisti de {obj.object_name}")


def cmd_submit(s3: Minio, path: str) -> None:
    team = os.environ.get("TEAM_NAME")
    if not team:
        sys.exit("Falta TEAM_NAME no .env")
    bucket = os.environ.get("SUBMISSION_BUCKET") or team
    name = Path(path).name
    if not name.startswith(f"{team}_v") or not name.endswith(".parquet"):
        sys.exit(f"Nome deve ser {team}_v<N>.parquet (recebido: {name})")
    s3.fput_object(bucket, name, path)
    print(f"enviado {name} -> {bucket}/")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"ls", "download", "submit"}:
        sys.exit(__doc__)
    s3 = client()
    match sys.argv[1]:
        case "ls":
            cmd_ls(s3)
        case "download":
            cmd_download(s3)
        case "submit":
            if len(sys.argv) != 3:
                sys.exit("uso: python src/s3.py submit <arquivo.parquet>")
            cmd_submit(s3, sys.argv[2])


if __name__ == "__main__":
    main()
