"""Descarga UNSW-NB15 (carpeta "CSV Files") desde el SharePoint oficial de UNSW.

Fuente: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Solo usa la biblioteca estándar. Escribe data/raw/ y un manifiesto con SHA-256
para verificar la integridad de lo descargado.

Uso: python src/data/download.py [--out data/raw] [--only-splits]
     python src/data/download.py --only-splits --out data/original   # datos del taller
"""
import argparse
import hashlib
import http.cookiejar
import json
import sys
import urllib.parse
from datetime import datetime, timezone
import urllib.request
from pathlib import Path

SHARE_LINK = (
    "https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/"
    "EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS"
)
SITE = "https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au"
ROOT = "/personal/z5025758_ad_unsw_edu_au/Documents/UNSW-NB15 dataset/CSV Files"
SPLITS_DIR = "Training and Testing Sets"


def build_opener():
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    # Abrir el enlace compartido entrega la cookie de acceso anónimo (FedAuth).
    opener.open(SHARE_LINK, timeout=60).read()
    return opener


def list_files(opener, folder):
    url = (
        f"{SITE}/_api/web/GetFolderByServerRelativeUrl('{urllib.parse.quote(folder)}')"
        "/Files?$select=Name,Length,ServerRelativeUrl"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json;odata=nometadata"})
    with opener.open(req, timeout=60) as r:
        return json.load(r)["value"]


def download(opener, server_url, dest, expected_size):
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"  ya existe: {dest.name}")
        return
    url = f"{SITE}/_api/web/GetFileByServerRelativeUrl('{urllib.parse.quote(server_url)}')/$value"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with opener.open(url, timeout=300) as r, open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {dest.name}: {done / 1e6:.1f}/{expected_size / 1e6:.1f} MB", end="", flush=True)
    print()
    if tmp.stat().st_size != expected_size:
        raise RuntimeError(f"Tamaño inesperado en {dest.name}")
    tmp.rename(dest)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--only-splits", action="store_true", help="solo training/testing set oficiales")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    opener = build_opener()

    folders = [f"{ROOT}/{SPLITS_DIR}"] if args.only_splits else [ROOT, f"{ROOT}/{SPLITS_DIR}"]
    manifest = {}
    for folder in folders:
        for f in list_files(opener, folder):
            dest = out / f["Name"]
            download(opener, f["ServerRelativeUrl"], dest, int(f["Length"]))
            manifest[f["Name"]] = {"bytes": int(f["Length"]), "sha256": sha256(dest)}

    record = {
        "source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "share_link": SHARE_LINK,
        "folders": folders,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": manifest,
    }
    (out / "MANIFEST.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"{len(manifest)} archivos en {out}/")


if __name__ == "__main__":
    sys.exit(main())
