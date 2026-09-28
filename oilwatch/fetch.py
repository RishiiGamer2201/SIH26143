"""Open-data connectors (no login): Sentinel-1 GRD measurement COGs (Microsoft Planetary Computer, anonymous SAS token),
NOAA MarineCadastre AIS daily files, BOEM offshore platform structures.

  python -m oilwatch.fetch s1 S1A_IW_GRDH_1SDV_20230103T000142_20230103T000207_046612_059620 --dest <SAFE>/measurement
  python -m oilwatch.fetch ais 2023-01-01 2023-01-03 --dest data/ais/raw
  python -m oilwatch.fetch boem --dest data/infrastructure/boem
Downloads are resumable-by-skip (existing complete files are kept) and hashed into <dest>/fetch_manifest.json.
"""
import argparse
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .manifest import sha256

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-grd"
NOAA_AIS = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{y}/AIS_{y}_{m:02d}_{d:02d}.zip"
BOEM = {"PlatStrucRawData.zip": "https://www.data.boem.gov/Platform/Files/PlatStrucRawData.zip"}
UA = {"User-Agent": "oilwatch/0.1 (SIH26143 research)"}


def _json(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))


def _head(url):
    r = urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"}), timeout=60)
    cr = r.headers.get("Content-Range")  # "bytes 0-0/<total>" when ranges are supported
    r.close()
    return int(cr.split("/")[-1]) if cr and r.status == 206 else None


def download_parallel(url, dest, size, n=8, chunk=8 << 20):
    """Ranged download with n concurrent connections (per-connection throughput is often the bottleneck)."""
    from concurrent.futures import ThreadPoolExecutor
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        f.truncate(size)
    ranges = [(a, min(a + chunk, size) - 1) for a in range(0, size, chunk)]
    done, t0 = [0], time.time()

    def get(rg):
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={rg[0]}-{rg[1]}"})
                data = urllib.request.urlopen(req, timeout=120).read()
                assert len(data) == rg[1] - rg[0] + 1
                with open(tmp, "r+b") as f:
                    f.seek(rg[0])
                    f.write(data)
                done[0] += len(data)
                return
            except Exception as e:
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"range {rg} failed")

    with ThreadPoolExecutor(n) as ex:
        for i, _ in enumerate(ex.map(get, ranges)):
            if i % 8 == 0:
                print(f"  {dest.name}: {done[0] / 1e6:.0f}/{size / 1e6:.0f} MB "
                      f"({done[0] / 1e6 / max(time.time() - t0, 1e-3):.1f} MB/s)", flush=True)
    tmp.replace(dest)
    print(f"  done {dest.name} ({size / 1e6:.0f} MB, {time.time() - t0:.0f} s)", flush=True)
    return dest


def download(url, dest, retries=3):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        size = _head(url)
    except Exception:
        size = None
    if size and dest.exists() and dest.stat().st_size == size:
        print(f"  have {dest.name} ({size / 1e6:.0f} MB)", flush=True)
        return dest
    if size and size > 32 << 20:
        return download_parallel(url, dest, size)
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                size = int(r.headers.get("Content-Length") or 0)
                if dest.exists() and size and dest.stat().st_size == size:
                    print(f"  have {dest.name} ({size / 1e6:.0f} MB)", flush=True)
                    return dest
                tmp, got, t0, last = dest.with_suffix(dest.suffix + ".part"), 0, time.time(), 0
                with open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if time.time() - last > 15:
                            last = time.time()
                            print(f"  {dest.name}: {got / 1e6:.0f}/{size / 1e6:.0f} MB "
                                  f"({got / 1e6 / max(time.time() - t0, 1e-3):.1f} MB/s)", flush=True)
                tmp.replace(dest)
                print(f"  done {dest.name} ({got / 1e6:.0f} MB)", flush=True)
                return dest
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries} {dest.name}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def record(dest_dir, files, source):
    m = Path(dest_dir) / "fetch_manifest.json"
    old = json.loads(m.read_text()) if m.exists() else {}
    old.update({Path(f).name: {"sha256": sha256(f), "bytes": Path(f).stat().st_size, "source": source,
                               "fetched_utc": pd.Timestamp.utcnow().isoformat()} for f in files})
    m.write_text(json.dumps(old, indent=1))


def s1_item(scene_prefix):
    """STAC item for a GRD scene; Planetary Computer ids drop the 4-char product unique id suffix."""
    t = pd.Timestamp(scene_prefix.split("_")[4])
    q = f"{PC_STAC}/collections/sentinel-1-grd/items/{scene_prefix}"
    return _json(q), t


def fetch_s1(scene_prefix, dest, pols=("vv", "vh"), safe_names=None):
    item, _ = s1_item(scene_prefix)
    token = _json(PC_TOKEN)["token"]
    out = []
    for p in pols:
        href = item["assets"][p]["href"]
        name = (safe_names or {}).get(p) or Path(href.split("?")[0]).name
        out.append(download(f"{href}?{token}", Path(dest) / name))
    record(dest, out, "Microsoft Planetary Computer sentinel-1-grd (" + scene_prefix + ")")
    return out


def fetch_ais(start, end, dest, bbox=None):
    """NOAA daily zips -> one parquet per day (optionally clipped to bbox = W,S,E,N) -> data/ais/processed style."""
    out = []
    for day in pd.date_range(start, end, freq="D"):
        z = download(NOAA_AIS.format(y=day.year, m=day.month, d=day.day), Path(dest) / f"AIS_{day:%Y_%m_%d}.zip")
        pq = Path(dest) / f"AIS_{day:%Y_%m_%d}{'_bbox' if bbox else ''}.parquet"
        if not pq.exists():
            with zipfile.ZipFile(z) as zf:
                csv = [n for n in zf.namelist() if n.endswith(".csv")][0]
                parts = []
                for ch in pd.read_csv(zf.open(csv), chunksize=2_000_000, low_memory=False):
                    if bbox:
                        ch = ch[ch.LON.between(bbox[0], bbox[2]) & ch.LAT.between(bbox[1], bbox[3])]
                    parts.append(ch)
            df = pd.concat(parts, ignore_index=True)
            df["BaseDateTime"] = pd.to_datetime(df.BaseDateTime, utc=True)
            df.to_parquet(pq, index=False)
            print(f"  {pq.name}: {len(df):,} rows", flush=True)
        out.append(pq)
    record(dest, out, "NOAA MarineCadastre AIS (coast.noaa.gov/htdata/CMSP/AISDataHandler)")
    return out


def fetch_boem(dest):
    out = [download(u, Path(dest) / n) for n, u in BOEM.items()]
    for z in out:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(Path(dest) / z.stem)
    record(dest, out, "BOEM Platform Structures (data.boem.gov)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oilwatch.fetch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("s1"); a.add_argument("scene"); a.add_argument("--dest", required=True)
    a.add_argument("--pols", default="vv,vh"); a.add_argument("--safe-name-vv"); a.add_argument("--safe-name-vh")
    b = sub.add_parser("ais"); b.add_argument("start"); b.add_argument("end"); b.add_argument("--dest", required=True)
    b.add_argument("--bbox", default=None, help="W,S,E,N")
    c = sub.add_parser("boem"); c.add_argument("--dest", required=True)
    x = ap.parse_args(argv)
    if x.cmd == "s1":
        fetch_s1(x.scene, x.dest, x.pols.split(","), {"vv": x.safe_name_vv, "vh": x.safe_name_vh})
    elif x.cmd == "ais":
        fetch_ais(x.start, x.end, x.dest, tuple(map(float, x.bbox.split(","))) if x.bbox else None)
    else:
        fetch_boem(x.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
