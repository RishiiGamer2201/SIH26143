"""E005-A Kaggle kernel: runs next to the mounted mentor dataset (no 108 GB download).

Outputs (/kaggle/working):
  input_tree.txt          top of the mounted input tree
  trujillo_sha256.csv     rel_path, size, sha256 for every file of the Trujillo-Acatitla subset
  m4d.zip                 the M4D subset ("Semantic Segmentation/dataset") as-is, for local audit
"""
import glob
import hashlib
import os
import shutil
from multiprocessing import Pool

import pandas as pd

hits = glob.glob("/kaggle/input/**/Semantic Segmentation", recursive=True)
assert hits, os.listdir("/kaggle/input")
SEG = hits[0]
OUT = "/kaggle/working"
with open(f"{OUT}/input_tree.txt", "w") as f:
    for root, dirs, files in os.walk(os.path.dirname(SEG)):
        depth = root.count(os.sep) - SEG.count(os.sep)
        if depth <= 3:
            f.write(f"{root}  dirs={len(dirs)} files={len(files)}\n")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return os.path.relpath(p, SEG), os.path.getsize(p), h.hexdigest()


tru = sorted(glob.glob(f"{SEG}/Sentinel-1_SAR_Oil_spill_image_dataset/**/*.tif", recursive=True))
print("trujillo files", len(tru), flush=True)
with Pool(4) as pool:
    rows = pool.map(sha, tru, chunksize=8)
pd.DataFrame(rows, columns=["rel_path", "size", "sha256"]).to_csv(f"{OUT}/trujillo_sha256.csv", index=False)

shutil.make_archive(f"{OUT}/m4d", "zip", root_dir=SEG, base_dir="dataset")
print("done", os.listdir(OUT), flush=True)
