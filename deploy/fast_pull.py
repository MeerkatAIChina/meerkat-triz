import json, os, time, hashlib, tarfile, shutil
import requests
from concurrent.futures import ThreadPoolExecutor

REGS = [
    ("https://docker.m.daocloud.io", "https://m.daocloud.io/auth/token", "docker.m.daocloud.io"),
    ("https://docker.1ms.run", None, None),
    ("https://hub.rat.dev", None, None),
]
REPO = "vllm/vllm-openai"
REF = "sha256:2f726db9babd627fb59addaed2038577fc767108680901271b069d91759d1286"
OUT = "/tmp/vllm-img"
RANGES = 8
STALL = 25  # seconds without data before reopening connection

_tokens = {}


def get_session(idx):
    reg, auth_url, service = REGS[idx % len(REGS)]
    if idx not in _tokens:
        try:
            probe = requests.get(f"{reg}/v2/", timeout=15)
            wa = probe.headers.get("www-authenticate", "")
            if wa.startswith("Bearer"):
                realm = wa.split('realm="')[1].split('"')[0]
                svc = wa.split('service="')[1].split('"')[0] if 'service="' in wa else ""
                tok = requests.get(realm, params={"service": svc,
                                   "scope": f"repository:{REPO}:pull"}, timeout=20).json()["token"]
                _tokens[idx] = (reg, {"Authorization": f"Bearer {tok}"})
            else:
                _tokens[idx] = (reg, {})
        except Exception as e:
            print(f"session {idx} auth fail: {e}", flush=True)
            _tokens[idx] = (reg, {})
    return _tokens[idx]


ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def get_manifest():
    reg, H = get_session(0)
    r = requests.get(f"{reg}/v2/{REPO}/manifests/{REF}", headers={**H, "Accept": ACCEPT}, timeout=60)
    r.raise_for_status()
    top = r.json()
    mt = top.get("mediaType", "")
    if "index" in mt or "manifest.list" in mt:
        m = next(x for x in top["manifests"] if x["platform"]["architecture"] == "arm64")
        r = requests.get(f"{reg}/v2/{REPO}/manifests/{m['digest']}",
                         headers={**H, "Accept": ACCEPT}, timeout=60)
        r.raise_for_status()
        return r.json()
    return top


def fetch_range_resumable(digest, start, end, path, mirror_idx):
    """Download bytes [start,end] of blob digest, reopening on stall."""
    got = os.path.getsize(path) if os.path.exists(path) else 0
    target = end - start + 1
    attempt = 0
    while got < target:
        attempt += 1
        reg, H = get_session(mirror_idx)
        url = f"{reg}/v2/{REPO}/blobs/{digest}"
        headers = {**H, "Range": f"bytes={start + got}-{end}"}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(20, STALL)) as resp:
                resp.raise_for_status()
                mode = "ab" if got else "wb"
                with open(path, mode) as f:
                    for chunk in resp.iter_content(1 << 19):
                        f.write(chunk)
                        got += len(chunk)
        except Exception as e:
            mirror_idx += 1  # rotate mirror on failure
            if attempt % 10 == 0:
                print(f"  range {start//(1<<20)}MB retry#{attempt} got {got}/{target}: {type(e).__name__}", flush=True)
            time.sleep(min(attempt, 10))
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_blob(desc, dest):
    size = desc["size"]
    dig = desc["digest"]
    if os.path.exists(dest) and os.path.getsize(dest) == size and sha256_file(dest) == dig.split(":")[1]:
        print(f"cached {dig[:24]} ({size/1e6:.0f}MB)", flush=True)
        return
    if size < 32 << 20:
        fetch_range_resumable(dig, 0, size - 1, dest, 0)
    else:
        bounds = [(i * size // RANGES, (i + 1) * size // RANGES - 1) for i in range(RANGES)]
        parts = [f"{dest}.part{i}" for i in range(RANGES)]
        with ThreadPoolExecutor(RANGES) as ex:
            futs = [ex.submit(fetch_range_resumable, dig, b[0], b[1], p, i % len(REGS))
                    for i, (p, b) in enumerate(zip(parts, bounds))]
            for f_ in futs:
                f_.result()
        with open(dest, "wb") as out:
            for p in parts:
                with open(p, "rb") as pf:
                    shutil.copyfileobj(pf, out)
                os.remove(p)
    assert sha256_file(dest) == dig.split(":")[1], f"sha mismatch {dig}"
    print(f"ok {dig[:24]} ({size/1e6:.0f}MB)", flush=True)


man = get_manifest()
cfg = man["config"]
layers = man["layers"]
total_gb = sum(l["size"] for l in layers) / 1e9
print(f"config {cfg['size']}B, {len(layers)} layers, total {total_gb:.1f} GB", flush=True)

os.makedirs(OUT, exist_ok=True)
fetch_blob(cfg, f"{OUT}/config.json")
names = []
for i, l in enumerate(layers):
    n = f"layer-{i:02d}.tar.gz"
    fetch_blob(l, f"{OUT}/{n}")
    names.append(n)

manifest_json = [{"Config": "config.json", "RepoTags": ["vllm/vllm-openai:v0.25.0"], "Layers": names}]
with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(manifest_json, f)
with tarfile.open("/tmp/vllm-image.tar", "w") as tf:
    tf.add(f"{OUT}/manifest.json", arcname="manifest.json")
    tf.add(f"{OUT}/config.json", arcname="config.json")
    for n in names:
        tf.add(f"{OUT}/{n}", arcname=n)
print("TAR DONE /tmp/vllm-image.tar", flush=True)
