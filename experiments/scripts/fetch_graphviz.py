import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

VERSION = "12.2.1"
api = f"https://gitlab.com/api/v4/projects/4207231/releases/{VERSION}"
with urllib.request.urlopen(api, timeout=30) as response:
    release = json.load(response)
if datetime.fromisoformat(release["released_at"]) >= datetime(2025, 6, 1, tzinfo=timezone.utc):
    raise RuntimeError("Post-period Graphviz binary")
name = f"windows_10_cmake_Release_Graphviz-{VERSION}-win64.zip"
url = next(a["url"] for a in release["assets"]["links"] if a["name"] == name)
with urllib.request.urlopen(url + ".sha256", timeout=30) as response:
    expected = response.read().decode().split()[0]
with urllib.request.urlopen(url, timeout=60) as response:
    archive = response.read()
actual = hashlib.sha256(archive).hexdigest()
if actual != expected:
    raise RuntimeError("Graphviz archive checksum mismatch")
root = Path("results/tools/graphviz").resolve()
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
    for member in bundle.infolist():
        target = (root / member.filename).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError("Unsafe archive path")
    bundle.extractall(root)
provenance = {
    "version": VERSION,
    "released_at": release["released_at"],
    "source": api,
    "archive_url": url,
    "sha256": actual,
}
Path("results/graphviz_asset.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps(provenance, indent=2))
