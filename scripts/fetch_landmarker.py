import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
target = Path("results/models/face_landmarker_v1.task")
target.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(URL, timeout=60) as response:
    modified = response.headers.get("Last-Modified")
    if not modified or parsedate_to_datetime(modified) >= datetime(2025, 6, 1, tzinfo=timezone.utc):
        raise RuntimeError(f"Historical model provenance not established: {modified}")
    data = response.read()
target.write_bytes(data)
provenance = {
    "url": URL,
    "last_modified": modified,
    "bytes": len(data),
    "sha256": hashlib.sha256(data).hexdigest(),
    "historical_cutoff_verified": True,
}
Path("results/landmarker_asset.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps(provenance, indent=2))
