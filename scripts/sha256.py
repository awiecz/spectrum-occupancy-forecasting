"""Print the SHA-256 of a (large) file, streaming. Usage: python scripts/sha256.py FILE"""
import hashlib
import sys

h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
