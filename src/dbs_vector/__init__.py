from importlib.metadata import version

try:
    __version__ = version("dbs-vector")
except Exception:
    __version__ = "unknown"
