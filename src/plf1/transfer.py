# transfer.py
import json
import zipfile
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import hashlib
import shutil
from copy import deepcopy

from .context import get_shared_data, set_shared_data
from ._pipeline import PipeLine

# ---------------------------
# Role enforcement
# ---------------------------
def _ensure_base(settings):
    if settings.get("lab_role") != "base":
        raise RuntimeError("Operation allowed only in BASE lab")

def _ensure_remote(settings):
    if settings.get("lab_role") != "remote":
        raise RuntimeError("Operation allowed only in REMOTE lab")

# ---------------------------
# TransferContext
# ---------------------------
class TransferContext:
    """Runtime context for remapping paths and components on remote."""
    def __init__(self, path_map=None, component_map=None, transfer_id=None):
        self.path_map = {Path(k).as_posix(): Path(v).as_posix() for k, v in (path_map or {}).items()}
        self.component_map = component_map or {}
        self.transfer_id = transfer_id

    def map_path(self, path: str) -> str:
        path = Path(path).as_posix()
        for src, dst in self.path_map.items():
            if path.startswith(src):
                return path.replace(src, dst, 1)
        return path

    def map_component(self, loc: str) -> str:
        return self.component_map.get(loc, loc)

# ---------------------------
# Safe ZIP extraction
# ---------------------------
def _safe_extract(zip_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            p = Path(member.filename)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe ZIP content detected")
        zf.extractall(target_dir)

# ---------------------------
# Extract paths and locs from config
# ---------------------------
def extract_paths_and_locs(config: dict):
    paths = set()
    locs = set()

    def recurse(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "loc":
                    locs.add(v)
                elif k in ("src", "path", "data_path"):
                    paths.add(v)
                else:
                    recurse(v)
        elif isinstance(d, list):
            for item in d:
                recurse(item)
    recurse(config)
    return paths, locs

# ---------------------------
# Export Transfer
# ---------------------------
def export_transfer(ppls, clone_id=None, transfer_type="run", prev_transfer_id=None, mode="copy"):
    """
    Export pipelines from current lab.

    BASE lab: full export
    REMOTE lab: results-only export
    """
    settings = get_shared_data()
    role = settings.get("lab_role")

    if role == "base":
        if not clone_id:
            raise ValueError("clone_id required for base export")
        return _export_base_to_remote(ppls, clone_id, transfer_type, mode)

    if role == "remote":
        if transfer_type != "results":
            raise RuntimeError("Remote can only export results back to base")
        return _export_remote_to_base(ppls, prev_transfer_id, mode)

    raise RuntimeError("Unknown lab role")

# ---------------------------
# Import Transfer
# ---------------------------
def import_transfer(zip_path: Path):
    """Import a transfer ZIP into current lab"""
    settings = get_shared_data()

    with zipfile.ZipFile(zip_path) as zf:
        meta = json.loads(zf.read("transfer.json"))

    direction = meta.get("direction")

    if direction == "base_to_remote":
        _ensure_remote(settings)
        return _import_on_remote(zip_path, meta)

    if direction == "remote_to_base":
        _ensure_base(settings)
        return _import_on_base(zip_path, meta)

    raise RuntimeError("Unknown transfer direction")

# ---------------------------
# Internal: BASE -> REMOTE
# ---------------------------
def _export_base_to_remote(ppls, clone_id, transfer_type, mode):
    settings = get_shared_data()
    _ensure_base(settings)
    base_data = Path(settings["data_path"])

    clone_dir = base_data / "Clones" / clone_id
    if not clone_dir.exists():
        raise ValueError(f"Clone '{clone_id}' not registered")

    transfer_id = f"t_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    zip_path = clone_dir / "transfers" / f"{transfer_id}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    transfer_meta = {
        "transfer_id": transfer_id,
        "origin_lab_id": settings.get("lab_id"),
        "target_lab_id": clone_id,
        "direction": "base_to_remote",
        "transfer_type": transfer_type,
        "created_at": datetime.utcnow().isoformat(),
        "ppls": {},
        "path_map": {},
        "component_map": {}
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pplid in ppls:
            P = PipeLine()
            if not P.verify(pplid=pplid):
                raise ValueError(f"Invalid pplid: {pplid}")
            P.load(pplid)

            # Config
            cfg_path = Path(P.get_path(of="config"))
            cfg_bytes = cfg_path.read_bytes()
            cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
            transfer_meta["ppls"][pplid] = {"config_hash": cfg_hash}

            zf.writestr(f"Configs/{pplid}.json", cfg_bytes)

            # Artifacts (files or dirs)
            for art in list(P.paths):
                if art == "config":
                    continue
                try:
                    p = Path(P.get_path(of=art))
                    if not p.exists():
                        continue
                    if p.is_dir():
                        for f in p.rglob("*"):
                            if f.is_file():
                                zf.write(f, f"Artifacts/{pplid}/{art}/{f.relative_to(p)}")
                    else:
                        zf.write(p, f"Artifacts/{pplid}/{art}/{p.name}")
                except Exception:
                    pass

        # write transfer.json last
        zf.writestr("transfer.json", json.dumps(transfer_meta, indent=4))

    # register in clone.json
    clone_json = clone_dir / "clone.json"
    clone_cfg = json.loads(clone_json.read_text())
    clone_cfg.setdefault("transfers", []).append(transfer_id)
    clone_json.write_text(json.dumps(clone_cfg, indent=4))

    # Move/copy mode
    if mode == "move":
        for pplid in ppls:
            # remove source artifacts after zip creation
            P = PipeLine(pplid)
            for art in list(P.paths):
                if art == "config":
                    continue
                p = Path(P.get_path(of=art))
                if p.exists():
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()

    return zip_path

# ---------------------------
# Internal: REMOTE -> BASE (results only)
# ---------------------------
def _export_remote_to_base(ppls, prev_transfer_id=None, mode="copy"):
    settings = get_shared_data()
    _ensure_remote(settings)

    transfer_id = f"r_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    zip_path = Path(settings["data_path"]) / "TransfersOut" / f"{transfer_id}.zip"
    zip_path.parent.mkdir(exist_ok=True)

    meta = {
        "transfer_id": transfer_id,
        "origin_lab_id": settings["lab_id"],
        "prev_transfer_id": prev_transfer_id,
        "direction": "remote_to_base",
        "transfer_type": "results",
        "created_at": datetime.utcnow().isoformat(),
        "ppls": ppls
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("transfer.json", json.dumps(meta, indent=4))

        for pplid in ppls:
            P = PipeLine(pplid)
            for art in list(P.paths):
                if art == "config":
                    continue
                p = Path(P.get_path(of=art))
                if not p.exists():
                    continue
                if p.is_dir():
                    for f in p.rglob("*"):
                        if f.is_file():
                            zf.write(f, f"Results/{pplid}/{art}/{f.relative_to(p)}")
                else:
                    zf.write(p, f"Results/{pplid}/{art}/{p.name}")

    # Move mode
    if mode == "move":
        for pplid in ppls:
            P = PipeLine(pplid)
            for art in list(P.paths):
                if art == "config":
                    continue
                p = Path(P.get_path(of=art))
                if p.exists():
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()

    return zip_path

# ---------------------------
# Internal: import on remote
# ---------------------------
def _import_on_remote(zip_path, meta):
    settings = get_shared_data()
    transfers_dir = Path(settings["data_path"]) / "Transfers"
    transfers_dir.mkdir(exist_ok=True)

    extract_dir = transfers_dir / meta["transfer_id"]
    _safe_extract(zip_path, extract_dir)

    ctx = TransferContext(
        path_map=meta.get("path_map", {}),
        component_map=meta.get("component_map", {}),
        transfer_id=meta["transfer_id"]
    )

    settings["transfer_context"] = ctx
    set_shared_data(settings)

# ---------------------------
# Internal: import on base
# ---------------------------
def _import_on_base(zip_path, meta):
    settings = get_shared_data()
    results_dir = Path(settings["data_path"]) / "RemoteResults" / meta["transfer_id"]
    _safe_extract(zip_path, results_dir)



import json
from pathlib import Path
import pandas as pd

def get_clones():
    """
    Return a DataFrame of all registered clones.

    Only valid in BASE lab.
    """
    settings = get_shared_data()

    if settings.get("lab_role") != "base":
        raise RuntimeError("get_clones() is allowed only in BASE lab")

    clones_root = Path(settings["data_path"]) / "Clones"
    rows = []

    if not clones_root.exists():
        return pd.DataFrame(
            columns=[
                "clone_id",
                "clone_type",
                "name",
                "desc",
                "created_at",
                "num_transfers",
            ]
        )

    for clone_dir in clones_root.iterdir():
        if not clone_dir.is_dir():
            continue

        cfg_path = clone_dir / "clone.json"
        if not cfg_path.exists():
            continue

        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)

            rows.append({
                "clone_id": cfg.get("clone_id"),
                "clone_type": cfg.get("clone_type"),
                "name": cfg.get("name"),
                "desc": cfg.get("desc"),
                "created_at": cfg.get("created_at"),
                "num_transfers": len(cfg.get("transfers", [])),
            })

        except Exception:
            # Skip broken clone entries safely
            continue

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values("created_at").reset_index(drop=True)

    return df
