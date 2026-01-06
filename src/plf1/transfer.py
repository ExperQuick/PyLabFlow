# transfer.py
import json
import zipfile
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import hashlib
import shutil
from copy import deepcopy

from .utils import Db
from .context import get_shared_data, set_shared_data
from ._pipeline import PipeLine
from ._transfer_utils import TransferContext

# ---------------------------
# Role enforcement
# ---------------------------
def _ensure_base(settings):
    if settings.get("lab_role") != "base":
        raise RuntimeError("Operation allowed only in BASE lab")

def _ensure_remote(settings):
    if settings.get("lab_role") != "remote":
        raise RuntimeError("Operation allowed only in REMOTE lab")


def _save_transfer_config(transfers_dir: Path, cfg: dict):
    cfg_path = transfers_dir / "transfer_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")



#---------------------
# Safe ZIP extraction
# ---------------------------
def _safe_extract(zip_path: Path, target_dir: Path):
    # extract  srcs  and locs
    #  then configs
    # then other artifacts
    print(target_dir)
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
        return _export_base_to_remote(ppls, clone_id, transfer_type)

    if role == "remote":
        if transfer_type != "results":
            raise RuntimeError("Remote can only export results back to base")
        return _export_remote_to_base(ppls, prev_transfer_id, mode)

    raise RuntimeError("Unknown lab role")

# ---------------------------
# Import Transfer
# ---------------------------
def import_transfer(zip_path: Path, **kwargs):
    """Import a transfer ZIP into current lab"""
    settings = get_shared_data()

    with zipfile.ZipFile(zip_path) as zf:
        meta = json.loads(zf.read("transfer.json"))

    direction = meta.get("direction")

    if direction == "base_to_remote":
        _ensure_remote(settings)
        return _import_on_remote(zip_path, meta, **kwargs)

    if direction == "remote_to_base":
        _ensure_base(settings)
        return _import_on_base(zip_path, meta)

    raise RuntimeError("Unknown transfer direction")


def collect_transfer_meta(ppls):
    all_paths = set()
    all_locs = set()

    for pplid in ppls:
        P = PipeLine(pplid)
        P.load(pplid)

        cfg_path = Path(P.get_path(of="config"))
        config = json.loads(cfg_path.read_text())

        paths, locs = extract_paths_and_locs(config)
        all_paths.update(paths)
        all_locs.update(locs)
    return  sorted(all_paths), sorted(all_locs)


def _payload_name(src: str) -> str:
    h = hashlib.sha1(src.encode()).hexdigest()[:8]
    return f"p_{h}"   ##  extention cdonsistency


def _write_src_payload(zf, lab_base: Path, paths: set):
    path_map = {}

    for src in sorted(paths):
        src_path = Path(src).resolve()
        payload_name = _payload_name(src)
        zip_root = Path("payload") / payload_name

        path_map[src] = zip_root.as_posix()

        if not src_path.exists():
            continue

        if src_path.is_dir():
            for f in src_path.rglob("*"):
                if f.is_file():
                    zf.write(
                        f,
                        zip_root / f.relative_to(src_path)
                    )
        else:
            zf.write(src_path, zip_root / src_path.name)

    return path_map

import ast
import hashlib
from pathlib import Path
from copy import deepcopy

# ---------------------------
# Internal: BASE -> REMOTE
# ---------------------------
def _export_base_to_remote(ppls, clone_id, transfer_type):
    settings = get_shared_data()
    _ensure_base(settings)

    lab_base = Path(settings["data_path"]).resolve()
    clone_dir = lab_base / "Clones" / clone_id
    if not clone_dir.exists():
        raise ValueError(f"Clone '{clone_id}' not registered")
    # datetime.now(datetime.timetz().)
    transfer_id = f"t_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    zip_path = clone_dir / "transfers" / f"{transfer_id}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    paths, locs = collect_transfer_meta(ppls)


    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        path_map = _write_src_payload(zf, lab_base, paths)
        # ---- LOCS: single .py file ----
        loc_map = _write_loc_payload(zf, locs, transfer_id)

        transfer_meta = {
            "transfer_id": transfer_id,
            "origin_lab_id": settings.get("lab_id"),
            "target_lab_id": clone_id,
            "direction": "base_to_remote",
            "transfer_type": transfer_type,
            "created_at": datetime.utcnow().isoformat(),
            "ppls": _collect_ppls_meta(ppls),
            "transfer_meta": {
                "src_map": path_map,
                "loc_map": loc_map,
            }
        }
        for pplid in ppls:
            P = PipeLine()
            if not P.verify(pplid=pplid):
                raise ValueError(f"Invalid pplid: {pplid}")

            P.load(pplid)

            # ---- CONFIG (always file) ----
            cfg = Path(P.get_path(of="config")).resolve()
            arcname = cfg.relative_to(lab_base)
            zf.write(cfg, arcname)

            # ---- ARTIFACTS (file or dir) ----
            for art in P.paths:
                # if art == "config":
                #     continue

                try:
                    p = Path(P.get_path(of=art)).resolve()
                    if not p.exists():
                        continue

                    if p.is_dir():
                        for f in p.rglob("*"):
                            if f.is_file():
                                zf.write(
                                    f,
                                    f.relative_to(lab_base)
                                )
                    else:
                        zf.write(
                            p,
                            p.relative_to(lab_base)
                        )
                except Exception:
                    pass

        # ---- transfer.json LAST ----
        zf.writestr("transfer.json", json.dumps(transfer_meta, indent=4))

    # ---- REGISTER TRANSFER ----
    clone_json = clone_dir / "clone.json"
    clone_cfg = json.loads(clone_json.read_text())
    clone_cfg.setdefault("transfers", []).append(transfer_id) # keep  ppls  too
    clone_json.write_text(json.dumps(clone_cfg, indent=4))

    return zip_path
# ---------------------------
# Helper: check conflicts
# ---------------------------
def _check_conflicting_ppls(ppls_meta: dict):
    """
    Check if any pplid in ppls_meta already exists in remote DB.
    Raise RuntimeError if conflict detected.
    """
    settings = get_shared_data()
    db = Db(db_path=str(Path(settings["data_path"]) / "ppls.db"))

    existing = {row[0] for row in db.query("SELECT pplid FROM ppls")}
    db.close()

    conflicts = [pplid for pplid in ppls_meta if pplid in existing]
    if conflicts:
        raise RuntimeError(f"Conflicting pplids exist in remote: {conflicts}")

# ---------------------------
# Helper: clean conflicting ppls
# ---------------------------
def clean_conflicting_ppls(ppls_meta: dict):
    """
    Delete pplid entries from remote DB and remove their transfer folders.
    """
    settings = get_shared_data()
    lab_base = Path(settings["data_path"])
    transfers_dir = lab_base / "Transfers"

    db = Db(db_path=str(lab_base / "ppls.db"))

    for pplid in ppls_meta:
        # Remove from DB
        db.execute("DELETE FROM ppls WHERE pplid=?", (pplid,))
        
        # Remove associated transfer folder(s)
        for folder in transfers_dir.glob(f"*{pplid}*"):
            if folder.is_dir():
                shutil.rmtree(folder)
    
    db.close()

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
def _import_on_remote(zip_path: Path, meta: dict, mode="copy", allow_overwrite=False):
    settings = get_shared_data()
    _ensure_remote(settings)

    lab_base = Path(settings["data_path"]).resolve()
    component_dir = Path(settings["component_dir"]).resolve()

    transfers_dir = lab_base / "Transfers"
    transfers_dir.mkdir(exist_ok=True)

    transfer_id = meta["transfer_id"]
    ppls_meta = meta.get("ppls", {})
    incoming_ppls = set(ppls_meta.keys())

    # --------------------------------------------------
    # Load / init transfer_config.json
    # --------------------------------------------------
    def _load_cfg():
        cfg_path = lab_base / "Transfers" / "transfer_config.json"
        if not cfg_path.exists():
            return {
                "active_transfer_id": None,
                "history": [],
                "ppl_to_transfer": {}
            }
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    cfg = _load_cfg()

    ppl_to_transfer = cfg.setdefault("ppl_to_transfer", {})

    # --------------------------------------------------
    # Enforce: one pplid → one transfer
    # --------------------------------------------------
    for pplid in incoming_ppls:
        if pplid in ppl_to_transfer:
            existing_tid = ppl_to_transfer[pplid]
            if existing_tid != transfer_id:
                if not allow_overwrite:
                    raise RuntimeError(
                        f"pplid '{pplid}' already belongs to transfer '{existing_tid}'"
                    )

    # --------------------------------------------------
    # DB conflict handling
    # --------------------------------------------------
    if not allow_overwrite:
        _check_conflicting_ppls(ppls_meta)
    else:
        clean_conflicting_ppls(ppls_meta)

    # --------------------------------------------------
    # Extract ZIP into Transfers/<transfer_id>
    # --------------------------------------------------
    transfer_dir = transfers_dir / transfer_id
    transfer_dir.mkdir(parents=True, exist_ok=True)

    _safe_extract(zip_path, transfer_dir)

    # --------------------------------------------------
    # Move generated LOC .py file into component_dir
    # --------------------------------------------------
    loc_py = transfer_dir / f"{transfer_id}.py"
    if loc_py.exists():
        component_dir.mkdir(parents=True, exist_ok=True)
        target = component_dir / loc_py.name

        if target.exists():
            target.unlink()

        if mode == "move":
            shutil.move(loc_py, target)
        else:
            shutil.copy2(loc_py, target)

    # --------------------------------------------------
    # Register pipelines in DB
    # --------------------------------------------------
    _register_ppls_in_db(ppls_meta)

    # --------------------------------------------------
    # 1️⃣ Materialize CONFIGS FIRST
    # --------------------------------------------------
    configs_dst = lab_base / "Configs"
    configs_dst.mkdir(parents=True, exist_ok=True)

    for pplid in incoming_ppls:
        src = transfer_dir / "Configs" / f"{pplid}.json"
        dst = configs_dst / f"{pplid}.json"

        if not src.exists():
            raise FileNotFoundError(f"Missing config for pplid {pplid}")

        if dst.exists():
            if not allow_overwrite:
                raise FileExistsError(f"Config already exists: {dst}")
            dst.unlink()

        shutil.move(src, dst)



    # --------------------------------------------------
    # Persist per-transfer metadata
    # --------------------------------------------------
    tm = meta.get("transfer_meta", {})

    transfer_meta = {
        "transfer_id": transfer_id,
        "origin_lab_id": meta.get("origin_lab_id"),
        "created_at": meta.get("created_at"),
        "ppls": sorted(incoming_ppls),
        "path_map": tm.get("path_map", {}), #  use from transfer dir with map
        "loc_map": tm.get("loc_map", {})
    }

    with open(transfer_dir / "transfer.json", "w", encoding="utf-8") as f:
        json.dump(transfer_meta, f, indent=4)

    # --------------------------------------------------
    # Update global transfer_config.json
    # --------------------------------------------------
    for pplid in incoming_ppls:
        ppl_to_transfer[pplid] = transfer_id

    if transfer_id not in cfg["history"]:
        cfg["history"].append(transfer_id)

    cfg["active_transfer_id"] = transfer_id

    _save_transfer_config(transfers_dir, cfg)
    # --------------------------------------------------
    # 2️⃣ Now materialize remaining paths via PipeLine
    # --------------------------------------------------
    for pplid in incoming_ppls:
        P = PipeLine(pplid=pplid)

        for key in P.paths:
            if key == "config":
                continue

            dst = Path(P.get_path(key))
            rel = dst.relative_to(lab_base)
            src = transfer_dir / rel

            if not src.exists():
                continue

            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.rglob("*"):
                    if f.is_file():
                        out = dst / f.relative_to(src)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(f, out)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)

    return True


# ---------------------------
# Internal: import on base
# ---------------------------

def _import_on_base(zip_path, meta):
    settings = get_shared_data()
    results_dir = Path(settings["data_path"]) / "RemoteResults" / meta["transfer_id"]
    _safe_extract(zip_path, results_dir)

    # Optionally register empty transfer_context for consistency
    ctx = TransferContext(path_map={}, component_map={}, transfer_id=meta["transfer_id"])
    settings["transfer_context"] = ctx
    set_shared_data(settings)


def _register_ppls_in_db(ppls_meta: dict):
    settings = get_shared_data()
    db = Db(db_path=str(Path(settings["data_path"]) / "ppls.db"))

    existing = {
        row[0] for row in db.query("SELECT pplid FROM ppls")
    }

    for pplid, meta in ppls_meta.items():
        if pplid in existing:
            continue

        db.execute(
            """
            INSERT INTO ppls (pplid, args_hash, status, created_time)
            VALUES (?, ?, ?, ?)
            """,
            (
                meta["pplid"],
                meta.get("args_hash"),
                meta.get("status", "init"),
                meta.get("created_time"),
            )
        )

    db.close()

def _collect_ppls_meta(ppls):
    settings = get_shared_data()
    db = Db(db_path=str(Path(settings["data_path"]) / "ppls.db"))

    rows = db.query(
        """
        SELECT pplid, args_hash, status, created_time
        FROM ppls
        WHERE pplid IN ({})
        """.format(",".join("?" * len(ppls))),
        tuple(ppls)
    )

    db.close()

    meta = {}
    for pplid, args_hash, status, created_time in rows:
        meta[pplid] = {
            "pplid": pplid,
            "args_hash": args_hash,
            "status": status,
            "created_time": created_time
        }

    return meta

def register_transfer_mapping(ppls_meta: dict, transfer_id: str, loc_map: dict):
    """
    Register pplid -> transfer info and loc_map in shared data
    """
    settings = get_shared_data()
    registry = settings.get("transfer_registry", {})

    for pplid in ppls_meta:
        registry[pplid] = {
            "transfer_id": transfer_id,
            "loc_map": loc_map
        }

    settings["transfer_registry"] = registry
    set_shared_data(settings)


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
        print("get_clones() is allowed only in BASE lab")
        return

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


def get_transfers():

    settings = get_shared_data()
    if settings.get("lab_role") == "base":
        print('for base lab  call  `get_clones')
        return 

    lab_base = Path(settings["data_path"]).resolve()

    transfers_dir = lab_base / "Transfers"
    transfers_dir.mkdir(exist_ok=True)

    cfg_path = transfers_dir / "transfer_config.json"
    transfers = json.loads(cfg_path.read_text(encoding="utf-8"))
    return transfers["ppl_to_transfer"]







import ast
import hashlib
from pathlib import Path
from copy import deepcopy

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _loc_hash(loc: str) -> str:
    return hashlib.sha1(loc.encode("utf-8")).hexdigest()[:8]

def _class_dependencies(class_defs, old_to_new):
    """
    Returns: dict[new_class_name] -> set[new_base_names]
    Only considers local classes in old_to_new.
    """
    deps = {}
    for _, (class_name, cls_node, _) in class_defs.items():
        new_name = old_to_new[class_name]
        deps[new_name] = set()
        for base in cls_node.bases:
            if isinstance(base, ast.Name) and base.id in old_to_new:
                deps[new_name].add(old_to_new[base.id])
    return deps

def _toposort(deps):
    ordered = []
    temp = set()
    perm = set()

    def visit(n):
        if n in perm:
            return
        if n in temp:
            raise RuntimeError("Cyclic class dependency detected")
        temp.add(n)
        for m in deps.get(n, []):
            visit(m)
        temp.remove(n)
        perm.add(n)
        ordered.append(n)

    for n in deps:
        visit(n)
    return ordered[::-1]  # bases first

def _parse_module(module_path: Path):
    code = module_path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    imports = []
    classes = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node
    return imports, classes

def _resolve_base_name(base):
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None

def _resolve_imports(import_nodes):
    out = {}
    for node in import_nodes:
        if isinstance(node, ast.ImportFrom) and node.module:
            for n in node.names:
                out[n.asname or n.name] = f"{node.module}.{n.name}"
        elif isinstance(node, ast.Import):
            for n in node.names:
                out[n.asname or n.name] = n.name
    return out

# ------------------------------------------------------------
# AST Renamer
# ------------------------------------------------------------

class RenameSymbols(ast.NodeTransformer):
    def __init__(self, rename_map):
        self.rename_map = rename_map

    def visit_Name(self, node):
        if node.id in self.rename_map:
            return ast.copy_location(ast.Name(id=self.rename_map[node.id], ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self.rename_map:
            node.value.id = self.rename_map[node.value.id]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.returns and isinstance(node.returns, ast.Name) and node.returns.id in self.rename_map:
            node.returns.id = self.rename_map[node.returns.id]
        return node

    def visit_arg(self, node):
        if node.annotation and isinstance(node.annotation, ast.Name):
            if node.annotation.id in self.rename_map:
                node.annotation.id = self.rename_map[node.annotation.id]
        return node

# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------

def _write_loc_payload(zf, locs: set, transfer_id: str):
    settings = get_shared_data()
    component_dir = Path(settings["component_dir"]).resolve()

    # ------------------------------
    # Step 1: Collect all classes recursively
    # ------------------------------
    collected = {}  # loc -> (class_name, cls_node, imports)

    def collect_loc(loc: str):
        if loc in collected or "." not in loc:
            return

        module_str, class_name = loc.rsplit(".", 1)
        module_path = (component_dir / Path(*module_str.split("."))).with_suffix(".py")
        if not module_path.exists():
            return

        imports, classes = _parse_module(module_path)
        if class_name not in classes:
            return

        cls_node = deepcopy(classes[class_name])
        collected[loc] = (class_name, cls_node, imports)

        import_map = _resolve_imports(imports)

        # recurse into base classes
        for base in cls_node.bases:
            base_name = _resolve_base_name(base)
            if not base_name:
                continue
            if base_name in import_map:
                base_loc = import_map[base_name]
            else:
                base_loc = f"{module_str}.{base_name}"
            base_module = (component_dir / Path(*base_loc.split(".")[:-1])).with_suffix(".py")
            if base_module.exists():
                collect_loc(base_loc)

    for loc in sorted(locs):
        collect_loc(loc)

    if not collected:
        raise RuntimeError("No component classes found for provided LOCs")

    # ------------------------------
    # Step 2: Generate unique names + loc_map
    # ------------------------------
    old_to_new = {}
    loc_map = {}
    for loc, (class_name, _, _) in collected.items():
        h = _loc_hash(loc)
        new_name = f"{class_name}{h}"
        old_to_new[class_name] = new_name
        loc_map[loc] = f"{transfer_id}.{new_name}"

    # ------------------------------
    # Step 3: Topologically sort classes by base dependencies
    # ------------------------------
    deps = _class_dependencies(collected, old_to_new)
    emit_order = _toposort(deps)
    name_to_entry = {old_to_new[class_name]: (loc, class_name, cls_node, imports)
                     for loc, (class_name, cls_node, imports) in collected.items()}

    # ------------------------------
    # Step 4: Rewrite ASTs with full renaming
    # ------------------------------
    renamer = RenameSymbols(old_to_new)
    all_imports = set()
    class_chunks = []

    for new_name in emit_order:
        loc, class_name, cls_node, imports = name_to_entry[new_name]
        cls_node.name = new_name
        cls_node = renamer.visit(cls_node)
        ast.fix_missing_locations(cls_node)

        for imp in imports:
            all_imports.add(ast.unparse(imp))

        mod = ast.Module(body=[cls_node], type_ignores=[])
        ast.fix_missing_locations(mod)

        class_chunks.append(f"# --- {loc} ---\n{ast.unparse(mod)}\n")

    # ------------------------------
    # Step 5: Write final .py file
    # ------------------------------
    imports_code = "\n".join(sorted(all_imports))
    py_code = imports_code + "\n\n" + "\n\n".join(class_chunks[::-1])
    py_name = f"{transfer_id}.py"
    zf.writestr(py_name, py_code)

    return loc_map
