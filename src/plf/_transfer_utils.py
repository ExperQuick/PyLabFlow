import json
from pathlib import Path
from .context import get_shared_data
    
    
    


def _load_transfer_config():
    settings = get_shared_data()

    lab_base = Path(settings["data_path"]).resolve()

    transfers_dir = lab_base / "Transfers"
    transfers_dir.mkdir(exist_ok=True)

    cfg_path = transfers_dir / "transfer_config.json"
    if not cfg_path.exists():
        return {
            "active_transfer_id": None,
            "history": [],
            "ppl_to_transfer": {} #sqlit3
        }
    return json.loads(cfg_path.read_text(encoding="utf-8"))
# ---------------------------


# ---------------------------
# TransferContext
# ---------------------------
class TransferContext:
    """Runtime context for remapping paths and components on remote."""

    def __init__(self):
        settings = get_shared_data()

        transfers_dir = Path(settings["data_path"]).resolve() / "Transfers"
        self.transfers_dir = transfers_dir
        self._cfg = _load_transfer_config()

    def _load_transfer_meta(self, transfer_id: str) -> dict:
        meta_path = self.transfers_dir / transfer_id / "transfer.json"
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def map_cnfg(self, cnfg): 

        pplid = cnfg['pplid']       
        def remap(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if "loc" in k and isinstance(v, str):
                        # Map LOC via transfer context
                        d[k] = self.map_loc(v, pplid=pplid)
                    elif 'src' in k and isinstance(v, str):
                        # Map file paths via transfer context
                        d[k] = self.map_src(v)
                    else:
                        remap(v)
            elif isinstance(d, list):
                for item in d:
                    remap(item)
        remap(cnfg)
        return cnfg

    def map_src(self, src: str, pplid: str) -> str:
        src = Path(src).as_posix()
        transfer_id = self._cfg["ppl_to_transfer"].get(pplid)
        if not transfer_id:
            return src

        meta = self._load_transfer_meta(transfer_id)
        path_map = meta.get("path_map", {})

        for src, dst in path_map.items():
            dst = self.transfers_dir / transfer_id /"payload"/ dst
            if src.startswith(src):
                return src.replace(src, dst, 1)

        return src

    def map_loc(self, loc: str, pplid: str) -> str:
        transfer_id = self._cfg["ppl_to_transfer"].get(pplid)
        if not transfer_id:
            return loc

        meta = self._load_transfer_meta(transfer_id)
        loc_map = meta.get("loc_map", {})

        return loc_map.get(loc, loc)
