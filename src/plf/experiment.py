"""
This module  have all  function  for initiating pipeline and training
"""

from typing import Optional, Dict, List
import json
import os
import shutil
from collections import defaultdict
import pandas as pd

from .context import get_shared_data
from .utils import Db, filter_configs, get_matching, is_comp
from ._pipeline import PipeLine


__all__ = [
    "PipeLine",
    "get_ppls",
    "get_ppl_details",
    "get_ppl_status",
    "archive_ppl",
    "delete_ppl",
    "transfer_ppl",
    "group_by_common_columns",
    'filter_ppls'
]


def get_ppls() -> List[str]:
    """
    Retrieves a list of all pipeline IDs from the database.

    Returns
    -------
    list of str
        A list containing all pipeline IDs.
    """
    db = Db(db_path=f"{get_shared_data()['data_path']}/ppls.db")
    rows = db.query("SELECT pplid from ppls")
    db.close()
    rows = [i[0] for i in rows]
    return rows



def get_ppl_details(ppls: Optional[list] = None) -> pd.DataFrame:
    ppls = get_ppls() if ppls is None else ppls
    records = {}

    for pplid in ppls:
        P = PipeLine(pplid=pplid)
        wf = P.cnfg['workflow'].get('loc', f"workflow_{pplid}")  # fallback if 'loc' missing

        P.workflow = P.load_component(**P.cnfg['workflow'])
        data = {}
        for comp_name in P.workflow.template:
            comp_cfg = P.cnfg['args'][comp_name]
            data[comp_name] = comp_cfg['loc'] if is_comp(comp_cfg) else comp_cfg

        # Save data under this workflow and pipeline ID
        if wf not in records:
            records[wf] = {}
        records[wf][pplid] = data

    # Convert each workflow's dict of pipeline data to a DataFrame
    for wf_key in records:
        records[wf_key] = pd.DataFrame.from_dict(records[wf_key], orient='index')

    # If only one workflow, return its DataFrame directly
    if len(records) == 1:
        return next(iter(records.values()))

    # Otherwise, return dict of DataFrames keyed by workflow
    return records

def get_ppl_status(ppls: Optional[list] = None) -> pd.DataFrame:
    data = { }
    ppls = get_ppls() if ppls is None else ppls
    for i in ppls:
        P = PipeLine(pplid=i)
        data[i] = P.status()
    df = pd.DataFrame.from_dict(data, orient='index')
    return df

def multi_run(ppls: Dict[str, int], last_epoch: int = 10, patience: int = 5) -> None:
    """
    Train multiple pipelines up to a maximum number of epochs with optional patience.

    Parameters
    ----------
    ppls : dict[str, int]
        Dictionary of pipeline IDs to some integer values (usage unclear).
    last_epoch : int, optional
        Maximum number of epochs to train each pipeline, by default 10.
    patience : int, optional
        Number of epochs to wait for improvement before stopping (currently unused), by default 5.

    Raises
    ------
    ValueError
        If any pipeline ID in `ppls` is not found in the existing pipelines.
    """
    exs = get_ppls()
    if not all(ex in ppls for ex in exs):
        raise ValueError(f"pplids should be from {', '.join(exs)}")
    for exp in ppls:
        P = PipeLine(pplid=exp)
        P.prepare()
        P.run()

def get_runnings():
    db = Db(db_path=f"{PipeLine().settings['data_path']}/ppls.db")

    cursor = db.execute("SELECT * FROM runnings")
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]

    df = pd.DataFrame(rows, columns=col_names)
    return df

def archive_ppl(ppls: List[str], reverse: bool = False) -> None:
    """
    Archive or unarchive pipelines by moving their related files
    between active and archived folders.
    """

    settings = get_shared_data()
    if isinstance(ppls, str):
        ppls = [ppls]

    source = settings["data_path"]
    destin = os.path.join(source, "Archived")
    if reverse:
        source, destin = destin, source

    db = Db(db_path=os.path.join(source, "ppls.db"))
    existing = [i[0] for i in db.query("SELECT pplid FROM ppls")]

    if not all(p in existing for p in ppls):
        raise ValueError(f"One or more ppls in {ppls} are invalid")

    # Check if any ppl is currently running
    for pplid in ppls:
        rows = db.query("SELECT logid FROM runnings WHERE pplid = ?", (pplid,))
        if rows:
            print(f"pplid: {pplid} is running in logid: {rows[0][0]}")
            return

    logging = settings["logging"]

    for pplid in ppls:
        file_moves = []  # Track successful moves for rollback

        try:
            # Move logging files
            for log in logging:
                src = os.path.join(source, *log.split("."))
                dst = os.path.join(destin, *log.split("."))
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)
                    file_moves.append((dst, src))  # for rollback
                else:
                    print(f"Missing file: {src}")

            # Move config file
            src_cfg = os.path.join(source, "config", f"{pplid}.json")
            dst_cfg = os.path.join(destin, "config", f"{pplid}.json")
            if os.path.exists(src_cfg):
                os.makedirs(os.path.dirname(dst_cfg), exist_ok=True)
                shutil.move(src_cfg, dst_cfg)
                file_moves.append((dst_cfg, src_cfg))
            else:
                print(f"Missing config: {src_cfg}")

            # Database copy
            rows = db.query("SELECT pplid, args_hash FROM ppls WHERE pplid = ?", (pplid,))
            if rows:
                db1 = Db(db_path=os.path.join(destin, "ppls.db"))
                db1.execute("INSERT INTO ppls (pplid, args_hash) VALUES (?, ?)", rows[0])
                db1.close()

            # Delete original DB record only after all moves succeed
            db.execute("DELETE FROM ppls WHERE pplid = ?", (pplid,))

            txt = "unarchived" if reverse else "archived"
            print(f"{pplid} {txt} successfully")

        except Exception as e:
            print(f"Error while processing {pplid}: {e}")
            # Rollback any files moved
            for moved_dst, moved_src in reversed(file_moves):
                try:
                    os.makedirs(os.path.dirname(moved_src), exist_ok=True)
                    shutil.move(moved_dst, moved_src)
                    print(f"Rolled back {moved_dst} → {moved_src}")
                except Exception as rollback_err:
                    print(f"Rollback failed for {moved_dst}: {rollback_err}")

        finally:
            db.close()

def archive_ppl(ppls: List[str], reverse: bool = False) -> None:

    if isinstance(ppls, str):
        ppls = [ppls]

    data_path = os.path.abspath(get_shared_data()["data_path"])  # Normalize path

    def make_dst_path(src_path: str) -> str:
        """Insert or remove 'Archived' after data_path based on reverse flag."""
        rel = os.path.relpath(src_path, data_path)
        parts = rel.split(os.sep)
        if reverse:
            if parts[0] == "Archived":
                return os.path.join(data_path, *parts[1:])
            else:
                raise ValueError(f"Path is not archived: {src_path}")
        else:
            return os.path.join(data_path, "Archived", *parts)

    for pplid in ppls:
        try:
            # Choose correct DB
            db_path = os.path.join(data_path, "Archived" if reverse else "", "ppls.db")
            db = Db(db_path=db_path)

            if not reverse:

                df = get_runnings()
                if pplid in df['pplid']:
                    print(f"pplid: {pplid} is running ")
                    print(df[df['pplid']==pplid].values)
                    continue

            # Prepare pipeline
            P = PipeLine()
            if reverse:
                cfg_path = os.path.join(data_path, "Archived", "Configs", f"{pplid}.json")
                if not os.path.exists(cfg_path):
                    print(f"Missing archived config for {pplid}: {cfg_path}")
                    continue
                with open(cfg_path) as fl:
                    P.cnfg = json.load(fl)
                    P.pplid = P.cnfg['pplid']
                    P.settings['data_path'] = os.path.join(P.settings['data_path'], "Archived")
            else:
                P.load(pplid=pplid)

            file_paths = [P.get_path(p) for p in P.paths]

            # Start moving files
            file_moves = []

            for src in file_paths:
                src = os.path.abspath(src)

                if not src.startswith(data_path):
                    print(f"Skipping unrelated file: {src}")
                    continue

                try:
                    dst = make_dst_path(src)
                except ValueError as ve:
                    print(ve)
                    continue

                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)
                    file_moves.append((dst, src))  # for rollback
                else:
                    print(f"Missing file: {src}")

            # DB transfer
            rows = db.query("SELECT pplid, args_hash FROM ppls WHERE pplid = ?", (pplid,))
            if not rows:
                print(f"{pplid} not found in DB.")
            else:
                dest_db_path = os.path.join(data_path, "" if reverse else "Archived", "ppls.db")
                dest_db = Db(db_path=dest_db_path)
                dest_db.execute("INSERT INTO ppls (pplid, args_hash) VALUES (?, ?)", rows[0])
                dest_db.close()
                db.execute("DELETE FROM ppls WHERE pplid = ?", (pplid,))

            print(f"{pplid} {'unarchived' if reverse else 'archived'} successfully")

        except Exception as e:
            print(f"Error while processing {pplid}: {e}")
            for dst, src in reversed(file_moves):
                try:
                    os.makedirs(os.path.dirname(src), exist_ok=True)
                    shutil.move(dst, src)
                    print(f"Rolled back {dst} → {src}")
                except Exception as rollback_err:
                    print(f"Rollback failed for {dst}: {rollback_err}")

        finally:
            db.close()

def filter_ppls(
        query: str, ppls: Optional[List[str]] = None, params: bool = False
    ) -> list:
    """
    Filters pipelines based on a query string applied to their configurations.

    Parameters
    ----------
    query : str
        A query string used to filter pipeline configurations.
    ppls : list or None, optional
        List of pipeline IDs to filter. If None, all pipelines are considered.
    params : bool, optional
        Whether to return parameters of matching pipelines along with their IDs.

    Returns
    -------
    list
        Filtered list of pipeline IDs or tuples of (pplid, params) if `params` is True.
    """
    ppls = get_ppls() if ppls is None else ppls

    def loader(pplid):
        P = PipeLine()
        P.load(pplid=pplid)
        return P.cnfg["args"]

    return filter_configs(query, ppls, loader, params)

def get_matching_ppls(
        base_pplid: str, query: Optional[str] = None, include=False
    ) -> List:
    """
    Retrieve pipelines matching a base pipeline ID and optional query.

    Parameters
    ----------
    base_pplid : str
        The base pipeline ID to compare against.
    query : str or None, optional
        Optional query string to filter matching pipelines.

    Returns
    -------
    list
        A list of pipeline IDs matching the criteria.
    """

    def loader(pplid):
        P = PipeLine()
        P.load(pplid=pplid)
        return P.cnfg

    return get_matching(
        base_id=base_pplid,
        get_ids_fn=get_ppls,
        loader_fn=loader,
        query=query,
        include=include,
    )


import os
import shutil
from typing import List

def delete_ppl(ppls: List[str]) -> None:
    """
    Permanently delete archived pipelines, including config files,
    logging files, and database records.

    Parameters
    ----------
    ppls : list[str]
        List of pipeline IDs to delete from archive.
    """
    settings = get_shared_data()
    if isinstance(ppls, str):
        ppls = [ppls]

    archive_path = os.path.join(settings["data_path"], "Archived")
    db_path = os.path.join(archive_path, "ppls.db")

    if not os.path.exists(db_path):
        print("Archived DB not found.")
        return

    db = Db(db_path=db_path)
    existing = [i[0] for i in db.query("SELECT pplid FROM ppls")]

    for pplid in ppls:
        if pplid not in existing:
            print(f"Pipeline '{pplid}' not found in archive. Skipping.")
            continue

        try:
            # Delete config file
            cfg_path = os.path.join(archive_path, "config", f"{pplid}.json")
            if os.path.exists(cfg_path):
                os.remove(cfg_path)
                print(f"Deleted config: {cfg_path}")
            else:
                print(f"Config file not found: {cfg_path}")

            # Delete logging files
            for log in settings["logging"]:
                log_path = os.path.join(archive_path, *log.split("."))
                if os.path.exists(log_path):
                    if os.path.isdir(log_path):
                        shutil.rmtree(log_path)
                    else:
                        os.remove(log_path)
                    print(f"Deleted log path: {log_path}")
                else:
                    print(f"Log path not found: {log_path}")

            # Delete DB entry
            db.execute("DELETE FROM ppls WHERE pplid = ?", (pplid,))
            print(f"Deleted DB record for: {pplid}")

        except Exception as e:
            print(f"Error while deleting '{pplid}': {e}")

    db.close()

def transfer_ppl(
    ppls: List[str], transfer_type: str = "export", mode: str = "copy", env=True
) -> None:
    """
        Transfers pipeline data between main storage and transfer folder.

        Args
        ----
            ppls (list[str]): List of pipeline IDs to transfer.
            transfer_type (str, optional): Type of transfer, either 'export' (default) or 'import'.
                'export' moves data from main storage to transfer folder,
                'import' moves data from transfer folder back to main storage.
            mode (str, optional): Transfer mode, either 'copy' (default) or 'move'.
                'copy' duplicates files, 'move' relocates files.
        

        Raises
        ------
            ValueError: If `transfer_type` or `mode` is invalid,
                        or if any pipeline ID is not found in the source records.

        Returns
        -------
            None
    """

    settings = get_shared_data()

    if isinstance(ppls, str):
        ppls = [ppls]

    base_path = settings["data_path"]

    if transfer_type == "export":
        source = base_path
        destin = f"{base_path}/Transfer"
    elif transfer_type == "import":
        source = f"{base_path}/Transfer"
        destin = base_path
    else:
        raise ValueError(
            f"Invalid transfer_type: {transfer_type}. Expected 'export' or 'import'."
        )

    df = pd.read_csv(f"{source}/ppls.csv")
    records = df["name"] if "name" in df.columns else df["pplid"]

    if not all(exp in records.values for exp in ppls):
        raise ValueError(f"One or more of ppls: {ppls} is/are invalid")

    if mode == "copy":
        for exp in ppls:
            shutil.copy2(f"{source}/Configs/{exp}.json", f"{destin}/Configs/{exp}.json")
            shutil.copy2(
                f"{source}/Histories/{exp}.csv", f"{destin}/Histories/{exp}.csv"
            )
            shutil.copytree(f"{source}/Weights/{exp}", f"{destin}/Weights/{exp}")
            shutil.copytree(f"{source}/Gradients/{exp}", f"{destin}/Gradients/{exp}")

        if "name" in df.columns:
            df_to_transfer = df[df["name"].isin(ppls)]
        else:
            df_to_transfer = df[df["pplid"].isin(ppls)]
        df_to_transfer.to_csv(f"{destin}/ppls.csv", mode="a", header=False, index=False)
        print(f"{ppls} are transferred successfully")

    elif mode == "move":
        for exp in ppls:
            shutil.move(f"{source}/Weights/{exp}", f"{destin}/Weights/")
            shutil.move(f"{source}/Configs/{exp}.json", f"{destin}/Configs/{exp}.json")
            shutil.move(
                f"{source}/Histories/{exp}.csv", f"{destin}/Histories/{exp}.csv"
            )
            shutil.move(f"{source}/Gradients/{exp}", f"{destin}/Gradients/")

        if "name" in df.columns:
            df_to_move = df[df["name"].isin(ppls)]
            df_remaining = df[~df["name"].isin(ppls)]
        else:
            df_to_move = df[df["pplid"].isin(ppls)]
            df_remaining = df[~df["pplid"].isin(ppls)]

        df_remaining.to_csv(f"{source}/ppls.csv", index=False)
        df_to_move.to_csv(f"{destin}/ppls.csv", mode="a", header=False, index=False)

        print(f"{ppls} are transferred successfully")

    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'copy' or 'move'.")


def group_by_common_columns(
    records: Dict[str, pd.DataFrame],
    ) -> Dict[frozenset, List[str]]:
    """
        Group pipeline records by their common set of DataFrame columns.

        Parameters
        ----------
            records (dict): A dictionary where keys are pipeline IDs and values are pandas DataFrames
                            (e.g., training histories with various metrics).

        Returns
        -------
            dict: A dictionary mapping each unique set of column names (as a `frozenset`) to a list of
                pipeline IDs sharing that column structure.

        Example
        -------
            >>> records = {
            ...     "exp1": pd.DataFrame(columns=["epoch", "train_loss", "val_loss"]),
            ...     "exp2": pd.DataFrame(columns=["epoch", "train_loss", "val_loss"]),
            ...     "exp3": pd.DataFrame(columns=["epoch", "accuracy", "val_accuracy"])
            ... }
            >>> group_by_common_columns(records)
            {
                frozenset({'epoch', 'train_loss', 'val_loss'}): ['exp1', 'exp2'],
                frozenset({'epoch', 'accuracy', 'val_accuracy'}): ['exp3']
            }
    """
    cols = {k: frozenset(v.columns) for k, v in records.items()}
    group_map = defaultdict(list)
    for k, colset in cols.items():
        group_map[colset].append(k)
    return group_map
