"""
create or use uor lab
"""
from pathlib import Path
import os
import json
from datetime import datetime
from typing import Optional
import pandas as pd

from .context import set_shared_data, get_caller, register_libs_path, get_shared_data
from .utils import Db

__all__ = ["lab_setup", "create_project", "get_logs"]
 
def export_settigns():
    settings = get_shared_data()
    # Change project_path to data_path parent
    pth = os.path.join(Path(settings['data_path']).parent, settings["project_name"] + ".json")
    with open(pth, "w", encoding="utf-8") as out_file:
        json.dump(settings, out_file, indent=4)
    return pth

def create_project(settings: dict) -> str:
    """
    Create the project directory structure, databases, and settings file.
    Returns the absolute path to the settings JSON.
    """
    project_dir = os.path.abspath(settings["project_dir"])
    project_name = settings["project_name"]
    component_dir = os.path.abspath(settings["component_dir"])

    # Derived paths
    data_path = os.path.join(project_dir, project_name)
    setting_path = os.path.join(data_path, f"{project_name}.json")

    # Update settings with absolute paths
    settings.update({
        "project_dir": project_dir,
        "component_dir": component_dir,
        "data_path": data_path,
        "setting_path": setting_path,
    })

    # Create required directories
    for key in ["data_path", "component_dir"]:
        os.makedirs(settings[key], exist_ok=True)
        
    # Remove old databases if any
    for db_file in ["logs.db", "ppls.db"]:
        db_path = os.path.join(data_path, db_file)
        if os.path.exists(db_path):
            os.remove(db_path)

    # Setup DBs and shared data
    setup_databases(settings)
    set_shared_data(settings)

    # Save settings file
    with open(setting_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)

    return setting_path

def create_and_init_db(db_path: str, tables: list, init_statements: list = None):
    db = Db(db_path=db_path)
    for table_sql in tables:
        db.execute(table_sql)
    if init_statements:
        for stmt, params in init_statements:
            db.execute(stmt, params)
    db.close()

def setup_databases(settings: dict):
    """
    Sets up the required databases for the lab project, including:
    - logs.db (with logs table)
    - ppls.db (with ppls, edges, runnings tables)
    - Archived/ppls.db (with ppls table)
    """
    # ---- logs.db ----
    logs_db_path = os.path.join(settings["data_path"], "logs.db")
    logs_table = """
        CREATE TABLE IF NOT EXISTS logs (
            logid TEXT PRIMARY KEY,
            called_at TEXT NOT NULL,
            created_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """
    log_init = [("INSERT INTO logs (logid, called_at) VALUES (?, ?)", ('log0', get_caller()))]
    create_and_init_db(logs_db_path, [logs_table], log_init)

    # ---- ppls.db ----
    ppls_db_path = os.path.join(settings["data_path"], "ppls.db")
    ppls_tables = [
        """
        CREATE TABLE IF NOT EXISTS ppls (
            pplid TEXT PRIMARY KEY,
            args_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'init'
                CHECK(status IN ('init', 'running', 'frozen', 'cleaned')),
            created_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS edges (
            edgid INTEGER PRIMARY KEY AUTOINCREMENT,
            prev TEXT NOT NULL,
            next TEXT NOT NULL,
            desc TEXT,
            directed BOOL DEFAULT TRUE,
            FOREIGN KEY(prev) REFERENCES ppls(pplid),
            FOREIGN KEY(next) REFERENCES ppls(pplid)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS runnings (
            runid INTEGER PRIMARY KEY AUTOINCREMENT,
            pplid NOT NULL,
            logid TEXT DEFAULT NULL,
            parity TEXT DEFAULT NULL,
            started_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(pplid) REFERENCES ppls(pplid)
        );
        """
    ]
    create_and_init_db(ppls_db_path, ppls_tables)

    os.makedirs(os.path.join(settings["data_path"], "Archived"), exist_ok=True)
    # ---- Archived/ppls.db ----
    archived_ppls_db_path = os.path.join(settings["data_path"], "Archived", "ppls.db")
    archived_ppls_table = """
        CREATE TABLE IF NOT EXISTS ppls (
            pplid TEXT PRIMARY KEY,
            args_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'init'
                CHECK(status IN ('init', 'running', 'frozen', 'cleaned')),
            created_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """
    create_and_init_db(archived_ppls_db_path, [archived_ppls_table])


def lab_setup(settings_path: Optional[str]) -> None:
    if settings_path and os.path.exists(settings_path):
        with open(settings_path, encoding="utf-8") as sp:
            settings = json.load(sp)
    else:
        raise ValueError("Provide either settings_path or settings for lab setup")


    caller = get_caller()

    log_path = os.path.join(settings["data_path"], "logs.db")
    db = Db(db_path=log_path)

    # Get current number of logs
    cursor = db.execute("SELECT COUNT(*) FROM logs")
    row_count = cursor.fetchone()[0]
    logid = f"log{row_count}"
    # Insert new log
    db.execute(
        "INSERT INTO logs (logid, called_at) VALUES (?, ?)",
        (logid,  caller)
    )

    db.close()
    set_shared_data(settings, logid)
    register_libs_path(settings["component_dir"])
   
def get_logs():
    settings = get_shared_data()
    log_path = os.path.join(settings["data_path"], "logs.db")
    db = Db(db_path=log_path)
    cursor = db.execute("SELECT * FROM logs")
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]
    db.close()
    df = pd.DataFrame(rows, columns=col_names)
    return df

