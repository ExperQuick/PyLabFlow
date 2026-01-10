"""
This module manages shared runtime context across different parts of an application.
It tracks session-specific data, caller identification (e.g., Jupyter, scripts), and allows for
runtime library path registration. It supports reproducibility, logging, and analysis by preserving
contextual metadata during execution.
"""

import os
import sys
import inspect
import json
from ipykernel.connect import get_connection_file

_CONTEXT_DATA = {}


def _get_context_id() -> str:
    """
    Retrieve a unique identifier for the current file or module's execution context.

    Returns
    -------
    str
        The absolute file path of the root frame (entry point of the call stack),
        used as a context identifier.
    """

    frame = inspect.currentframe()
    try:
        # Walk up the call stack to find the caller's frame
        while frame.f_back:
            frame = frame.f_back
        # Get the filename of the caller
        filename = frame.f_code.co_filename
        return filename
    finally:
        # Clean up to prevent reference cycles
        del frame


# Ensure a context exists in the dictionary
def _ensure_context() -> str:
    """
    Ensure the current context ID exists in the internal shared data dictionary.

    Returns
    -------
    str
        The context ID (usually the filename of the root calling script).
    """
    context_id = _get_context_id()
    if context_id not in _CONTEXT_DATA:
        _CONTEXT_DATA[context_id] = {}
    return context_id


# Function to get data for the current context
def get_shared_data() -> dict:
    """
    Retrieve the shared data dictionary associated with the current context.

    Returns
    -------
    dict
        A dictionary that holds all shared data for the current context/session.
    """
    context_id = _ensure_context()
    return _CONTEXT_DATA[context_id]


# Function to set data for the current context


def set_shared_data(data: dict, logid: str=None) -> dict:
    """
    Set the shared data dictionary for the current context.

    Parameters
    ----------
    data : dict
        The context data to store. If not a dictionary, only 'logid' is saved.
    logid : str
        A string identifier used for logging or session tracking.

    Returns
    -------
    dict
        The updated context dictionary.
    """
    context_id = _ensure_context()

    if isinstance(data, dict):
        if logid:
            data["logid"] = logid
        _CONTEXT_DATA[context_id] = data
    else:
        _CONTEXT_DATA[context_id] = {"logid": logid}

    return _CONTEXT_DATA[context_id]


def get_caller() -> str:
    """
    Identify the current session or script origin.
    Returns a string like 'jupyter_session:<id>' or 'script:<filename>'.
    """
    try:
        with open(get_connection_file(), encoding="utf-8") as c:
            caller = json.load(c)["jupyter_session"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, RuntimeError):
        if len(sys.argv) > 0 and sys.argv[0]:  # means it's being run as a script
            caller = f"script:{os.path.basename(sys.argv[0])}"
        else:
            caller = "unknown-session"
    return caller


def register_libs_path(libs_dir: str) -> None:
    """
    Add a local library directory to Python's sys.path if it exists.

    Useful for loading local modules dynamically without installing them system-wide.

    Parameters
    ----------
    libs_dir : str
        The path to the directory containing the libraries.

    Raises
    ------
    ValueError
        If the given path does not point to a valid directory.
    """
    libs_path = os.path.abspath(libs_dir)

    if not os.path.isdir(libs_dir):
        raise ValueError(f"Invalid directory: {libs_dir}")
    # libs_parent = os.path.dirname(libs_path)
    if libs_path not in sys.path:
        sys.path.append(libs_path)
