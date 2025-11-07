"""
This module provides
"""

import sys
import os
from typing import overload, List, Callable, Dict, Any, Union, Optional, Literal
import json
import hashlib
import warnings
import importlib
import sqlite3
import traceback
from abc import ABC, abstractmethod

from copy import deepcopy

import pandas as pd

__all__ = [
    "load_component",
    "hash_args",
    "Component",
    "WorkFlow",
    "Db",
    "extract_all_locs",
    "get_invalid_loc_queries",
    'filter_configs',
    'get_matching'
]


class ComponentLoadError(Exception):
    """
    ComponentLoadError
    """


def load_component(
    loc: str, args: Optional[Dict[str, Any]] = None, setup: bool = True
) -> Callable:
    """
        Dynamically load and optionally initialize a component class.

        This utility imports a class from a given module path and instantiates it.
        If the class defines a `setup` method and `setup=True`, it calls `setup(args)`
        and returns the initialized component. Otherwise, it returns the raw instance.

        Parameters
        ----------
        loc : str
            Fully qualified class location in dot notation (e.g., 'CompBase.models.MyModel').
            If no dot is present, it is assumed the class is defined in `__main__`.

        args : dict, optional
            Dictionary of arguments to pass to the `setup()` method, if applicable.
            Defaults to an empty dict.

        setup : bool, optional
            Whether to invoke the component’s `setup` method after instantiation. Defaults to True.

        Returns
        -------
        Any
            An instance of the loaded class, either raw or configured via `setup()`.

        Raises
        ------
        ComponentLoadError
            If the specified class is not found in the target module.
        ImportError
            If the module cannot be imported.
    """
    args = args or {}

    # Parse module and class name
    if "." in loc:
        module_path, class_name = loc.rsplit(".", 1)
        if module_path in sys.modules:
            module = sys.modules[module_path]
            if getattr(module, "__spec__", None):
                module = importlib.reload(module)
        else:
            module = importlib.import_module(module_path)
    else:
        # No dot means class is in __main__
        module = sys.modules["__main__"]
        class_name = loc
        warnings.warn(
            f"{loc} component is not saved. "
            "Make sure to save it in an appropriate location before"
            "initiating an experiment, test, or report."
        )
    if not hasattr(module, class_name):
        raise ComponentLoadError(
            f"Class '{class_name}' not found in module '{module.__name__}'"
        )
    component_cls = getattr(module, class_name)
    component = component_cls()

    # If setup method exists and setup flag is True, call it with args
    if setup and hasattr(component, "_setup"):
        return component.setup(args)
    return component


class Component(ABC):
    """
        Base class for all components with dynamic loading capability.

        Attributes:
            loc (str): Location identifier for the component.
            args (dict): Expected keys for arguments.
    """

    def __init__(self, loc: str = None):
        self.loc = self.__class__.__name__ if loc is None else loc
        self.args = {}
        self.P = None
    
    def load_component(self, loc: str, args: Optional[Dict[str, Any]] = None, setup: bool = True):
        comp =  load_component(loc=loc, args=args,setup=setup)
        comp.P = self.P
        return comp

    def check_args(self, args: dict) -> bool:
        """Check whether provided args contain all required keys."""
        return all(arg in args for arg in self.args)

    def setup(self, args: Dict[str, Any]) -> Optional[Any]:
        """
            Set up the component with provided arguments.

            Args:
                args: Dictionary of arguments to initialize the component.

            Returns:
                Optional[Any]: Initialized component or setup result.
        """
        if self.check_args(args):
            # print(154,args)
            try:
                self._setup(args)
                return self
            except AttributeError as exc:
                raise AttributeError(
                    f"Component '{self.loc}' does not implement '_setup'"
                ) from exc
        
        traceback.print_exc()
        raise ValueError(f"Arguments {args} are incompatible with '{self.loc}'")
        
    
    @abstractmethod
    def _setup(self, args: Dict[str, Any],P=None) -> Optional[Any]:
        """
            Private setup to be overridden in subclasses.

            Args:
                args: Dictionary of arguments.

            Returns:
                Optional[Any]
        """
        raise NotImplementedError(f"Component '{self.loc}' must implement '_setup'")

class WorkFlow(Component, ABC):
    """
    Abstract base class for all workflows.

    Workflows are intended to be managed by the `PipeLine` class. 
    When a pipeline is created via `PipeLine.new()`, it takes a workflow configuration,
    instantiates the workflow, and passes the workflow-specific arguments to it. 
    Therefore, all required workflow parameters should be validated at this point 
    using the workflow's template.

    ----------------------------
    GUIDELINES
    ----------------------------

    1. Initialization and Argument Validation
       - When `PipeLine.new()` is called:
           * The pipeline verifies that the workflow exists and is properly defined.
           * All workflow-specific arguments (`args`) should be checked against the workflow's
             template to ensure completeness and correctness.
           * Duplicate configurations (same args) should be detected and prevented.
       - The workflow must implement a `new(name: str, **kwargs)` method to initialize itself
         with workflow-specific arguments.

    2. Preparation
       - The workflow's `prepare()` method is called by the pipeline to initialize
         all necessary components or resources required for execution.
       - Workflow implementations should convert any required objects or configuration
         entries from the pipeline config (`self.P.cnfg`) into Python objects here.
       - After `prepare()` completes, `run()` should be safe to execute.

    3. Execution
       - The workflow's `run()` method is called by the pipeline when execution starts.
       - `run()` should implement the main computation or processing according to the workflow's purpose.
       - Workflows should assume that `prepare()` has already been called.

    4. Path Management
       - Workflows must implement `get_path(of: str, args: Optional[Dict] = None) -> str`.
       - The pipeline only handles the path for the configuration file; all other paths
         are redirected to the workflow.
       - All output, intermediate, or artifact paths should be tracked in `self.paths`.
       - Avoid hard-coded paths; always generate paths dynamically so pipelines can move or copy artifacts safely.

    5. Optional Methods
       - `clean()`: Delete temporary files, cached outputs, or intermediate artifacts.
       - `status() -> str`: Return workflow status or progress information.
       - These methods are called by the pipeline when needed.

    6. Best Practices
       - Ensure deterministic behavior: same inputs should produce the same outputs.
       - Handle missing resources or exceptions gracefully with clear error messages.
       - Use consistent naming for workflow IDs, versions, and artifact paths.
       - Load components dynamically via `self.load_component`.
       - Workflows should be independent of any specific domain or technology.

    ----------------------------
    REQUIRED METHODS
    ----------------------------
    - new(self, name: str, **kwargs)
    - prepare(self, *args, **kwargs)
    - run(self, *args, **kwargs)
    - get_path(self, of: str, args: Optional[Dict] = None) -> str

    ----------------------------
    OPTIONAL METHODS
    ----------------------------
    - clean(self, *args, **kwargs)
    - status(self, *args, **kwargs) -> str
    """

    @abstractmethod
    def prepare(self):
        """
        Called when PipeLine.prepare() is executed.
        Convert necessary components from the configuration dictionary
        into Python objects here so that the workflow is ready for run().
        """

    @abstractmethod
    def run(self):
        """
        Called when PipeLine.run() is executed.
        Implement the main computation or processing logic here.
        """

    @abstractmethod
    def new(self, args):
        """
        Initialize a new workflow instance with the given name and arguments.
        """

    @abstractmethod
    def get_path(self, of: str, args: Optional[Dict] = None) -> str:
        """
        Return a standardized path for the requested artifact type (`of`).
        All workflow-specific path options should be listed in `self.paths`.
        This ensures that when a pipeline is transferred, all artifacts are correctly located.
        """


    def clean(self):
        """
        Clean up temporary files, cached outputs, or intermediate artifacts.
        """
        pass


    def status(self) -> str:
        """
        Return the current status or progress of the workflow.
        """
        return {}
    
class Db:
    """
        Lightweight SQLite wrapper with foreign key enforcement.

        Args:
            db_path (str): Path to the SQLite database file.

        Raises:
            FileNotFoundError: If the directory for the DB path doesn't exist.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

        # Check if directory exists
        dir_name = os.path.dirname(db_path)
        if dir_name and not os.path.isdir(dir_name):
            raise FileNotFoundError(f"Directory does not exist: {dir_name}")

        self._connect()

    def _connect(self) -> None:
        """Establishes a database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")

    def execute(self, query: str, params: tuple = ()) -> Optional[sqlite3.Cursor]:
        """
        Execute a SQL query (INSERT, UPDATE, DELETE).

        Args:
            query (str): SQL query string.
            params (tuple): Query parameters.

        Returns:
            sqlite3.Cursor or None
        """
        if not self.conn:
            raise ConnectionError("No database connection.")
        try:
            cur = self.conn.cursor()
            cur.execute(query, params)
            self.conn.commit()
            return cur
        except sqlite3.Error as e:
            print(f"[SQLITE ERROR] {e}")
            return None

    def query(self, query: str, params: tuple = ()) -> list:
        """
        Execute a SELECT query and fetch all results.

        Args:
            query (str): SQL query string.
            params (tuple): Query parameters.

        Returns:
            list: Query results.
        """
        cursor = self.execute(query, params)
        return cursor.fetchall() if cursor else []

    def close(self) -> None:
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def hash_args(args: Dict[str, Any]) -> str:
    """
        Generate a SHA-256 hash from a dictionary of arguments.

        This is commonly used to uniquely identify a configuration or set of parameters.

        Parameters
        ----------
        args : dict
            The dictionary of arguments to be hashed. Must be JSON-serializable.

        Returns
        -------
        str
            A SHA-256 hash string representing the input dictionary.

        Raises
        ------
        TypeError
        If the dictionary contains non-serializable values.
    """
    dict_str = json.dumps(args, sort_keys=True, separators=(",", ":"))
    # print(dict_str)
    hhas  = hashlib.sha256(dict_str.encode()).hexdigest()
    # print(hhas)
    return hhas

from typing import Union, Dict, List, Any

def extract_all_locs(d: Union[Dict, List]) -> List[str]:
    """
    Recursively extract all 'loc' values from nested dictionaries or lists.
    A component is defined as a dict with a 'loc' key and optional 'args'.
    """
    locs = []

    if isinstance(d, dict):
        # If this is a component dict
        if "loc" in d:
            locs.append(d["loc"])
            if "args" in d and isinstance(d["args"], (dict, list)):
                locs.extend(extract_all_locs(d["args"]))
        else:
            # Otherwise, check all values in the dict
            for v in d.values():
                locs.extend(extract_all_locs(v))

    elif isinstance(d, list):
        for item in d:
            locs.extend(extract_all_locs(item))

    return locs

from typing import Union, Dict, List

def get_invalid_loc_queries(d: Union[Dict, List], parent_key: str = "") -> List[str]:
    """
    Recursively search a nested dictionary or list for invalid 'loc' entries.

    A 'loc' entry is considered invalid if it is not a string or does not contain a dot ('.').

    Parameters
    ----------
    d : Union[Dict, List]
        The nested dictionary or list to inspect.
    parent_key : str, optional
        The concatenated key path used during recursion, by default "".
        This helps identify where in the nested structure the invalid 'loc' is.

    Returns
    -------
    List[str]
        A list of key paths (strings) to all invalid 'loc' entries found.
        Each path uses '>' for dict keys and '[index]' for list indices.
    """

    queries = []

    if isinstance(d, dict):
        # Check this dict itself
        if "loc" in d:
            loc_val = d["loc"]
            if not isinstance(loc_val, str) or "." not in loc_val:
                queries.append(parent_key)

        # Now recurse into its values
        for k, v in d.items():
            new_key = f"{parent_key}>{k}" if parent_key else k
            queries.extend(get_invalid_loc_queries(v, new_key))

    elif isinstance(d, list):
        for idx, item in enumerate(d):
            item_key = f"{parent_key}[{idx}]" if parent_key else f"[{idx}]"
            queries.extend(get_invalid_loc_queries(item, item_key))

    return queries

def _flatten_nested_locs(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested dictionaries by extracting the 'loc' value from inner dictionaries.

    For each top-level key in the dictionary, if its value is a dictionary and contains
    nested dictionaries, the nested dictionary is replaced with its 'loc' value.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary with potentially nested dictionaries to flatten.

    Returns
    -------
    Dict[str, Any]
        The modified dictionary where nested dictionaries are replaced by their 'loc' value.
    """
    for exp in data:
        if isinstance(data[exp], dict):
            for key in list(data[exp].keys()):
                if isinstance(data[exp][key], dict):
                    data[exp][key] = data[exp][key].get("loc")
    return data


def _apply_key_filter(data: Dict[str, Any], q: str) -> Dict[str, Any]:
    """
    Filter a dictionary by extracting the value corresponding to a specific key.

    If the value of `q` is a dictionary, it extracts the 'loc' value.
    If it is a simple value (str or int), it keeps it. Otherwise, the entry is removed.

    Parameters
    ----------
    data : Dict[str, Any]
        The input dictionary where each value is a dictionary.
    q : str
        The key to filter and extract from each value dictionary.

    Returns
    -------
    Dict[str, Any]
        A dictionary where each top-level key now maps to the extracted 'loc' value,
        simple value, or is removed if no valid value exists.
    """
    for exp in list(data.keys()):
        val = data[exp].get(q)
        if isinstance(val, dict):
            data[exp] = val.get("loc")
        elif isinstance(val, (int, str)):
            data[exp] = val
        else:
            data.pop(exp)
    return data


def _apply_kv_filter(data: Dict[str, Any], k: str, v: str) -> Dict[str, Any]:
    if v == "":
        data = {exp: args[k]["loc"] for exp, args in data.items() if k in args}
        return pd.DataFrame.from_dict(data, orient="index", columns=[k])

    to_del = []
    for exp in list(data.keys()):
        t = None
        val = data[exp].get(k)
        if isinstance(val, dict):
            t = val.get("loc")
            data[exp] = val.get("args", {})
        elif isinstance(val, (int, str)):
            t = str(val)
        if t != v:
            to_del.append(exp)
    for d in to_del:
        data.pop(d, None)
    return data


@overload
def filter_configs(
    query: str,
    ids: List[str],
    loader_func: Callable[[str], Dict[str, Any]],
    params: Literal[True],
) -> pd.DataFrame: ...


@overload
def filter_configs(
    query: str,
    ids: List[str],
    loader_func: Callable[[str], Dict[str, Any]],
    params: Literal[False] = False,
) -> List[str]: ...


def filter_configs(
    query: str,
    ids: List[str],
    loader_func: Callable[[str], Dict[str, Any]],
    params: bool = False,
) -> Union[List[str], pd.DataFrame]:
    """
    Filter and extract information from a collection of configurations.
    """
    qry = query.split(">")
    data = {i: deepcopy(loader_func(i)) for i in ids}

    for q in qry:
        if q == "":
            for _, exp in data.items():
                return list(exp.keys())
        elif "=" in q:
            k, v = q.split("=")
            result = _apply_kv_filter(data, k, v)
            if isinstance(result, pd.DataFrame):
                return result
            data = result
        else:
            data = _apply_key_filter(data, q)

    if params:
        data = _flatten_nested_locs(data)
        return pd.DataFrame.from_dict(data, orient="index")
    return list(data.keys())

def get_matching(
    base_id: str,
    get_ids_fn: Callable[[], List[str]],
    loader_fn: Callable[[str], Dict[str, Any]],
    query: str = None,
    include=False,
) -> Dict[str, List[str]]:
    """
    Get IDs of configurations that match the same flattened key-value pair(s) as a base config.

    Args:
        base_id (str): ID of the base configuration.
        get_ids_fn (Callable): Function to retrieve all configuration IDs.
        loader_fn (Callable): Function to load a configuration given its ID.
        query (str, optional): Specific query key or 'key=value' pair.

    Returns:
        Dict[str, List[str]]: Mapping of matched query to list of matching IDs.
    """

    def flatten(d, parent_key="", sep=">"):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                if "loc" in v:
                    items[new_key] = v["loc"]
                elif "args" in v:
                    items.update(flatten(v["args"], new_key, sep=sep))
                else:
                    items.update(flatten(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    all_ids = [eid for eid in get_ids_fn() if eid != base_id]

    base_args = flatten(deepcopy(loader_fn(base_id)))

    # Load and flatten others
    flat_data = {}
    for eid in all_ids:
        obj = loader_fn(eid)
        flat_data[eid] = flatten(deepcopy(obj))

    result = {}

    if query:
        if "=" in query:
            key, val = query.split("=")
        else:
            key, val = query, base_args.get(query)

        if key not in base_args:
            return {}

        base_val = base_args.get(key)
        if val is not None and str(val) != str(base_val):
            return {}

        matched = [eid for eid, args in flat_data.items() if args.get(key) == base_val]
        if matched:
            result[f"{key}={base_val}"] = matched
    else:
        for key, base_val in base_args.items():
            matched = [
                eid for eid, args in flat_data.items() if args.get(key) == base_val
            ]
            if matched:
                result[f"{key}={base_val}"] = matched
    if include:
        for i in result:
            result[i] += [base_id]
    return result

