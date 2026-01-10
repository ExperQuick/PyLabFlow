"""
Pipeline class  holder
"""

from typing import TypedDict, Optional, Dict, Union, Any
import json
import os
from pathlib import Path

import traceback

from .utils import (
    load_component,
    hash_args,
    get_invalid_loc_queries,
    Db,
    Component)

from ._transfer_utils import TransferContext
from .context import get_shared_data

class CompsDict(TypedDict):
    """
    fgfvv
    """

    model: Component
    loss: Component
    optimizer: Component
    dataset: Component
    metrics: Dict[str, Component]


# _core.py
class PipeLine:
    """
    khgkjv
    """

    def __init__(self, pplid=None):
        """
        Initialize the pipeline with default settings and empty components.
        """
        self._paths = ['config']
        self.settings = get_shared_data()

        self.pplid = None
        self.workflow = None
        
        self.cnfg = None
        self._prepared = False
        self.__db = Db(db_path=f"{self.settings['data_path']}/ppls.db")

        if pplid:
            self.load(pplid=pplid)

    def _save_config(self) -> None:
        """
        Save the current experiment configuration to a JSON file.

        This method writes the configuration stored in `self.cnfg` to a config file,
        but only if the hash of the current arguments matches the stored experiment ID.
        This ensures consistency and prevents accidental overwrites due to argument changes.

        Raises
        ------
        ValueError
            If the current arguments do not match the stored experiment's arguments,
            indicating that the configuration has been modified since it was created.
        """
        if self.verify(cnfg=self.cnfg) == self.cnfg["pplid"]:
            with open(self.get_path(of="config"), "w", encoding="utf-8") as out_file:
                json.dump(self.cnfg, out_file, indent=4)
        else:
            raise ValueError(
                f"can not save config for Experiment: {self.cnfg['pplid']}."
                "\n it's args has been changed"
            )

    def get_path(
        self,
        of: str,
        pplid: Optional[str] = None,
        args: Optional[Dict] = None
    ) -> str:
        """
            Generate a standardized file path for various experiment artifacts.

            Constructs and returns a file path based on the type of file (`of`), experiment ID,
            epoch number, and batch index, where applicable. Automatically creates necessary
            directories if they do not exist.

            Parameters
            ----------
            of : str
                The type of file to retrieve the path for. Supported values:
                - "config": Configuration file path.
                - "weight": Model weights file path.
                - "gradient": Saved gradients file path.
                - "history": Training history file path.
                - "quick": Quick config file path.
            pplid : str, optional
                Experiment ID. If not provided, uses the currently set `self.pplid`.
            epoch : int, optional
                Epoch number. Required for weight and gradient file paths.
                For weights, if not specified, the best epoch from config is used.
            batch : int, optional
                Batch index, required for gradient file paths.

            Returns
            -------
            str
                Full path to the specified artifact as a string with forward slashes.

            Raises
            ------
            ValueError
                If `pplid` is not set or invalid.
                If required parameters (`epoch`, `batch`) are missing for gradient paths.
                If the `of` argument is not one of the supported values.
        """
        pplid = pplid or self.pplid
        if not pplid:
            raise ValueError("Experiment ID (pplid) must be provided.")

        base_path = Path(self.settings["data_path"])

        if of == "config":
            path = Path("Configs") / f"{pplid}.json"

        else:
            if self.workflow is None:
                self.workflow = self.load_component(**self.cnfg['workflow'])
            path = self.workflow.get_path(of=of, pplid=pplid, args=args)

        path = base_path / path 
        path = path.as_posix()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def load(self, pplid: str, prepare: bool = False):
        """Load a pipeline configuration from disk"""
        self.reset()

        if not self.verify(pplid=pplid):
            raise ValueError(f"The pplid: {pplid} does not exist")

        cfg_path = self.get_path(of="config", pplid=pplid)
        with open(cfg_path, encoding="utf-8") as f:
            self.cnfg = json.load(f)
        self.pplid = pplid
    

        if prepare:
            self.prepare()






    def reset(self):
        """
        reset
        """
        self.pplid = None
        
        self.settings = get_shared_data()
        self.cnfg = None
        self._prepared = False
        self.workflow = None
        self.__db = Db(db_path=f"{self.settings['data_path']}/ppls.db")

    def load_component(self,loc: str, args: Optional[Dict[str, Any]] = None, setup: bool = True):
        if self.settings.get("lab_role") != "base":
            Tsx = TransferContext()

            loc = Tsx.map_loc(loc, pplid=self.pplid)

        comp =  load_component(loc=loc, args=args, setup=setup)
        comp.P = self
        return comp

    def verify(self, *, pplid: str = None, cnfg: Dict = None) -> Union[str, bool]:
        """
        Check whether a given experiment ID exists in the experiment database.

        Queries the experiments table to verify whether the specified experiment ID is recorded.

        Parameters
        ----------
        pplid : str
            The experiment ID to check.

        Returns
        -------
        Union[str, bool]
            Returns the `pplid` if it exists in the database, otherwise returns `False`.

        Examples
        --------
        >>> pipeline.verify("exp_001")
        'exp_001'

        >>> pipeline.verify("nonexistent_exp")
        False
        """

        if pplid:
            result = self.__db.query(
                "SELECT 1 FROM ppls WHERE pplid = ? LIMIT 1", (pplid,)
            )
            if len(result) > 0:
                return pplid
        elif cnfg:
            args = {
                'workflow':cnfg['workflow'],
                'args': cnfg['args']
            }
            args_hash = hash_args(args)
            rows = self.__db.query(
                "SELECT pplid FROM ppls WHERE args_hash =? LIMIT 1", (args_hash,)
            )
            if rows:
                pplid = rows[0][0]
                return pplid
        return False

    def _check_args(self, cnfg):
        t = get_invalid_loc_queries(cnfg)
        if t:
            raise ValueError(
                "Make sure all components are saved.\nReff: " + ", ".join(t)
            )
        t = self.verify(cnfg=cnfg)
        if t:
            raise ValueError(f"same configuration is already exists in: {t}")
    
    def new(
        self,
        pplid: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        prepare: bool = False,
    ) -> None:
        """
        Create a new experiment configuration and initialize its tracking files.

        Parameters
        ----------
        pplid : str, optional
            Unique experiment identifier. Raises ValueError if it already exists.
        args : dict, optional
            Configuration arguments for the experiment.
        prepare : bool, optional
            If True, calls `self.prepare()` after creation. Defaults to False.

        Raises
        ------
        ValueError
            If the experiment ID already exists or if monitor mode is invalid.
        KeyError
            If 'metrics' key is missing from settings.

        Behavior
        --------
        - Checks if the experiment ID already exists; raises an error if so.
        - Checks if the same configuration already exists using `verify`.
        - Initializes configuration dictionary with metadata.
        - Saves the configuration.
        - Creates an empty history CSV with columns for training and validation metrics and loss.
        - Initializes quick checkpoint file with default best and last epoch metrics.
        - Appends experiment metadata to the main experiments CSV.
        - Optionally calls `self.prepare()` if `prepare=True`.
        """
        if self.settings.get("lab_role") != "base":
            print("cant use new in remote lab")
            return
        if self.verify(pplid=pplid):
            raise ValueError(f"{pplid} is already exists  try  different id")
        self._check_args(args)
        t = {
            "pplid": pplid,
            **args
        }

        self.pplid = pplid
        self.cnfg = t

        try:
            self.workflow = self.load_component(**args['workflow'])
            self.workflow.new(args['args'])
        except:
            traceback.print_exc()
            
            raise
        
        self.__db.execute(
            "INSERT INTO ppls (pplid, args_hash) VALUES (?, ?)",
            (pplid, hash_args(args)),
        )

        self._save_config()        
        if prepare:
            self.prepare()
    
    def prepare(self) -> None:
        """
        Prepare the experiment by loading model, optimizer, metrics, loss, and data loaders.

        Loads components according to current configuration, initializes data loaders,
        and sets the best metric value based on the stored history and strategy.

        Raises
        ------
        ValueError
            If strategy monitor mode is not 'min' or 'max'.

        Behavior
        --------
        - Loads model and moves it to device.
        - Loads optimizer with model parameters.
        - Loads metrics and loss functions to device.
        - Creates training and validation data loaders.
        - Loads last saved model weights.
        - Initializes the best metric value from saved checkpoints or sets default.
        - Sets internal flag `_prepared` to True on success.
        """
        try:

            if self.settings.get("lab_role") != "base":
                Tsx = TransferContext()

                self.cnfg = Tsx.map_cnfg(self.cnfg)


            self.workflow = self.load_component(**self.cnfg['workflow'])
            self._prepared = self.workflow.prepare()



        except:
            traceback.print_exc()
   
    def run(self) -> None:

        if not self._prepared:
            print(
                "Preparation Error. Execute prepare() or set prepare=True before training."
            )
            return

        rows = self.__db.query(
            "SELECT logid FROM runnings WHERE pplid = ?", (self.pplid,)
        )
        if rows:
            print(f"pplid: {self.pplid} is running in logid: {rows[0][0]}")
            return

        try:
            self.__db.execute(
                "INSERT INTO runnings (pplid, logid) VALUES (?, ?)",
                (self.pplid, self.settings["logid"]),
            )

            self.workflow.run()
        except (RuntimeError, ValueError, KeyError) as e:
            print("Error in training loop:", e)
            traceback.print_exc()
        except BaseException as e:
            print("Unexpected error in training loop:", type(e).__name__, e)
            traceback.print_exc()
        finally:
            self.__db.execute("DELETE FROM runnings WHERE pplid = ?", (self.pplid,))

    def is_running(self):
        rows = self.__db.query(
            "SELECT logid FROM runnings WHERE pplid = ?", (self.pplid,)
        )
        if rows:
            return rows[0][0]
        return False
    
    @property    
    def should_running(self):
        rows = self.__db.query(
            "SELECT parity FROM runnings WHERE pplid = ?", (self.pplid,)
        )
        if rows and rows[0][0]=='stop':
            return False
        return True

    def stop_running(self):
        logid = self.is_running()
        if logid:
            self.__db.execute(
                "UPDATE runnings SET parity = ? WHERE logid = ?", ('stop', logid)
            )
            print(f"ppid:{self.pplid} will be stopped at logid:{logid} after current iteration")
        else:
            print("it is not running anywhere")
    
    @property
    def paths(self):
        artifs = self._paths
        if self.workflow is None and self.pplid:
            self.workflow = self.load_component(**self.cnfg['workflow'])
        if self.workflow:
            artifs += self.workflow.paths
        return artifs
   
    def clean(self):
        if self.cnfg==None:
            print("Empty Pipeline")
            return
        try:
            if self.workflow is None:
                self.workflow = self.load_component(**self.cnfg['workflow'])
            self.workflow.clean()
        except:
            traceback.print_exc()

    def status(self):
        if self.cnfg==None:
            print("Empty Pipeline")
            return
        try:
            if self.workflow is None:
                self.workflow = self.load_component(**self.cnfg['workflow'])
            return self.workflow.status()
        except:
            traceback.print_exc()

from copy import deepcopy

