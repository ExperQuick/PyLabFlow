# PyLabFlow 
<div align="center">

[![Documentation Status](https://readthedocs.org/projects/PyLabFlow/badge/?version=latest)](https://PyLabFlow.readthedocs.io/en/latest/?badge=latest) 
[![PyPI version](https://badge.fury.io/py/PyLabFlow.svg?icon=si%3Apython)](https://pypi.org/project/PyLabFlow/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/PyLabFlow?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/PyLabFlow)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

</div>

## Domain-Independent, Secure, and Offline Computational Research Management

PyLabFlow is a self-contained Python framework for managing computational research experiments. It is designed to work with various domains, from machine learning and data processing to simulation and numerical analysis. Built on the principles of flexibility, reproducibility, and data privacy, PyLabFlow allows researchers to define, run, track, and transfer entire custom workflows offline.

PyLabFlow is **domain-agnostic**, leveraging abstract `Component` and `WorkFlow` classes that users can customize to fit their needs. This makes it ideal for use with frameworks like **PyTorch**, **TensorFlow**, or any other complex pipeline structures used in scientific computing.

---

## ✨ Key Capabilities

- **Offline-First**:
  - All experiment tracking and data management are handled locally using file systems and SQLite databases, ensuring **complete data privacy** and uninterrupted work, even without an internet connection.

- **100% Customization**:
  - Built on abstract `Component` and `WorkFlow` classes, users can define their own logic for data loading, processing, simulation, and analysis.

- **Pipeline Tracking (PPLs)**:
  - Centralized management of experiment configurations, status (e.g., `init`, `running`, `frozen`, `cleaned`), and history.

- **Reproducibility Guarantee**:
  - Each pipeline (PPL) is uniquely identified by the cryptographic hash of its entire configuration, preventing configuration drift and ensuring exact reproducibility.

- **Seamless Transfer**:
  - Easily archive experiments, or transfer project setups (including configurations and output artifacts) between local directories, cloud storage, or high-performance computing (HPC) nodes without path reconfiguration.

- **Session Logging**:
  - Automatic tracking of the session origin (e.g., Jupyter session, script filename) and execution time for enhanced auditability.

---

## 🛠 Installation

You can install **PyLabFlow** using pip:

```

pip install PyLabFlow

```

Alternatively, clone the repository for development purposes:

```

git clone https://github.com/ExperQuick/PyLabFlow.git
```
```
cd PyLabFlow
```
```
pip install -e .

````

---

## 🚀 Getting Started: Running an Experiment Pipeline

PyLabFlow structures work around **Labs** (your project environment) and **Pipelines (PPLs)** (customizable experiment runs).

### 1. Setting Up the Lab Environment

The Lab manages file paths and databases. You only need to run this once per project.

```python
import os
from plf.lab import create_project, lab_setup

# Define mandatory project settings
SETTINGS = {
    "project_name": "General_Research_Lab",
    "project_dir": "/path/to/my/research_projects",  # Parent directory for the project
    "component_dir": "/path/to/my/custom_components",  # Directory for custom components and workflows
}

# Create project structure, databases, and settings file
settings_path = create_project(SETTINGS)
print(f"Project structure and settings created at: {settings_path}")

# Set up the lab environment for the current Python session
lab_setup(settings_path)
````

---

### 2. Defining a Custom WorkFlow

Since PyLabFlow is domain-independent, you define your experiment logic by subclassing `WorkFlow` and `Component`.

```python
import os
from plf.utils import WorkFlow, Component
from typing import Dict, Any

# A generic computational component
class MyComputationalComponent(Component):
    def _setup(self, args: Dict[str, Any]):
        print(f"Setting up component with args: {args}")
        self.data = args.get("initial_value", 0)

# Define the flow that combines components and executes the run logic
class GenericDataWorkflow(WorkFlow):
    
    # Initialize the pipeline run configuration
    def new(self, args: Dict[str, Any]):
        # Ensure all required configuration keys are provided
        if not self.template.issubset(set(args.keys())):
            raise ValueError(f'the args should have {", ".join(self.template- set(list(args.keys())))}')
        
        

    # Perform setup (e.g., loading large datasets/models into memory)
    def prepare(self):
        self.data_source = self.load_component(**args['data_source'])
        self.algorithm = self.load_component(**args['algorithm'])
        print("Preparing workflow: loading external data or setting up environment...")
        return True

    # Main execution logic (e.g., training loop, simulation run)
    def run(self):
        print(f"Running PPL: {self.P.pplid}")
        result = self.data_source.data + 10
        print(f"Final result: {result}")

    # Define standardized paths for saving artifacts
    def get_path(self, of: str, pplid: str, args: Dict) -> str:
        if of == 'results':
            return os.path.join(self.P.settings['data_path'], 'Results', f'{pplid}_output.txt')
        raise NotImplementedError(f"Path for artifact type '{of}' is not defined in GenericDataWorkflow.")

    # Clean up temporary files or output artifacts
    def clean(self):
        print(f"Cleaning artifacts for {self.P.pplid}...")
        # Add logic to delete files here

    # Return execution status or key metrics
    def status(self):
        return {"last_result": 100, "status_detail": "Completed successfully"}
```

---

### 3. Creating and Executing a Pipeline

Once your components are defined, you can create and execute the pipeline using the `PipeLine` class.

```python
from plf.experiment import PipeLine

# Define the full configuration with fully qualified paths to components
pipeline_config = {
    "workflow": {
        "loc": "my_workflows.GenericDataWorkflow",
        "args": {}
    },
    "args": {
        "data_source": {"loc": "my_workflows.MyComputationalComponent", "args": {"initial_value": 42}},
        "algorithm": {"loc": "my_workflows.MyComputationalComponent", "args": {"param_b": 5}},
    }
}

# Create a new pipeline. Configuration is hashed and logged
P = PipeLine()
P.new(pplid="ppl_data_run_001", args=pipeline_config)

# Prepare the environment
P.prepare()

# Run the workflow
P.run()
```

---

## 💾 Experiment Management Tools

The `plf.experiment` module provides powerful tools for managing your PPL database:

* **`get_ppls()`**: List all active pipeline IDs in the current Lab.
* **`get_ppl_status()`**: Returns a DataFrame summarizing the status, last run, and key metrics for all PPLs.
* **`filter_ppls(query)`**: Filters PPLs based on configuration arguments (e.g., `filter_ppls("data_source=my_workflows.MyComponent")`).
* **`archive_ppl(ppls)`**: Archives a pipeline, moving its configurations and artifacts to an archived folder for safe storage.
* **`archive_ppl(ppls, reverse=True)`**: Unarchives a pipeline and returns it to the active environment.
* **`delete_ppl(ppls)`**: Permanently deletes a pipeline from the archive.
* **`stop_running()`**: Gracefully stops a currently running pipeline after its current iteration completes.

---

## 📜 License

This project is licensed under the **Apache License 2.0**.  © 2025 BBEK Anand




If you use PyLabFlow in your research, please consider citing it's ititial version:

```bibtex
@article{Hati2025PyTorchLabFlow,
  title={PyTorchLabFlow: A Local-First Framework for Reproducible Deep Learning Experiments},
  author={Bibekananda Hati and Yogeshwar Singh Dadwhal},
  journal={Journal of Applied Bioanalysis},
  volume={11},
  number={S6},
  pages={538--546},
  year={2025},
  doi={10.53555/jab.v11si6.1940},
  url={https://doi.org/10.53555/jab.v11si6.1940}
}


```

