# LwFlow (Lab workFlow)

## Domain-Independent, Secure, and Offline Computational Research Management

LwFlow is a self-contained Python framework for managing computational research experiments. It is designed to work with various domains, from machine learning and data processing to simulation and numerical analysis. Built on the principles of flexibility, reproducibility, and data privacy, LwFlow allows researchers to define, run, track, and transfer entire custom workflows offline.

LwFlow is **domain-agnostic**, leveraging abstract `Component` and `WorkFlow` classes that users can customize to fit their needs. This makes it ideal for use with frameworks like **PyTorch**, **TensorFlow**, or any other complex pipeline structures used in scientific computing.

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

You can install **LwFlow** using pip:

```

pip install lwflow

```

Alternatively, clone the repository for development purposes:

```

git clone [https://github.com/ExperQuick/lwf.git](https://github.com/ExperQuick/lwf.git)
cd LwFlow
pip install -e .

````

---

## 🚀 Getting Started: Running an Experiment Pipeline

LwFlow structures work around **Labs** (your project environment) and **Pipelines (PPLs)** (customizable experiment runs).

### 1. Setting Up the Lab Environment

The Lab manages file paths and databases. You only need to run this once per project.

```python
import os
from lwf.lab import create_project, lab_setup

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

Since LwFlow is domain-independent, you define your experiment logic by subclassing `WorkFlow` and `Component`.

```python
import os
from lwf.utils import WorkFlow, Component
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
        self.data_source = self.load_component(**args['data_source'])
        self.algorithm = self.load_component(**args['algorithm'])
        self.paths = ['results']  # Define artifact paths

    # Perform setup (e.g., loading large datasets/models into memory)
    def prepare(self):
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
from lwf.experiment import PipeLine

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

The `lwf.experiment` module provides powerful tools for managing your PPL database:

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
