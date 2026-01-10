==============================
Framework Overview
==============================

This framework provides a modular and reproducible environment for managing experiments.
It introduces three core concepts — **Component**, **Workflow**, and **Pipeline** — each
representing a different layer of abstraction and flexibility in the experimental process.

These concepts allow users to design, execute, and reproduce complex computational or
machine learning experiments with a clear separation of concerns between logic,
configuration, and execution history.

.. contents::
   :local:
   :depth: 2


------------------------------
1. Component
------------------------------

A **Component** is the smallest reusable unit in the framework.  
It represents a **functional block** — such as a data loader, model, optimizer, or evaluator —
that performs a single, well-defined task.

A component can be:

- A **Python function** or **class** encapsulated in a file.
- A **reference** to a shared resource, library, or custom code base.
- A **configurable unit**, whose behavior depends on arguments specified at runtime.

**Key Ideas**

- **Reusability:** Components can be reused across different workflows and pipelines.
- **Configurability:** Each component can have a flexible configuration (`args`) dictionary.
- **Isolation:** Components do not depend directly on other components — they only define
  *what* they do, not *when* they run.

**Example**

.. code-block:: json

    {
        "loc": "components.data.load_dataset",
        "args": {
            "path": "./data/train.csv",
            "batch_size": 32
        }
    }

In this example, the component loads a dataset and exposes it to downstream steps.
The `"loc"` field points to the function to execute, and `"args"` defines how it behaves.

This separation of *location* and *arguments* allows the same logical component to be
reused in many workflows with different parameters — increasing modularity.


------------------------------
2. Workflow
------------------------------

A **Workflow** defines **the structure and order** in which components are executed.
While a component is a single building block, a workflow describes *how components connect*.

Think of a workflow as a **template or blueprint** for a process.  
It defines *what happens* and *in what sequence* but does not fix the data or model parameters.

**Key Ideas**

- **Declarative Composition:** A workflow lists components and their dependencies.
- **Dynamic Construction:** Workflows can be loaded from JSON, YAML, or Python definitions.
- **Reproducibility:** The same workflow can be reused across multiple runs, ensuring consistency.

**Example**

.. code-block:: json

    {
        "workflow": {
            "loc": "components.training.supervised_training",
            "template": ["load_data", "build_model", "train", "evaluate"]
        },
        "args": {
            "load_data": {"loc": "components.data.load_dataset", "args": {"path": "data/train.csv"}},
            "build_model": {"loc": "components.model.create_cnn", "args": {"num_layers": 5}},
            "train": {"loc": "components.training.train_epoch", "args": {"epochs": 10}},
            "evaluate": {"loc": "components.eval.compute_accuracy", "args": {}}
        }
    }

Here, the workflow specifies **the topology** of execution (the “template”) and how each step
is realized via components.

Each step can be substituted or reconfigured without breaking the structure.
This makes workflows **flexible, composable, and shareable** across projects.


------------------------------
3. Pipeline
------------------------------

A **Pipeline** is a **runtime instantiation** of a workflow with concrete settings, logs,
and status tracking.

While a workflow defines *what should happen*, a pipeline defines *what actually happened*.

It binds together:
- The workflow definition.
- The specific component configurations.
- Metadata about the environment, logs, and execution history.

Each pipeline has a unique identifier (`pplid`) and is tracked in a database (`ppls.db`),
which stores:
- Pipeline metadata (hashes, creation time, status).
- Relationships between pipelines (via the `edges` table).
- Active runs (`runnings` table).

**Pipeline Lifecycle**

1. **Creation:**  
   A pipeline is created using `PipeLine(pplid=...)`, loading its configuration and workflow.

2. **Preparation:**  
   It sets up its directories, loads components, and initializes resources.

3. **Execution:**  
   The pipeline runs through its workflow components in order (or dynamically).

4. **Status Tracking:**  
   Each run is recorded in the database, making pipelines reproducible and auditable.

5. **Archival / Transfer:**  
   Finished pipelines can be archived, deleted, or transferred between environments,
   preserving their full state.

**Flexibility**

- You can create many pipelines from a single workflow with different component arguments.
- You can rerun or resume a pipeline at any stage.
- Pipelines can be programmatically filtered, grouped, and compared using utilities like:
  - :func:`experiment.get_ppl_status`
  - :func:`experiment.filter_ppls`
  - :func:`experiment.group_by_common_columns`


------------------------------
4. Hierarchical View
------------------------------

+-------------+-----------------------------------+-------------------------------------------+
| Level       | Represents                        | Purpose                                   |
+=============+===================================+===========================================+
| Component   | A single reusable operation       | Define atomic behavior (e.g., load,       |
|             |                                   | preprocess, train, evaluate).             |
+-------------+-----------------------------------+-------------------------------------------+
| Workflow    | A structured composition of       | Define *how* components connect.          |
|             | components                        | Manage process logic and dependencies.    |
+-------------+-----------------------------------+-------------------------------------------+
| Pipeline    | A concrete, executable instance   | Execute and track a specific run.         |
|             | of a workflow                     | Store results and ensure reproducibility. |
+-------------+-----------------------------------+-------------------------------------------+

Together, these layers create a **flexible, declarative, and traceable experimental system**
that supports both **research iteration** and **production reproducibility**.


------------------------------
5. Design Philosophy
------------------------------

- **Modularity:** Each layer is independent and composable.
- **Transparency:** All configurations and runs are logged and queryable.
- **Reproducibility:** Every experiment can be reloaded, re-executed, or audited.
- **Portability:** Pipelines and workflows can be transferred between machines or environments.
- **Scalability:** Supports many experiments with shared or divergent configurations.

This design makes it easy to iterate on ideas quickly while preserving the integrity and
traceability of experimental data — crucial for scientific and ML workflows alike.
