User's workflow doing experiments
---------------------------------

Before you start designing workflows and pipelines, you need a **project environment**, often called a “lab.” The lab organizes all your experiments, components, and logs in a reproducible way.

Creating a Project
~~~~~~~~~~~~~~~~~~

You can create a project using the `create_project` function:

.. code-block:: python

    from plf.lab import create_project

    settings = {
        "project_name": "my_experiment",
        "project_dir": "./projects",
        "component_dir": "path/to/component/dir/"
    }

    settings_path = create_project(settings)

**What `create_project` does:**

1. **Directory Setup:**  
   Creates the main project directory and subdirectories for components, data, logs, workflows, and pipelines.

2. **Settings Management:**  
   Generates a JSON file storing all paths and configuration parameters, which ensures reproducibility.

3. **Database Initialization:**  
   Sets up the following databases under the project’s data directory:
   
   - `logs.db`: Tracks execution logs.
   - `ppls.db`: Tracks pipelines, their status, and dependencies.
   - `Archived/ppls.db`: Stores completed or archived pipelines.

4. **Shared Data Setup:**  
   Registers paths and project metadata globally so all parts of the framework can access them consistently.

**Result:** You now have a fully prepared workspace where you can safely develop, run, and track experiments.

Accessing an Existing Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a project already exists, you can **load it** using `lab_setup`:

.. code-block:: python

    from plf.lab import lab_setup
    lab_setup(settings_path)

**What `lab_setup` does:**

- Loads the project settings JSON file.
- Sets up shared data so that all utilities, components, and workflows can access project paths.
- Registers the component directories so you can import and use them dynamically.
- Creates a new **log entry** in `logs.db` to track that this project was accessed.

**Why this matters:**

- Every session of your lab is tracked.
- Components and workflows can safely access the correct directories.
- Your experiment history is auditable and reproducible.

