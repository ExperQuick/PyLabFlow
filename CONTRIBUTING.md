# Contributing to PyLabFlow 🧪

First off, thank you for considering contributing to PyLabFlow! We're thrilled you're here. This project is built by the community, for the community, and we welcome any contribution, from fixing typos to implementing major new features.

This document provides guidelines to help make the contribution process easy and effective for everyone involved.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](./CODE_OF_CONDUCT.md). Please read it to understand the standards we follow to ensure our community is welcoming and inclusive.

## How Can I Contribute?

There are many ways to contribute to PyLabFlow, and every contribution matters.

* **🐛 Report Bugs**
  Found a bug? Open an issue and include clear steps, logs, or screenshots if possible.

* **💡 Suggest Enhancements**
  Have ideas for new features or improvements? Start a discussion by opening an issue.

* **📝 Improve Documentation**
  Help make the docs clearer by fixing mistakes, adding explanations, or improving examples.

* **🧪 Write Test Cases**
  Strengthen the project by adding or improving tests to ensure reliability.

* **💻 Contribute Code**
  Fix bugs or implement new features by submitting a pull request.

Every contribution—big or small—helps improve the project 🚀

## Your First Contribution

Unsure where to begin? A great place to start is by looking for issues tagged `good first issue` or `help wanted`. These are typically well-defined and a great way to get familiar with the codebase.

## 🚀 Submitting a Pull Request (PR)

Ready to contribute code or documentation? Here’s how to set up your environment and submit a pull request.

#### 1. Fork the Repository
Click the "Fork" button at the top right of the [PyLabFlow GitHub page](https://github.com/ExperQuick/PyLabFlow) to create your own copy.

#### 2. Clone Your Fork
Clone your forked repository to your local machine.

```bash
git clone https://github.com/YOUR_USERNAME/PyLabFlow.git

cd PyLabFlow
```

#### 3\. Create a New Branch

Create a descriptive branch name for your changes. This keeps your work separate from the `main` branch.

```bash
# Example for a new feature
git checkout -b feature/add-new-visualization-plot

# Example for a bug fix
git checkout -b fix/resolve-config-save-error
```

#### 4\. Set Up Your Development Environment

We recommend using a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (on Windows)
# venv\Scripts\activate
# Activate it (on macOS/Linux)
source venv/bin/activate

# Install dependencies, including testing tools
pip install -r requirements.txt
pip install pytest
```

#### 5\. Make Your Changes

Now, write your code\! Make your changes to the codebase, and remember to follow our style guidelines.

#### 6\. Run Tests

Before submitting, make sure your changes haven't broken anything. Run the full test suite from the root directory.

```bash
pytest
```

If you've added new functionality, please add new tests to cover it\!

#### 7\. Commit Your Changes

Commit your changes with a clear and descriptive message. We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

```bash
# Example commit message
git commit -m "feat: Add performance_plot for multi-pipeline comparison"
```

#### 8\. Push to Your Fork

Push your new branch to your forked repository on GitHub.

```bash
git push origin feature/add-new-visualization-plot
```

#### 9\. Open a Pull Request

Go to your forked repository on GitHub. You will see a prompt to create a pull request from your new branch. Click it, and fill out the pull request template with details about your changes.

  - **Link to the issue** if your PR addresses one.
  - **Describe your changes** clearly.
  - **Explain the "why"** behind your changes.

Once you submit the PR, a project maintainer will review your code. We may suggest some changes or improvements. Thank you for your contribution\!

## Style Guides

### Git Commit Messages

  - Use the present tense ("Add feature" not "Added feature").
  - Use the imperative mood ("Move file to..." not "Moves file to...").
  - Limit the first line to 72 characters or less.
  - Reference issues and pull requests liberally in the body of the commit message.

### Python Code

  - Please follow the **PEP 8** style guide for Python code.
  - We recommend using a code formatter like `black` to automatically format your code.

Thank you again for your interest in making PyLabFlow better\! ❤️
