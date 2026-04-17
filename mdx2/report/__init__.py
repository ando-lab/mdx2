"""report generation utilities"""

import glob
import os
import pathlib
import platform
import subprocess
from datetime import datetime

import papermill as pm

import mdx2  # for version info

# a glob string matching all notebook templates
_templates_glob = os.path.join(os.path.dirname(__file__), "templates", "*.ipynb")

# need to search TEMPLATES_DIR for all .ipynb files and create a dict mapping template names (without .ipynb)
TEMPLATES = {os.path.splitext(os.path.basename(path))[0]: pathlib.Path(path) for path in glob.glob(_templates_glob)}

# TODO: allow users to override certain metadata fields, such as author.


def _get_default_metadata():
    """assign defaults to notebook metadata fields, return as a dict"""
    # First, try to get author using various methods
    # 1. if on linux, use the "getent" command to get the full name of the user
    # 2. if on mac, use the "id -F" command to get the full name of the user
    # 3. if the above fail, use the USER environment variable
    author = None
    try:
        if platform.system() == "Linux":
            author = subprocess.check_output(["getent", "passwd", os.getlogin()]).decode().split(":")[4].split(",")[0]
        elif platform.system() == "Darwin":
            author = subprocess.check_output(["id", "-F"]).decode().strip()
    except Exception:
        pass
    if not author:
        author = os.environ.get("USER", None)

    # Next, get the current date and time using python's datetime module,
    # formatted in a human-readable way (e.g. "2024-06-01 12:00:00")
    date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # get the mdx2 version using mdx2.__version__
    version = mdx2.__version__

    # now, get the environment (i,e. hostname)
    environment = platform.node()

    # get the current working directory
    working_directory = os.getcwd()

    return {
        "author": author,
        "date_created": date_created,
        "mdx2_version": version,
        "environment": environment,
        "working_directory": working_directory,
    }


def execute_notebook(template_name, output_path=None, parameters={}, metadata={}, **kwargs):
    """execute a notebook template and save the result to output_path

    Parameters
    ----------
    template_name : str
        name of the notebook template to execute. This should be the name of a .ipynb file in mdx2.report.templates
    output_path : str, optional
        path to save the executed notebook to. If None, the template name will be used (with .ipynb extension)
    parameters : dict, optional
        dictionary of parameters to pass to the notebook template.
    metadata : dict, optional
        dictionary of metadata fields, will override any default metadata fields populated by mdx2.report.
    **kwargs
        additional keyword arguments to pass to papermill.execute_notebook.

    See https://papermill.readthedocs.io/en/latest/api.html#papermill.execute_notebook for more details

    Returns
    -------
    None
    """

    # get the path to the notebook template
    template_path = TEMPLATES.get(template_name)
    if template_path is None:
        raise ValueError(f"Template '{template_name}' not found. Available templates: {list(TEMPLATES.keys())}")

    if output_path is None:
        output_path = f"{template_name}.ipynb"

    _metadata = _get_default_metadata()
    _metadata["notebook_template"] = template_name

    # allow user to override metadata using the metadata argument
    # disregard any keys with a value of None.
    for key, value in metadata.items():
        if value is not None:
            _metadata[key] = value

    injected_parameters = {}
    # add all parameters to injected_parameters, except those with a value of None
    # these are treated as optional parameters (do not override the defaults defined in the notebook)
    for key, value in parameters.items():
        if value is not None:
            injected_parameters[key] = value

    # inject metadata into parameters
    injected_parameters["_metadata"] = _metadata

    pm.execute_notebook(input_path=template_path, output_path=output_path, parameters=injected_parameters, **kwargs)
