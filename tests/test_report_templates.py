"""Verify that the mdx2.report CLI correctly passes parameters to notebook templates"""

import dataclasses
import re
import typing
import warnings

import nbformat
import pytest
from papermill import inspect_notebook

from mdx2.command_line.report import (
    ScalingModelParameters,
    VisualizationParameters,
    _ExampleParameters,
)
from mdx2.report import TEMPLATES


class TemplateParameterMismatchWarning(UserWarning):
    pass


SUPPORTED_TYPES = []
for t in ["int", "float", "str", "bool"]:
    SUPPORTED_TYPES.append(t)
    SUPPORTED_TYPES.append(f"list[{t}]")


def _canonical_type_name(type_hint):
    if hasattr(type_hint, "__origin__") and type_hint.__origin__ is typing.Union:
        non_none_types = [t for t in type_hint.__args__ if t is not type(None)]
        if len(non_none_types) == 1:
            type_hint = non_none_types[0]

    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)

    if origin is list and len(args) == 1:
        return f"list[{_canonical_type_name(args[0])}]"

    if type_hint in (int, float, str, bool):
        return type_hint.__name__

    return str(type_hint).replace("typing.", "")


@pytest.mark.parametrize("template_name, template_path", TEMPLATES.items())
def test_template_exists(template_name, template_path):
    """Verify that the templates defined in mdx2.report.TEMPLATES actually exist"""
    assert template_path.is_file(), f"Template '{template_name}' not found at path: {template_path}"


@pytest.mark.parametrize("template_name, template_path", TEMPLATES.items())
def test_template_formatting(template_name, template_path):
    """Verify that the templates defined in mdx2.report.TEMPLATES are valid Jupyter notebook templates"""
    # first, try opening with nbformat
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        pytest.fail(f"Template '{template_name}' at path {template_path} is not a valid Jupyter notebook: {e}")

    # next, make sure there are no evaluated cells.
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [], (
                f"Template '{template_name}' at path {template_path} contains evaluated cells. "
                "Please clear all outputs before using as a template."
            )

    # next, make sure that one (and only one) cell is tagged with "parameters".
    parameter_cells = [cell for cell in nb.cells if "parameters" in cell.metadata.get("tags", [])]
    assert len(parameter_cells) == 1, (
        f"Template '{template_name}' at path {template_path} should contain exactly one cell tagged with 'parameters', "
        f"but found {len(parameter_cells)}"
    )

    # check that the parameter_cells contains a parameter called "_metadata", initialized to an empty dict.
    parameter_cell = parameter_cells[0]
    # use a regex to check that the cell contains a line like "_metadata: dict = {}" (allowing for whitespace).
    # The type hint is optional. There should not be any whitespace before _metadata (at start of line)
    pattern = r"^_metadata\s*(?::\s*dict)?\s*=\s*{}"
    assert re.search(pattern, parameter_cell.source, re.MULTILINE), (
        f"Template '{template_name}' at path {template_path} should contain a parameter '_metadata' "
        f"initialized to an empty dict in the 'parameters' cell."
    )


@pytest.mark.parametrize("template_name, template_path", TEMPLATES.items())
def test_template_parameters_parseable(template_name, template_path):
    """Verify that the parameters defined in the notebook template can be parsed by papermill.inspect_notebook"""
    try:
        _ = inspect_notebook(template_path)
    except Exception as e:
        pytest.fail(f"Template '{template_name}' at path {template_path} has unparseable parameters: {e}")


@pytest.mark.parametrize("template_name, template_path", TEMPLATES.items())
def test_template_parameters_valid(template_name, template_path):
    """Verify that the parameters defined in the notebook template are valid (i.e. have supported types)"""
    params = inspect_notebook(template_path)
    for param in params.values():
        if param["name"].startswith("_"):
            continue  # _metadata is a special case, it can be any dict. Also allow private parameters here.
        assert param["inferred_type_name"] in SUPPORTED_TYPES, (
            f"Template '{template_name}' at path {template_path} "
            f"parameter '{param['name']}' has unsupported type '{param['inferred_type_name']}'. "
            f"Supported types are: {SUPPORTED_TYPES}"
        )


@pytest.mark.parametrize(
    "ParametersDataclass",
    [
        _ExampleParameters,
        ScalingModelParameters,
        VisualizationParameters,
    ],
)
def test_dataclass_matches_template(ParametersDataclass):
    # Verify that each ExecutableNotebook dataclass has a matching template in mdx2.report.TEMPLATES
    template_name = ParametersDataclass._template_name
    assert template_name in TEMPLATES, (
        f"ExecutableNotebook dataclass '{ParametersDataclass.__name__}' specifies template '{template_name}' "
        f"which does not exist in mdx2.report.TEMPLATES. Available templates: {list(TEMPLATES.keys())}"
    )
    # Verify that all of the fields in the dataclass are present in the parameters cell of the notebook template.
    template_path = TEMPLATES[template_name]
    params = inspect_notebook(template_path)
    for field in ParametersDataclass.__dataclass_fields__.values():
        if field.name.startswith("_"):
            continue  # skip private fields, such as _template_name
        assert field.name in params, (
            f"Field '{field.name}' in dataclass '{ParametersDataclass.__name__}' is not present in the parameters "
            f"cell of template '{template_name}' at path {template_path}"
        )
        # Verify that, for parameters with a default value of None in the template, the corresponding dataclass
        # specifies a required parameter or a parameter with a default value that is not None.
        template_param = params.get(field.name)
        if template_param["default"] == "None":
            is_required_in_dataclass = (
                field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
            )
            has_default_in_dataclass = not is_required_in_dataclass and field.default is not None
            # NOTE: do I need to check for the existence of a default factory?
            if not (is_required_in_dataclass or has_default_in_dataclass):
                warnings.warn(
                    (
                        f"Field '{field.name}' in dataclass '{ParametersDataclass.__name__}' corresponds to a "
                        f"parameter with default value None in template '{template_name}', but is not defined as "
                        f"a required parameter or a parameter with a default value that is not None in the "
                        f"dataclass."
                    ),
                    TemplateParameterMismatchWarning,
                    stacklevel=2,
                )
        # Verify that papermill's inferred data type matches the type hint in the parameters dataclass
        # HACK: Need to handle cases like Optional[list[str]] where the inferred type is "list[str]"
        # but the type hint is "typing.Optional[list[str]]"

        inferred_type = template_param["inferred_type_name"]
        expected_type = _canonical_type_name(field.type)

        assert inferred_type == expected_type, (
            f"Field '{field.name}' in dataclass '{ParametersDataclass.__name__}' has type hint '{field.type}', but "
            f"papermill inferred type '{inferred_type}' for the corresponding parameter in "
            f"template '{template_name}'. These should match to ensure that the CLI correctly parses parameters."
        )
