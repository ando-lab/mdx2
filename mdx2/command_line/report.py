"""
Generate jupyter notebook reports.
"""

from dataclasses import asdict, dataclass, field as dataclass_field
from typing import ClassVar, Optional

# from simple_parsing.helpers.fields import subparsers
from simple_parsing import field, subgroups

from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.data import HKLTable
from mdx2.geometry import Crystal, Symmetry
from mdx2.io import NexusFileIndex
from mdx2.report import execute_notebook
from mdx2.scaling import AbsorptionModel, DetectorModel, OffsetModel, ScalingModel


@dataclass
class Metadata:
    """metadata fields that can be overriden on the CLI (see mdx2.report._get_default_metadata)

    All fields are optional, and if not provided, will be populated by mdx2.report with default values.
    """

    author: Optional[str] = None
    date_created: Optional[str] = None
    mdx2_version: Optional[str] = None
    environment: Optional[str] = None
    working_directory: Optional[str] = None


@dataclass
class ExecutableNotebook:
    """base class for executable notebooks."""

    _template_name: ClassVar[str]
    # Mapping of source labels to auto-discovery rules, defined by subclasses.
    _input_source_mapping: ClassVar[dict]

    def update_sources_from_input_files(self, *input_files):
        file_index = NexusFileIndex(*input_files)
        for source_name, source_info in self._input_source_mapping.items():
            if getattr(self, source_info["parameter_name"]) is not None:
                continue  # skip if the source is already set by the user
            object_matches = file_index.find_objects(source_info["object_type"])
            if object_matches:
                if source_info["multiple"]:
                    setattr(self, source_info["parameter_name"], [f"{m[0]}:{m[1]}" for m in object_matches])
                elif len(object_matches) > 1:
                    raise ValueError(
                        f"Multiple {source_name} objects found in input files: "
                        f"{object_matches}. Please specify the source explicitly using the "
                        f"--{source_info['parameter_name']} parameter."
                    )
                else:
                    setattr(self, source_info["parameter_name"], f"{object_matches[0][0]}:{object_matches[0][1]}")
            elif source_info["required"]:
                raise ValueError(
                    f"No {source_name} object found in input files: {input_files}. "
                    f"Please provide a {source_name} object in the input files or specify "
                    f"the source explicitly using the --{source_info['parameter_name']} parameter."
                )

    def execute(self, **kwargs):
        """execute the report generation using the provided parameters"""
        execute_notebook(
            template_name=self._template_name,
            parameters=asdict(self),
            **kwargs,
        )


@dataclass
class _ExampleParameters(ExecutableNotebook):
    """dummy template for testing purposes"""

    _template_name = "_example"
    _input_source_mapping = {
        "scaling_model": {
            "object_type": ScalingModel,
            "parameter_name": "scaling_model_sources",
            "multiple": True,
            "required": False,
        },
    }

    scaling_model_sources: Optional[list[str]] = (
        None  # list of ScalingModel object source paths encoded as a string nexus_file_path:object_name
    )
    pi: Optional[float] = None  # optional numerical value of pi, overriding default defined in _template.ipynb


@dataclass
class VisualizationParameters(ExecutableNotebook):
    """parameters for the visualization.ipynb template"""

    _template_name = "visualization"
    _input_source_mapping = {
        "crystal": {
            "object_type": Crystal,
            "parameter_name": "crystal_source",
            "multiple": False,
            "required": True,
        },
        "symmetry": {
            "object_type": Symmetry,
            "parameter_name": "symmetry_source",
            "multiple": False,
            "required": True,
        },
        "hkl_table": {
            "object_type": HKLTable,
            "parameter_name": "hkl_table_source",
            "multiple": False,
            "required": True,
        },
    }

    crystal_source: Optional[str] = (
        None  # optional path encoded as a string nexus_file_path:object_name to load a crystal object
    )
    symmetry_source: Optional[str] = (
        None  # optional path encoded as a string nexus_file_path:object_name to load a crystal object
    )
    hkl_table_source: Optional[str] = (
        None  # optional path encoded as a string nexus_file_path:object_name to load a hkl table object
    )

    cartesian_coordinates: bool = (
        True  # whether to plot the slices in cartesian coordinates (sx, sy) or Miller indices (h, k, l).
    )


@dataclass
class MapStatisticsParameters(ExecutableNotebook):
    """parameters for the map_statistics.ipynb template"""

    _template_name = "map_statistics"
    _input_source_mapping = {
        "crystal": {
            "object_type": Crystal,
            "parameter_name": "crystal_source",
            "multiple": False,
            "required": True,
        },
        "symmetry": {
            "object_type": Symmetry,
            "parameter_name": "symmetry_source",
            "multiple": False,
            "required": True,
        },
        "hkl_table": {
            "object_type": HKLTable,
            "parameter_name": "hkl_table_source",
            "multiple": False,
            "required": True,
        },
    }

    crystal_source: Optional[str] = (
        None  # optional path encoded as a string nexus_file_path:object_name to load a crystal object
    )
    symmetry_source: Optional[str] = (
        None  # optional path encoded as a string nexus_file_path:object_name to load a crystal object
    )
    hkl_table_source: Optional[str] = (
        None  # optional path encoded as a string nexus_file_path:object_name to load a hkl table object
    )

    bin_width: float = 0.01  # bin width for statistics vs s


@dataclass
class ScalingModelParameters(ExecutableNotebook):
    """parameters for the scaling_model.ipynb template"""

    _template_name = "scaling_model"
    _input_source_mapping = {
        "scaling_model": {
            "object_type": ScalingModel,
            "parameter_name": "scaling_model_sources",
            "multiple": True,
            "required": False,
        },
        "offset_model": {
            "object_type": OffsetModel,
            "parameter_name": "offset_model_sources",
            "multiple": True,
            "required": False,
        },
        "absorption_model": {
            "object_type": AbsorptionModel,
            "parameter_name": "absorption_model_sources",
            "multiple": True,
            "required": False,
        },
        "detector_model": {
            "object_type": DetectorModel,
            "parameter_name": "detector_model_sources",
            "multiple": True,
            "required": False,
        },
    }

    scaling_model_sources: Optional[list[str]] = (
        None  # list of ScalingModel object source paths encoded as a string nexus_file_path:object_name
    )
    offset_model_sources: Optional[list[str]] = (
        None  # list of OffsetModel object source paths encoded as a string nexus_file_path:object_name
    )
    absorption_model_sources: Optional[list[str]] = (
        None  # list of AbsorptionModel object source paths encoded as a string nexus_file_path:object_name
    )
    detector_model_sources: Optional[list[str]] = (
        None  # list of DetectorModel object source paths encoded as a string nexus_file_path:object_name
    )


@dataclass
class Parameters:
    """parameters for the report generation"""

    # Input nexus files. Will be inspected for object sources, and passed to the template
    input_files: list[str] = field(positional=True, nargs="*", default_factory=list)

    template: ExecutableNotebook = subgroups(
        {
            p._template_name: p
            for p in [_ExampleParameters, ScalingModelParameters, VisualizationParameters, MapStatisticsParameters]
        },
        default="visualization",
    )

    metadata: Metadata = dataclass_field(default_factory=Metadata)  # metadata fields that can be overridden on the CLI

    # optional path to save the executed notebook to. If None, the template name us used with .ipynb extension
    output_path: Optional[str] = field(alias=["-o"], default=None)

    def __post_init__(self):
        # if the user provided input files at the top level, pass these to the template
        # to update sources that were not explicitly set by the user.
        if self.input_files:
            self.template.update_sources_from_input_files(*self.input_files)


def run_report(params):
    """main function to run the report generation"""
    params.template.execute(
        metadata=asdict(params.metadata),
        output_path=params.output_path,
    )


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_report))


if __name__ == "__main__":
    run()
