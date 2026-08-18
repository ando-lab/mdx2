"""Common utilities for command-line tools"""

import sys
import time
from datetime import datetime
from functools import wraps

from joblib import effective_n_jobs
from loguru import logger
from simple_parsing import ArgumentGenerationMode, ArgumentParser, NestedMode


def log_parallel_backend(parallel):
    """
    Log information about the active joblib Parallel backend.

    Attempts to access backend information via private API. If this fails
    (e.g., due to API changes), logs a warning but does not raise an error.

    Parameters
    ----------
    parallel : joblib.Parallel
        The Parallel instance to inspect
    """
    try:
        if hasattr(parallel, "_backend") and parallel._backend:
            backend_name = parallel._backend.__class__.__name__
        else:
            backend_name = "Unknown"
        n_jobs = effective_n_jobs(parallel.n_jobs)  # handle -1, -N cases
        logger.info("Using backend: {}, n_jobs: {}", backend_name, n_jobs)
    except Exception as e:
        logger.warning("Could not determine joblib backend details: {}", e)


def make_argument_parser(params_class, description):
    """
    Create a standardized argument parser for a Parameters class.

    This factory function generates a parse_arguments() function that can be used
    in command-line modules. It uses a consistent ArgumentParser configuration
    that supports both flat and nested dataclass parameters.

    Args:
        params_class: The Parameters dataclass for this command-line tool
        description: Module docstring (__doc__)

    Returns:
        A parse_arguments(args=None) function that parses command-line arguments
        and returns a Parameters instance

    Example:
        >>> @dataclass
        >>> class Parameters:
        >>>     input: str = field(positional=True)
        >>>     count: int = 10
        >>>
        >>> parse_arguments = make_argument_parser(Parameters, __doc__)
        >>> params = parse_arguments(["input.txt", "--count", "5"])
    """

    def parse_arguments(args=None):
        """Parse commandline arguments"""
        parser = ArgumentParser(
            description=description,
            argument_generation_mode=ArgumentGenerationMode.NESTED,
            nested_mode=NestedMode.WITHOUT_ROOT,
        )
        parser.add_arguments(params_class, dest="parameters")
        opts = parser.parse_args(args)
        return opts.parameters

    return parse_arguments


def with_parsing(parse_arguments):
    """
    Decorator to parse command-line arguments before calling the function.

    Args:
        parse_arguments: A function that takes (args=None) and returns a Parameters object.

    The decorated function should accept a single argument: the parsed Parameters object.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(args=None):
            params = parse_arguments(args)
            return func(params)

        return wrapper

    return decorator


def with_logging(log_level="INFO"):
    """
    Decorator to set up logging for command-line tools.

    Args:
        log_level: Logging level (default: INFO)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(params):
            # Remove default handler
            logger.remove()

            module_name = func.__module__.split(".")[-1]
            logfile = params.logfile

            # File format: detailed with full timestamp, NO color tags for plain text
            file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"

            # Stderr format: streamlined with fixed-width time and level, WITH colors
            stderr_format = "<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> | {message}"

            # Add handlers with different formats
            logger.add(logfile, level=log_level, format=file_format, colorize=False)
            logger.add(sys.stderr, level=log_level, format=stderr_format, colorize=True)

            # Log start with full timestamp and module name
            start_time = time.time()
            start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Starting mdx2.{module_name} at {start_datetime}")
            logger.info(params)

            try:
                result = func(params)
                elapsed = time.time() - start_time
                logger.success(f"mdx2.{module_name} completed in {elapsed:.2f} seconds")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"mdx2.{module_name} failed after {elapsed:.2f} seconds: {e}")
                raise

        return wrapper

    return decorator
