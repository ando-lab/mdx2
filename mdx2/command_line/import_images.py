"""
Import images using the dxtbx machinery
"""

import argparse

def parse_arguments():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Required arguments
    parser.add_argument("expt", help="expt file containing scan metadata (e.g. from dials.import)")

    return parser

def run(args=None):
    parser = parse_arguments()
    args = parser.parse_args(args)
    print(f"importing images from {args.expt}")

if __name__ == "__main__":
    run()
