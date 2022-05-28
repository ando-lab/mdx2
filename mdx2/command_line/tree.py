"""
View the NeXus file tree
"""

import argparse

from nexusformat.nexus import nxload

parser = argparse.ArgumentParser(
    description=__doc__,
)

parser.add_argument("filename", help="NeXus file name")

def run(args=None):
    args = parser.parse_args(args)
    nxs = nxload(args.filename,'r')
    print(f"{args.filename}:",nxs.tree)

if __name__ == "__main__":
    run()
