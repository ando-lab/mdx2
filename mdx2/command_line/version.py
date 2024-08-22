"""
Print the version number
"""

import argparse
import sys
import os

import mdx2

parser = argparse.ArgumentParser(
    description=__doc__,
)

def run(args=None):
    args = parser.parse_args(args)
    print("mdx2:",mdx2.getVersionNumber())
    print("Python {0.major}.{0.minor}.{0.micro}".format(sys.version_info))
    print(f"Installed in: {os.path.split(mdx2.__file__)[0]}")

if __name__ == "__main__":
    run()
