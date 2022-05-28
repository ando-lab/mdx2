"""
Print the version number
"""

import argparse

from mdx2 import getVersionNumber

parser = argparse.ArgumentParser(
    description=__doc__,
)

def run(args=None):
    args = parser.parse_args(args)
    print("mdx2:",getVersionNumber())

if __name__ == "__main__":
    run()
