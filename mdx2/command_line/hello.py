"""
This program is useful as a template for writing command line scripts.
"""

import argparse

parser = argparse.ArgumentParser(
    description="Say hello",
)

parser.add_argument("--chunks", nargs=3, type=int, default=[20,100,100], help="chunking for compression (frames, y, x)")

def run(args=None):
    args = parser.parse_args(args)
    print("hello")
    print("chunks = ",tuple(args.chunks))

if __name__ == "__main__":
    run()
