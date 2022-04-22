"""
This program is useful as a template for writing command line scripts.
"""

import argparse

parser = argparse.ArgumentParser(
    description="Say hello",
)

def run(args=None):
    args = parser.parse_args(args)
    print("hello")

if __name__ == "__main__":
    run()
