"""Command-line Interface to interact with Gelquant."""

import argparse


def cli():
    """Command-line interface main method."""
    parser = argparse.ArgumentParser()
    parser.add_argument('test')

    args = parser.parse_args()
    print(args.test)


if __name__ == "__main__":
    cli()
