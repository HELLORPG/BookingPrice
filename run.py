import argparse


def run_args_parser() -> argparse.ArgumentParser:
    """
    return an arguments' parser for run.py.
    :return: an ArgumentParser for run.py.
    """
    args_parser = argparse.ArgumentParser(description="Set how to run python file run.py.", add_help=False)

    args_parser.add_argument("--part", type=str)

    return args_parser


def main(args):
    print(args.part)
    return


if __name__ == '__main__':
    run_args = argparse.ArgumentParser(
        description="Set how to run python file run.py.",
        add_help=True,
        parents=[run_args_parser()]
    ).parse_args()
    main(run_args)
