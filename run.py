import argparse

from data.location import location_scatter_from_booking_info_file


def run_args_parser() -> argparse.ArgumentParser:
    """
    return an arguments' parser for run.py.
    :return: an ArgumentParser for run.py.
    """
    args_parser = argparse.ArgumentParser(description="Set how to run python file run.py.", add_help=False)

    # 确定需要运行的模块名称
    args_parser.add_argument("--part", type=str)

    # 输入路径
    args_parser.add_argument("--input-path", type=str)

    # 输出路径
    args_parser.add_argument("--output-path", type=str)

    return args_parser


def main(args):
    """
    run.py 主要运行的文件
    :param args: 传入的运行参数
    :return:
    """
    if args.part == "location_visualization":
        input_path = args.input_path
        save_path = args.output_path
        location_scatter_from_booking_info_file(input_path=input_path, save_path=save_path)
    else:
        print("run.py do not support part: %s." % args.part)
        exit(-1)

    return


if __name__ == '__main__':
    run_args = argparse.ArgumentParser(
        description="Set how to run python file run.py.",
        add_help=True,
        parents=[run_args_parser()]
    ).parse_args()
    main(run_args)
