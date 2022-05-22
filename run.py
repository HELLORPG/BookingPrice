import argparse

from data.location import location_scatter_from_booking_info_file
from data.dataset import split_train_val
from engine import train


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

    # 对于split_train_val独有的参数
    args_parser.add_argument("--train-path", type=str, default="./dataset/split/train.csv")     # 和train过程中的train path共用
    args_parser.add_argument("--val-path", type=str, default="./dataset/split/val.csv")         # 和train过程中的val path共用
    args_parser.add_argument("--train-ratio", type=float)
    args_parser.add_argument("--train-size", type=int)
    args_parser.add_argument("--val-size", type=int)

    # 对于训练和测试可能使用的参数 ===================================================================================
    # 设备
    args_parser.add_argument("--device", type=str, default="cpu")

    # 输出文件
    args_parser.add_argument("--data-meta-dir", type=str, default="./outputs/meta")

    # 是否增加验证输出
    args_parser.add_argument("--with-val", type=str, default="True", help="Train with eval val.")
    # args_parser.add_argument("--with-test", type=str, default="True", help="Test at final.")

    # 超参数
    args_parser.add_argument("--batch-size", type=int, default=4)
    args_parser.add_argument("--epoch", type=int, default=100)

    # 优化器
    args_parser.add_argument("--optimizer", type=str, default="Adam")
    args_parser.add_argument("--lr", type=float, default=1e-4)
    args_parser.add_argument("--weight-decay", type=float, default=1e-5)

    # 损失函数
    args_parser.add_argument("--loss-function", type=str, default="CEL")

    # 模型结构
    args_parser.add_argument("--activation", type=str, default="relu")
    args_parser.add_argument("--position-mlp-layers", default=[80, 20], type=int, nargs="+")
    args_parser.add_argument("--room-mlp-layers", default=[20, 20], type=int, nargs="+")
    args_parser.add_argument("--addition-mlp-layers", default=[20, 20], type=int, nargs="+")
    args_parser.add_argument("--review-mlp-layers", default=[20, 20], type=int, nargs="+")
    args_parser.add_argument("--final-mlp-layers", default=[200, 100, 6], type=int, nargs="+")

    return args_parser


def main(args):
    """
    run.py 主要运行的文件
    :param args: 传入的运行参数
    :return:
    """
    if args.part == "location-visualization":
        input_path = args.input_path
        save_path = args.output_path
        location_scatter_from_booking_info_file(input_path=input_path, save_path=save_path)
    elif args.part == "split-train-val":
        whole_train_path = args.input_path
        train_path = args.train_path
        val_path = args.val_path
        if args.train_ratio is not None:
            split_train_val(
                input_path=whole_train_path,
                train_path=train_path,
                val_path=val_path,
                train_ratio=args.train_ratio
            )
        elif args.train_size is not None:
            split_train_val(
                input_path=whole_train_path,
                train_path=train_path,
                val_path=val_path,
                train_size=args.train_size
            )
        elif args.val_size is not None:
            split_train_val(
                input_path=whole_train_path,
                train_path=train_path,
                val_path=val_path,
                val_size=args.val_size
            )
        else:
            print("Do Not Know how to split dataset.")
            exit(-1)
    elif args.part == "train":
        train(args=args)
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
