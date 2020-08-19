import argparse
import os.path as osp
import torch


def get_parser():
    parser = argparse.ArgumentParser(description="Keep only model in ckpt")
    parser.add_argument(
        "path",
        default="output/person/blendmask/R_50_1x/model_final.pth",
        help="path to model weights",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    ckpt = torch.load(args.path)
    print("load model: {}".format(args.path))
    model = ckpt["model"]
    base_name = osp.basename(args.path).split(".")[0]
    out_path = osp.join(osp.dirname(args.path), "{}_no_optim.pth".format(base_name))
    torch.save(model, out_path)
    print("save model: {}".format(out_path))
