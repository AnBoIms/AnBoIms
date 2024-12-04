import argparse
from TIDC import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='path to the input text file')
    parser.add_argument('-o', '--output', type=str, help='path to the output text file')
    parser.add_argument('-f', '--font', type=str, help='path to the font dir')
    parser.add_argument('-sf', '--standardFont', type=str, help='path to the standard font file')
    parser.add_argument('-c', '--color', type=str, help='path to the color list file')
    parser.add_argument('-b', '--background', type=str, help='path to the background dir')
    parser.add_argument('-r', '--result', type=str, default='./TID', help='path to the result dir')
    parser.add_argument('-n', '--num', type=int, default=10, help='number of samples')
    parser.add_argument('-w', '--width', type=int, default=800, help='sample image width')
    parser.add_argument('-h', '--height', type=int, default=600, help='sample image height')
    parser.add_argument('-m', '--textMin', type=int, default=20, help='minimum text size')
    parser.add_argument('-M', '--textMax', type=int, default=150, help='maximum text size')
    parser.add_argument('-sn', '--startNum', type=int, default=1, help='start number of samples')
    parser.add_argument('-g', '--gpu', type=str, default="cpu", help='if you want to use gpu, write down the number')
    args = parser.parse_args()
    textIpaintingDatasetsCreate(
        args.input,
        args.output,
        args.font,
        args.standardFont,
        args.color,
        args.background,
        args.result,
        args.num,
        args.width,
        args.height,
        args.textMin,
        args.textMax,
        args.startNum,
        args.gpu)

if __name__ == "__main__":
    main()