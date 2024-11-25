import argparse
from TIDC import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='path to the input text file')
    parser.add_argument('-o', '--output', type=str, help='path to the output text file')
    parser.add_argument('-f', '--font', type=str, help='path to the font file')
    parser.add_argument('-sf', '--standardFont', type=str, help='path to the standard font file')
    parser.add_argument('-c', '--color', type=str, help='choose a color in str format')
    parser.add_argument('-b', '--background', type=str, help='path to the background file')
    parser.add_argument('-or', '--orientation', type=str, default='horizontal', help='choose orientation horizontal or vertical')
    parser.add_argument('-r', '--result', type=str, default='./TID', help='path to the result dir')
    parser.add_argument('-n', '--num', type=int, default=1000, help='number of samples')
    parser.add_argument('-s', '--size', type=tuple, default=(800,400), help='sample image size')
    parser.add_argument('-sn', '--startNum', type=int, default=1, help='start number of samples')
    args = parser.parse_args()
    textIpaintingDatasetsCreate(
        args.input,
        args.output,
        args.font,
        args.standardFont,
        args.color,
        args.background,
        args.orientation,
        args.result,
        args.num,
        args.size,
        args.startNum)

if __name__ == "__main__":
    main()