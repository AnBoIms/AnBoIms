# TIDC(Text Inpainting Datasets Create)
Data generator for use in SRNet

## Environment example
Python  3.10.12  
Numpy   1.26.4  
matplotlib  3.8.0  
pillow  11.0.0  
scikit-image    0.24.0  

## Usage example
!python create.py -i xx.txt -o xx.txt -f xx -sf xx.ttf -c black -b xx -g 0

### Options
'-i', '--input': type=str, help='path to the input text file'  
'-o', '--output': type=str, help='path to the output text file'  
'-f', '--font': type=str, help='path to the font dir'  
'-sf', '--standardFont': type=str, help='path to the standard font file'  
'-c', '--color': type=str, help='path to the color list file'  
'-b', '--background': type=str, help='path to the background dir'   
'-r', '--result': type=str, default='./TID', help='path to the result dir'  
'-n', '--num': type=int, default=1000, help='number of samples'  
'-w', '--width', type=int, default=800, help='sample image width'  
'-h', '--height', type=int, default=600, help='sample image height'  
'-m', '--textMin', type=int, default=20, help='minimum text size'  
'-M', '--textMax', type=int, default=150, help='maximum text size'  
'-sn', '--startNum': type=int, default=1, help='start number of samples'  
'-g', '--gpu': type=str, default="cpu", help='if you want to use gpu, write down the number'

## Number of results
input_line * output_line * font * color * background * num * 7(files)