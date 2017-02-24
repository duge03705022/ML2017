from PIL import Image
from PIL import ImageChops
import sys

filenameA = sys.argv[1];
filenameB = sys.argv[2];

ImageA = Image.open(filenameA);
ImageB = Image.open(filenameB);

width, height=ImageA.size

DataA = ImageA.getdata();
DataB = ImageB.getdata();

ans = Image.new("RGBA",ImageA.size,color = (0,0,0,0));

for i in range(width):
	for j in range(height):
		if DataA[i,j] != DataB[i,j]:
			ans.putpixel((i,j), DataB[i,j]);

ans.save("ans_two.png");