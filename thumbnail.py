import sys
import glob
from PIL import Image

# for test only
#sys.argv += ['images/dot-*.png', 2, 3]

# get arguments
pattern = sys.argv[1]
rows = int(sys.argv[2])
cols = int(sys.argv[3])

# get filenames
filenames = glob.glob(pattern)

# load images and resize to (100, 100)
images = [Image.open(name).resize((100, 100)) for name in filenames]

# create empty image to put thumbnails
new_image = Image.new('RGB', (cols*100, rows*100))

# put thumbnails
i = 0
for y in range(rows):
    if i >= len(images):
        break
    y *= 100
    for x in range(cols):
        x *= 100
        img = images[i]
        new_image.paste(img, (x, y, x+100, y+100))
        print('paste:', x, y)
        i += 1

# save it
new_image.save('output.jpg')