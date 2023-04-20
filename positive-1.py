import os
import cv2

# Set parameters
w = 0.07
h = 0.07
x = 0.5
y = 0.17
path = "/Users/biggles/Documents/Code/pos/"


# Open file for writing
with open('info.txt', 'w') as f:
    # Loop over positive images
    for i in range(20):
        d= path +"img22"+str(i+1)+".png"+" 1 238.08 69.12 0.07 0.07"
        line = f"{d}\n"
        f.write(line)
    
    for i in range(20):
        d= path +"img22n"+str(i+1)+".png"+" 1 238.08 69.12 0.07 0.07"
        line = f"{d}\n"
        f.write(line)

    for i in range(20):
        d= path +"imgz"+str(i+1)+".png"+" 1 238.08 69.12 0.07 0.07"
        line = f"{d}\n"
        f.write(line)

    '''for filename in os.listdir(path):
        # Get image dimensions
        img = cv2.imread(os.path.join(path, filename))
        height, width, channels = img.shape
        # Calculate object coordinates
        x_obj = x * width - w * width / 2
        y_obj = y * height - h * height / 2
        # Write line to file
        line = f"{os.path.join(path, filename)} 1 {x_obj:.2f} {y_obj:.2f} {w:.2f} {h:.2f}\n"
        f.write(line)'''
