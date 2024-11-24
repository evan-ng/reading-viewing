import os, json, numpy as np
from PIL import Image
import math

data = []

# Access png assets
os.chdir("assets\\letters")
all_files=os.listdir(os.getcwd())
img_list=[filename for filename in all_files if filename[-4:] == ".png" and len(filename) == 5]

min_avg = 255
max_avg = 0

for img in img_list:
    print(img + "\n")
    img_file = Image.open(img)
    width, height = img_file.size
    img_arr = np.array(img_file)
    half_width = math.floor(width / 2)
    half_height = math.floor(height / 2)
    
    rects = []
    rects.append(img_arr[0:half_width, 0:half_height])
    rects.append(img_arr[half_width:width, 0:half_height])
    rects.append(img_arr[0:half_width, half_height:height])
    rects.append(img_arr[half_width:width, half_height:height])
    
    entry = [img]
    for r in rects:
        avg = np.mean(r)
        entry.append(avg)
        if (avg > max_avg):
            max_avg = avg
        if (avg < min_avg):
            min_avg = avg
    data.append(entry)

# normalize
diff_avg = max_avg - min_avg
for entry in data:
    for i in range(1,5):
        entry[i] = 255 * (entry[i] - min_avg) / diff_avg

os.chdir("..")
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump({"letters": data}, f, ensure_ascii=False, indent=4)