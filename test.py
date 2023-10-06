#!/bin/python
from model.resnet import build_model
from PIL import Image
import numpy as np
import argparse
import glob, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", default="input", type=str, help="Image path.")
parser.add_argument("-o", "--output", default="output", type=str, help="Output Image path.")
parser.add_argument("-m", "--model", default="weights.h5", type=str, help="Model path.")
args = parser.parse_args()

model = build_model(filters=64,block=16)
model.load_weights(args.model)



if ( "*" in  args.input) :
    files = glob.glob(args.input)
else:
    files = glob.glob(f"{args.input}/*.*p*g")

os.makedirs( args.output, exist_ok=True)

for file in files :
    basename = os.path.basename(file)
    basename = os.path.splitext(basename)[0]
    if ( os.path.isfile(os.path.join( args.output , basename + ".png")) ) :
        print("Skiping: " + basename)
    else:
        print("Prosscing: " + basename)
        input_image = Image.open(file).convert("RGB")
        input_image = np.expand_dims(np.asarray(input_image, "float32") / 127.5 - 1, axis=0)
        pred_image = model.predict(input_image)
        pred_image = Image.fromarray(np.clip((pred_image[0] + 1) * 127.5, 0, 255).astype("uint8"), "RGB")
        pred_image.save(os.path.join( args.output , basename + ".png"))

sys.exit(0)
