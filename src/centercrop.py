#!/usr/bin/env python3
"""Image center crop tool.  Loads an image file,
takes a square crop from the center, and outputs
a JPEG file.
"""

from PIL import Image
import argparse


def get_parser():
    # --help text taken from docstring at top of file.
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i","--infile",
            help="Name of source image",
            required=True,
            type=str)
    parser.add_argument("-o","--outfile",
            help="Where to write the JPEG",
            required=True,
            type=str)
    parser.add_argument("-s","--size",
            help="Size of square image",
            default=256,
            type=int)
    return parser


def center_crop_img(img, max_res):
    """CNN's expect square images.
    Take the largest possible square out of the middle.
    """
    # First, find center square crop
    wid,hgt = img.size
    if( wid > hgt ):
        clip = (wid - hgt) / 2
        box = (clip,0, clip+hgt,hgt)
    else:
        clip = (hgt - wid) / 2
        box = (0,clip, wid,clip+wid)
    img = img.crop(box)
    # Now convert to the desired resolution
    # (LANCZOS is highest quality for down-sampling.)
    img = img.resize((max_res,max_res), resample=Image.LANCZOS)
    return img


def rotate_image_by_exif(img):
    if not hasattr(img, '_getexif'): # Check EXIF data for rotation if present
        return img
    exif_data = dict(img._getexif().items())
    ORIENTATION_TAG = 274 # exif standard
    orientation = exif_data.get(ORIENTATION_TAG)
    if not orientation:
        return img
    if orientation == 1: 
        return img  # no rotation needed.
    if orientation == 3: 
        return img.rotate(180, expand=True)
    if orientation == 6: 
        return img.rotate(270, expand=True)
    if orientation == 8: 
        return img.rotate(90, expand=True)
    print("WARN: Unknown orientation %d" % orientation)
    return img


def open_resize_image(in_file, max_res):
    """Opens an image file and prepares it 
    as a small square, properly rotated.
    """
    img = Image.open(in_file)
    img = rotate_image_by_exif(img)
    img = center_crop_img(img, max_res)
    return img

def resize_image_file(in_file, out_file, max_res):
    """Creates a downsampled version of the file.
    Center-crops a square of specified size.
    """
    img = Image.open(in_file)
    img = rotate_image_by_exif(img)
    img = center_crop_img(img, max_res)
    img.save(out_file, "JPEG")


def main(opts):
    resize_image_file(opts.infile,opts.outfile,opts.size)

if __name__ == "__main__":
    opts = get_parser().parse_args()
    main(opts)

