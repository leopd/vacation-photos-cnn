#!/usr/bin/env python
"""Image center crop tool for directories.
Give the name of the input directory and the output directory.
"""

import argparse
import glob
import os
import traceback

from centercrop import resize_image_file

def get_parser():
    # --help text taken from docstring at top of file.
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i","--indir",
            help="directory with input images",
            required=True,
            type=str)
    parser.add_argument("-o","--outdir",
            help="directory to write images to (must exist)",
            required=True,
            type=str)
    parser.add_argument("-s","--size",
            help="Size of square image",
            default=256,
            type=int)
    return parser


def resize_all_images(indir, outdir, size):
    all_files = glob.glob(indir + "/*")
    print("Found %d files" % len(all_files))
    success=""
    failures=""
    for in_full in all_files:
        inpath, justfn = os.path.split(in_full)
        noext, fnext = os.path.splitext(justfn)
        out_full = "%s/%s.jpg" % (outdir, noext)
        print("%s -> %s" % (in_full, out_full))
        try:
            resize_image_file(in_full, out_full, size)
            success += justfn + "\n"
            print(".")
        except:
            failures += "%s: %s" % (justfn, traceback.format_exc())
    print("\n\nSuccesses...:\n%s" % success)
    if failures:
        print("\n\nFailures...:\n%s" % failures)
    else:
        print("No failures")

def main(opts):
    resize_all_images(opts.indir, opts.outdir, opts.size)

if __name__ == "__main__":
    opts = get_parser().parse_args()
    main(opts)

