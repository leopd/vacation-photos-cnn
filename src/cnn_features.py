#!/usr/bin/env python
"""Extracts CNN features for all the image files in a directory.
Saves as a JSON file.
"""

import argparse
from collections import namedtuple
import glob
import json
import numpy as np
import os
import mxnet as mx
import sys

Batch = namedtuple('Batch', ['data'])

class CnnFeatureExtractor(object):
    """Utility object to load a pre-trained CNN and extract features
    from images"""

    def __init__(self, use_gpu=False):
        if use_gpu:
            self.ctx = mx.gpu(0)
        else:
            self.ctx = mx.cpu()
        self.load_cnn()

    def load_cnn(self):
        download_path='http://data.mxnet.io/models/imagenet-11k/'
        mx.test_utils.download(download_path+'resnet-152/resnet-152-symbol.json')
        mx.test_utils.download(download_path+'resnet-152/resnet-152-0000.params')
        cnn_sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
        feature_layer = cnn_sym.get_internals()['flatten0_output']
        ftr_mod = mx.mod.Module(symbol=feature_layer, context=self.ctx, label_names=None)
        ftr_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
        ftr_mod.set_params(arg_params, aux_params)
        self.feature_module = ftr_mod

    def load_image_file_pil(self, filename):
        from PIL import Image
        img = Image.open(filename)
        img = img.resize( (self.cnn_res, self.cnn_res), Image.ANTIALIAS )

        # convert to numpy.ndarray
        sample = np.asarray(img)
        # swap axes to make image from (224, 224, 3) to (3, 224, 224)
        sample = np.swapaxes(sample, 0, 2)
        img = np.swapaxes(sample, 1, 2)
        img = img[np.newaxis, :] 
        return img

    def load_image_file_cv2(self, filename):
        import cv2
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        if img is None:
             return None

        # convert into format (batch, RGB, width, height)
        img = cv2.resize(img, (self.cnn_res, self.cnn_res))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]
        return img

    def features_from_image(self, img):
        self.feature_module.forward(Batch([mx.nd.array(img)]))
        features = self.feature_module.get_outputs()[0].asnumpy()
        assert features.shape == (1, 2048)
        return features

    def features_for_file(self, filename):
        img = self.load_image_file_pil(filename)
        ftrs = self.features_from_image(img)
        return ftrs

        
def get_parser():
    # --help text taken from docstring at top of file.
    parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i","--indir",
            help="directory with input images",
            required=True,
            type=str)
    parser.add_argument("-o","--outputfile",
            help="File to write all the features to in JSON",
            required=True,
            type=str)
    parser.add_argument("-g","--gpu",
            help="use GPU?",
            default=False,
            action='store_true')
    return parser



def save_features(feature_dict, filename):
    with open(filename,"w") as fh:
        json.dump(feature_dict, fh, indent=2)


def main(opts):
    cfe = CnnFeatureExtractor(use_gpu=opts.gpu)
    all_files = glob.glob(opts.indir + "/*")
    print("Found %d files" % len(all_files))
    feature_dict = {}
    for img_filename in all_files:
        inpath, justfn = os.path.split(img_filename)
        justname, fnext = os.path.splitext(justfn)
        ftrs = cfe.features_for_file(img_filename)
        feature_dict[justname] = ftrs.tolist()
        print(justname)
    save_features(feature_dict, opts.outputfile)

if __name__ == "__main__":
    opts = get_parser().parse_args()
    main(opts)

