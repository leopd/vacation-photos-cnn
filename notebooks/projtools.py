import json
import numpy as np

class FeatureDict(object):

    def __init__(self):
        self.ftr_dict = {}
        self.ftr_matrix = None
        self.names = None

    def load_json(self,filename):
        self.ftr_dict = json.load(open(filename))
        self.calc_matrix()

    def calc_matrix(self):
        self.names = self.ftr_dict.keys()
        out = []
        for name in self.names:
            ftrs = self.ftr_dict[name]
            if( type(ftrs[0]) == list ):
                ftrs = ftrs[0]
            out.append(ftrs)
        self.ftr_matrix = np.asarray(out)

