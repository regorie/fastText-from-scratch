from scipy import stats
import math
import argparse
import numpy as np

def cos_similarity(v1, v2):
    # v1, v2 : same shape numpy arrays
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / n1 / n2

parser = argparse.ArgumentParser()
parser.add_argument('--dim', '-d', required=True)
parser.add_argument('--word_vector', '-wv', required=True)
parser.add_argument('--subword_vector', '-swv', required=True)
parser.add_argument('--query_data', '-data', required=True)
args = parser.parse_args()


if __name__=="__main__":
    # read vector files

    word_vec = {}
    subword_vec = np.zeros((2000000, args.dim))

    # 1. word vectors
    with open(args.word_vector, 'rb') as fin:
        first_line = fin.readline()
        n_of_vocab, dim = first_line.split()

        for line in fin:
            pass


    # 2. subword vectors

    # read query

    # calculate correlation score
    pass