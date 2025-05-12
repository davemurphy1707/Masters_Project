import numpy as np
from hmmlearn import hmm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import itertools
import argparse # module for command line inputs

def main():

    parser = argparse.ArgumentParser(description="Input a file and a a number of states, Hmm is trained.")

    parser.add_argument("input_file", type=str, help="Path to the input file")
    parser.add_argument("scalar", type=int, help="A scalar value (the number of states)")
    args = parser.parse_args()

    data = pd.read_csv(args.input_file, header=0)
    n_components = args.scalar

    X = data.to_numpy()

    model = hmm.MultinomialHMM(n_components, n_iter = 100)
    model.fit(X, lengths = [X.shape[0]])

    state = model.predict(X)

    print(model.transmat_) 
    print(model.emissionprob_) 
    print(model.startprob_) 

    data["state"] = state
    data["chrom"] = "chr21"
    data["start"] = data.index * 10000
    data["end"] = data["start"] + 10000
    data["name"] = "state_" + data["state"].astype(str)
    data["score"] = 0
    data["strand"] = "."
    bed_df = data[["chrom", "start", "end", "name", "score", "strand"]]
    bed_df.to_csv("chr2_hmm_states.bed", sep="\t", header=False, index=False)

	
if __name__ == "__main__":
    main()
