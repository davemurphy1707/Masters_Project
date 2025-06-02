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
    
    initial_state_distribution = np.full(n_components, 1/n_components)
    
    model = hmm.MultinomialHMM(n_components, n_iter = 1000,algorithm = "map")
    model.fit(X, lengths = [X.shape[0]])

    print(model.transmat_) 
    print(model.emissionprob_) 
    print(model.startprob_) 
    
    # --- Line added to calculate Stationary Distribution ---
    print("\n--- Calculating Stationary Distribution Vector ---")
    transition_matrix = model.transmat_

    # Calculate eigenvalues and eigenvectors of the transpose of the transition matrix
    # The stationary distribution vector is the left eigenvector corresponding to eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

    # Find the eigenvector corresponding to the eigenvalue closest to 1.0
    # Use np.isclose for numerical stability, as eigenvalues might not be exactly 1.0
    stationary_vector = eigenvectors[:, np.isclose(eigenvalues, 1.0)][:, 0]

    # Normalize the vector to sum to 1.0 (probabilities)
    # The resulting vector might be complex if there are numerical issues, convert to real
    stationary_distribution = (stationary_vector / np.sum(stationary_vector)).real

    print("Stationary Distribution Vector:")
    print(stationary_distribution)

    # You can verify if it's indeed stationary:
    # print("\nVerification (pi * T - pi):")
    # print(np.dot(stationary_distribution, transition_matrix) - stationary_distribution
    
    aic = model.aic(X)
    print("aic")
    print(aic)
    
    bic = model.bic(X)
    print("bic")
    print(bic)
    
    state = model.predict(X)

    data["state"] = state
    data["chrom"] = "chr2"
    data["start"] = data.index * 10000
    data["end"] = data["start"] + 10000
    data["name"] = "state_" + data["state"].astype(str)
    bed_df = data[["chrom", "start", "end", "name"]]
    bed_df.to_csv("chr2_5_State_hidddenmm_states.bed", sep="\t", header=False, index=False)

	
if __name__ == "__main__":
    main()
