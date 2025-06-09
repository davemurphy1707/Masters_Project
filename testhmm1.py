import numpy as np
from hmmlearn import hmm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import itertools
import os
import argparse # module for command line inputs

def main():
    chr2_len = 242696752
    parser = argparse.ArgumentParser(description="Input a file and a a number of states, Hmm is trained.")

    parser.add_argument("input_file", type=str, help="Path to the input file")
    parser.add_argument("scalar", type=int, help="A scalar value (the number of states)")
    args = parser.parse_args()

    data = pd.read_csv(args.input_file, header=0)
    n_components = args.scalar

    X = data.to_numpy()
    
    initial_state_distribution = np.full(n_components, 1/n_components)

    
    model = hmm.MultinomialHMM(n_components, n_iter = 1000,algorithm = "viterbi")
    model.fit(X, lengths = [X.shape[0]])

    print("\n--- Calculating Transition Probability Matrix ---")
    print(model.transmat_) 
    print("\n--- Calculating Emission Probability Matrix ---")
    print(model.emissionprob_) 
    print("\n--- Calculating Initial State Probability Matrix ---")
    print(model.startprob_) 
    
 
    print("\n--- Calculating Stationary Distribution Vector ---")
    transition_matrix = model.transmat_

    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

    #find the eigenvector corresponding to the eigenvalue closest to 1.0
    stationary_vector = eigenvectors[:, np.isclose(eigenvalues, 1.0)][:, 0]

    #normalize the vector to sum to 1.0 
    stationary_distribution = (stationary_vector / np.sum(stationary_vector)).real

    print("Stationary Distribution Vector:")
    print(stationary_distribution)

    aic = model.aic(X)
    print("\n--- The aic for this model is:")
    print(aic)
    
    bic = model.bic(X)
    print(" \n--- The bic for this model is: ---")
    print(bic)
    
    print("\nPosterior Probability for each state in the model")
    posterior_probs = model.predict_proba(X)
    print(posterior_probs)
    
    print("Converged:", model.monitor_.converged)

    print("Iterations:", model.monitor_.iter)
    
    print("Final Log Likelihood:", model.monitor_.history[-1])
    #print("Log Likelihood History:", model.monitor_.history)
    
    state = model.predict(X)

    data["state"] = state
    data["chrom"] = "chr2"
    data["start"] = data.index * 200
    data["end"] = data["start"] + 200
    data["score"] = 1000
    data["strand"] = "."
    
    data = data[data["start"] < chr2_len]
    data.loc[data["end"] > chr2_len, "end"] = chr2_len
    
    data["name"] = "state_" + data["state"].astype(str)
    
   
    
    bed_df = data[["chrom", "start", "end", "name","score","strand"]]
    
    #set the nonclemature to be unique to the data 
    input_filename = os.path.basename(args.input_file)
    base_parts = input_filename.replace(".csv", "").split("_")
    chrom = base_parts[2]  
    kmer = base_parts[3]   
    win = base_parts[4]    
    out_prefix = f"{chrom}_hmm_{win}_{kmer}_{n_components}_states"
    
    #set the path for the results to go to
    results_dir = os.path.join("results", f"{chrom}_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save BED
    bed_filename = f"{out_prefix}.bed"
    bed_output_path = os.path.join(results_dir, bed_filename)
    bed_df.to_csv(bed_output_path, sep="\t", header=False, index=False)
    print(f"Saved BED file to: {bed_filename}")
    
    #save the model stats
    stats_filename = os.path.join(results_dir, f"{out_prefix}_stats.txt")
    with open(stats_filename, "w") as f:
        f.write("--- Hidden Markov Model Statistics ---\n")
        f.write(f"Input file: {args.input_file}\n")
        f.write(f"Number of states: {n_components}\n\n")
        f.write("Transition Probability Matrix:\n")
        f.write(np.array2string(model.transmat_) + "\n\n")
        f.write("Emission Probability Matrix:\n")
        f.write(np.array2string(model.emissionprob_) + "\n\n")
        f.write("Initial State Probabilities:\n")
        f.write(np.array2string(model.startprob_) + "\n\n")
        f.write("Stationary Distribution:\n")
        f.write(np.array2string(stationary_distribution) + "\n\n")
        f.write(f"AIC: {aic}\n")
        f.write(f"BIC: {bic}\n")
        f.write(f"Final Log Likelihood: {model.monitor_.history[-1]}\n")
        f.write(f"Converged: {model.monitor_.converged}\n")
        f.write(f"Iterations: {model.monitor_.iter}\n")

    print(f"Saved model summary to: {stats_filename}")

	
if __name__ == "__main__":
    main()
