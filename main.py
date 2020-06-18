import pandas as pd
import numpy as np
import os, shutil
import argparse
import pickle
from plot import plot

def create_label(sample_row):
    """Create a label for each dataset, given a SameAs row

    Args:
        sample_row (Pandas row): Row from SameAs, with the two entities which are the same

    Returns:
        sample_row (Pandas row): Same row as input with two new labeled entries
    """
    # Build label from DB (we use as label what's after the resource/)
    label = sample_row["DB"].split("/")[-1].replace(">","")
    # Use label definition for N-Triples from W3C RDF recommendation. See https://www.w3.org/TR/n-triples/
    label_str = '{resource} <http://www.w3.org/2000/01/rdf-schema#label> "{label}" . \n'

    # Add the label
    sample_row["DB_labeled"] = label_str.format(resource=sample_row["DB"], label=label)
    sample_row["FB_labeled"] = label_str.format(resource=sample_row["FB"], label=label)
    return sample_row

def create_sample(same_as, frac):
    """[summary]

    Args:
        same_as (Pandas Dataframe): Dataframe containing the ground truth
        frac (float): seed percentage to use

    Returns:
        same_as (Pandas Dataframe): Sampled dataframe with two labeled column (one for DB and one for FB)
    """
    # Get a sample and build label
    sampled = same_as.sample(frac=frac, replace=False)   # Create a sample without replacement.
    sampled = sampled.apply(lambda x: create_label(x), axis=1)
    return sampled

def precision(same_list, res_list):
    """Compute the precision, given the two result

    Args:
        same_list (list): Lines from the ground truth, elaborated to be compared with the PARIS result
        res_list (list): PARIS result, elaborated to be compared with the ground truth

    Returns:
        precision(float): Precision
    """
    # Use precision definition from Information Retrieval
    same_set = set(same_list)
    res_set = set(res_list)
    precision = len(same_set.intersection(res_set))/len(res_set) # Divide by the size of the found result
    return precision

def recall(same_list, res_list):
    """Compute the recall, given the two result

    Args:
        same_list (list): Lines from the ground truth, elaborated to be compared with the PARIS result
        res_list (list): PARIS result, elaborated to be compared with the ground truth

    Returns:
        recall(float): Recall
    """
    # Use recall definition from Information Retrieval
    same_set = set(same_list)
    res_set = set(res_list)
    recall = len(same_set.intersection(res_set))/len(same_set) # Divide by the size of the truth
    return recall

def f1_score(precision, recall):
    """Compute the F1 score, given Precision and recall

    Args:
        precision (float): Precision at the given iteration
        recall (float): Recall at the given iteration

    Returns:
        f1(float): F1 score
    """
    f1 = 2*precision*recall/(precision + recall)
    return f1

def check_result(same_list, out_path):
    """Compare the PARIS result with the ground truth to compute Precision, Recall, F1 score by iteration

    Args:
        same_list (list): Lines from the ground truth, elaborated to be compared with the PARIS result
        out_path (string): directory where PARIS results are stored

    Returns:
        precision_dict (list): Dict with the precision of the PARIS run by iteration
        recall_dict (list): Dict with the recall of the PARIS run by iteration
        f1_dict (list): Dict with the f1_score of the PARIS run by iteration
    """
    run = 0
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}

    # Go over all the iterations without knowing a priori when to stop
    while True:
        # Get result of this iteration and check if the file exist
        full_path = out_path + "/{run}_eqv.tsv".format(run=run)
        if os.path.exists(full_path):
            # PARIS create an empty file at the last_iter+1. If we encountered it, we can break
            if os.stat(full_path).st_size == 0:
                break
            
            # Get PARIS result from the .tsv and elaborate it a bit to be compared with the same_list
            res_list = pd.read_csv(full_path, delimiter="\t", header=None, usecols=[0,1]).values.tolist()
            res_list = [" ".join(x) for x in res_list]

            # Compute the wanted metrics and save the results
            prec = precision(same_list, res_list)
            rec = recall(same_list, res_list)
            f1 = f1_score(prec, rec)
            precision_dict[run] = prec
            recall_dict[run] = rec
            f1_dict[run] = f1
        else:
            # If we didn't find the file, we can stop
            break
        run+=1
    return precision_dict, recall_dict, f1_dict

def run_experiment(same_as, DB_lines, FB_lines, same_list, frac, n=20):
    """Run PARIS for n times with the given fraction of the ground truth and seed and save the results
       for each iteration/run

    Args:
        same_as (Pandas Dataframe): Dataframe containing the ground truth
        DB_lines (list): Lines from the DB dataset N-Triples file
        FB_lines (list): Lines from the FB dataset N-Triples file
        same_list (list): Lines from the ground truth, elaborated to be compared with the PARIS result
        frac (float): seed percentage to use
        n (int, optional): Number of times which PARIS will be executed. Defaults to 20.

    Returns:
        precisions (list): List of dict with the precision of each PARIS run by iteration
        recalls (list): List of dict with the recall of each PARIS run by iteration
        f1_scores (list): List of dict with the f1_score of each PARIS run by iteration
        timings (list): List of running time of each PARIS run
    """
    
    # Create the necessary static directories.
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir("data/seeded"):
        os.mkdir("data/seeded")
    if not os.path.isdir("output"):
        os.mkdir('output')
    
    print("Chosen fraction = {}%".format(frac*100))
    timings = []
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(n):
        print("Run {}".format(i))
        print("Creating seed...")
        # Create a seed sampling from the truth and adding it to the original files
        sampled = create_sample(same_as, frac)
        DB_lines_labeled = DB_lines + sampled["DB_labeled"].to_list()
        FB_lines_labeled = FB_lines + sampled["FB_labeled"].to_list()
        DB_path = "data/seeded/{frac}/DB_{i}.nt".format(frac = frac, i = i)
        FB_path = "data/seeded/{frac}/FB_{i}.nt".format(frac = frac, i = i)

        # Create the necessary directories if they are not already present.
        if not os.path.isdir("data/seeded/{frac}".format(frac=frac)):
            os.mkdir("data/seeded/{frac}".format(frac=frac))
        
        # Save the seeded files
        DB_label = open(DB_path, "w")
        FB_label = open(FB_path, "w")
        DB_label.writelines(DB_lines_labeled)
        FB_label.writelines(FB_lines_labeled)

        # Delete old PARIS directory and create again to be empty
        out_path = "output/{frac}/{i}".format(frac = frac, i = i)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        if not os.path.isdir("output/{frac}".format(frac=frac)):
            os.mkdir("output/{frac}".format(frac=frac))
        os.mkdir(out_path)

        # Run PARIS
        print("Running PARIS...")
        out_paris = os.popen("java -jar paris_0_3.jar {FB_path} {DB_path} {out_path}".format(FB_path=FB_path, DB_path=DB_path, out_path=out_path)).read()
        print("PARIS output:\n", out_paris)

        # Get total running time directly from PARIS output (printed in millisecond)
        timing = [int(s) for s in out_paris.split() if s.isdigit()][0]
        timings.append(timing)
        
        # Compute the wanted metrics
        precision, recall, f1_score = check_result(same_list, out_path)
        print("Computed precision per iteration:")
        for run, prec in precision.items():
            print("    Run {run}: {prec}".format(run=run, prec=prec))
        print("Computed recall per iteration:")
        for run, rec in recall.items():
            print("    Run {run}: {rec}".format(run=run, rec=rec))
        print("Computed F1 score per iteration:")
        for run, f1 in f1_score.items():
            print("    Run {run}: {f1}".format(run=run, f1=f1))
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
        
        # Final clean of open files/PARIS temporary files
        print("Cleaning...")
        DB_label.close()
        FB_label.close()
        os.system("rm run_*")

    return precisions, recalls, f1_scores, timings



def main(no_paris, plots):
    """Main function to run experiment on PARIS and produce plots

    Args:
        no_paris (bool): True if we want to load precomputed pickles instead of running PARIS
        plots (bool): Wheter to produce the plots, as seen in the notebook/report
    """

    if not no_paris:
        print("Start working on PARIS. Loading datasets...")
        # Loading the SameAs relation dataset to produce the seed
        same_as = pd.read_csv("data/DB15K_SameAsLink.nt", " ", header=None)[[0,2]]
        same_as.rename(columns={0:"FB", 2:"DB"}, inplace=True)
        # Loading the triples datasets and read the content
        DB = open("data/DB15K_EntityTriples.nt", "r")
        FB = open("data/FB15K_EntityTriples.nt", "r")
        DB_lines = DB.readlines()
        FB_lines = FB.readlines()
        # Loading again the SameAs (to be used as ground truth) and transform it so that's easy comparable with the PARIS result
        same_file = open("data/DB15K_SameAsLink.nt", "r")
        same_list = same_file.readlines()
        same_list = [same.replace(" <SameAs>", "").replace("<http://dbpedia.org/","dbp:")[:-4] for same in same_list]

        # Go over the different seed size and run PARIS
        for frac in [0.1, 0.2, 0.5]:
            precisions, recalls, f1_scores, timings = run_experiment(same_as, DB_lines, FB_lines, same_list, frac)
            # Save performance results from PARIS
            print("Finished with fraction {}%. Saving data...".format(frac*100))
            with open("data/pkl/{}/precisions.pkl".format(frac), "wb") as f:
                pickle.dump(precisions, f)
            with open("data/pkl/{}/recalls.pkl".format(frac), "wb") as f:
                pickle.dump(recalls, f)
            with open("data/pkl/{}/f1_scores.pkl".format(frac), "wb") as f:
                pickle.dump(f1_scores, f)
            with open("data/pkl/{}/timings.pkl".format(frac), "wb") as f:
                pickle.dump(timings, f)
        # Close original files
        DB.close()
        FB.close()
        same_file.close()
    else:
        print("Loading precomputed pickle...")
        # Read results from pickles
        seeds = ['0.1', '0.2', '0.5']
        metrics_names = ['precision', 'recall', 'f1_score']
        metrics = {}
        timings = {}

        for m in metrics_names:
            metrics[m] = {}
        for s in seeds:
            with open("data/pkl/{seed}/precisions.pkl".format(seed=s), "rb") as f:
                metrics['precision'][s] = pickle.load(f)
            with open("data/pkl/{seed}/recalls.pkl".format(seed=s), "rb") as f:
                metrics['recall'][s] = pickle.load(f)
            with open("data/pkl/{seed}/f1_scores.pkl".format(seed=s), "rb") as f:
                metrics['f1_score'][s] = pickle.load(f)
            with open("data/pkl/{seed}/timings.pkl".format(seed=s), "rb") as f:
                timings[s] = pickle.load(f)
        print("All pickles loaded!")
        
    if plots:
        print("\nStart saving plots")
        plot(seeds, metrics, metrics_names, timings)
        print("\nAll plots saved!")
    
    print("\nNothing else to do! Closing")
        
            


if __name__ == "__main__":
    """Run main with argument
    """
    parser = argparse.ArgumentParser(description="Run PARIS entity matching and compute metrics")

    parser.add_argument(
        "-no_paris", action="store_true", help="Use this flag to avoid running PARIS and load precomputed results from pickle instead. \
                                                If not set, the full algorithm will be executed 20 times for 3 different seed fractions (10/20/50%). \
                                                This may require about half an hour.",
    )
    parser.add_argument(
        "-plots", action="store_true", help="Produce the same plots as shown in the report and in the notebook and save them to pdf for later use. \
                                             Note that you must produce the plots if you want to have the 95% confidence intervals",
    )

    args = parser.parse_args()
    main(args.no_paris, args.plots)