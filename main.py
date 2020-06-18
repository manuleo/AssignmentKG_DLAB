import pandas as pd
import numpy as np
import os, shutil
import argparse
import pickle

def create_label(sample_row):
    label = sample_row["DB"].split("/")[-1].replace(">","")
    label_str = '{resource} <http://www.w3.org/2000/01/rdf-schema#label> "{label}" . \n'
    sample_row["DB_labeled"] = label_str.format(resource=sample_row["DB"], label=label)
    sample_row["FB_labeled"] = label_str.format(resource=sample_row["FB"], label=label)
    return sample_row

def create_sample(same_as, frac):
    sampled = same_as.sample(frac=frac, replace=False)   # Create a sample without replacement.
    sampled = sampled.apply(lambda x: create_label(x), axis=1)
    return sampled

def precision(same_list, res_list):
    same_set = set(same_list)
    res_set = set(res_list)
    precision = len(same_set.intersection(res_set))/len(res_set)
    return precision

def recall(same_list, res_list):
    same_set = set(same_list)
    res_set = set(res_list)
    recall = len(same_set.intersection(res_set))/len(same_set)
    return recall

def f1_score(precision, recall):
    f1 = 2*precision*recall/(precision + recall)
    return f1

def check_result(same_list, out_path):
    run = 0
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    while True:
        full_path = out_path + "/{run}_eqv.tsv".format(run=run)
        if os.path.exists(full_path):
            if os.stat(full_path).st_size == 0:
                break
            res_list = pd.read_csv(full_path, delimiter="\t", header=None, usecols=[0,1]).values.tolist()
            res_list = [" ".join(x) for x in res_list]
            prec = precision(same_list, res_list)
            rec = recall(same_list, res_list)
            f1 = f1_score(prec, rec)
            precision_dict[run] = prec
            recall_dict[run] = rec
            f1_dict[run] = f1
        else:
            break
        run+=1
    return precision_dict, recall_dict, f1_dict

def run_experiment(same_as, DB_lines, FB_lines, same_list, frac, n=20):
    
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
        sampled = create_sample(same_as, frac)
        DB_lines_labeled = DB_lines + sampled["DB_labeled"].to_list()
        FB_lines_labeled = FB_lines + sampled["FB_labeled"].to_list()
        DB_path = "data/seeded/{frac}/DB_{i}.nt".format(frac = frac, i = i)
        FB_path = "data/seeded/{frac}/FB_{i}.nt".format(frac = frac, i = i)

        # Create the necessary directories if they are not already present.
        if not os.path.isdir("data/seeded/{frac}".format(frac=frac)):
            os.mkdir("data/seeded/{frac}".format(frac=frac))
        
        DB_label = open(DB_path, "w")
        FB_label = open(FB_path, "w")
        DB_label.writelines(DB_lines_labeled)
        FB_label.writelines(FB_lines_labeled)

        out_path = "output/{frac}/{i}".format(frac = frac, i = i)
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        
        if not os.path.isdir("output/{frac}".format(frac=frac)):
            os.mkdir("output/{frac}".format(frac=frac))
        os.mkdir(out_path)
        print("Running PARIS...")
        out_paris = os.popen("java -jar paris_0_3.jar {FB_path} {DB_path} {out_path}".format(FB_path=FB_path, DB_path=DB_path, out_path=out_path)).read()
        print("PARIS output:\n", out_paris)
        timing = [int(s) for s in out_paris.split() if s.isdigit()][0]
        timings.append(timing)
        
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

        print("Cleaning...")
        DB_label.close()
        FB_label.close()
        os.system("rm run_*")

    return precisions, recalls, f1_scores, timings



def main(no_paris):

    if not no_paris:
        print("Start working on PARIS. Loading datasets...")
        same_as = pd.read_csv("data/DB15K_SameAsLink.nt", " ", header=None)[[0,2]]
        same_as.rename(columns={0:"FB", 2:"DB"}, inplace=True)
        DB = open("data/DB15K_EntityTriples.nt", "r")
        FB = open("data/FB15K_EntityTriples.nt", "r")
        DB_lines = DB.readlines()
        FB_lines = FB.readlines()
        same_file = open("data/DB15K_SameAsLink.nt", "r")
        same_list = same_file.readlines()
        same_list = [same.replace(" <SameAs>", "").replace("<http://dbpedia.org/","dbp:")[:-4] for same in same_list]

        for frac in [0.1, 0.2, 0.5]:
            precisions, recalls, f1_scores, timings = run_experiment(same_as, DB_lines, FB_lines, same_list, frac)
            print("Finished with fraction {}%. Saving data...".format(frac*100))
            with open("data/pkl/{}/precisions.pkl".format(frac), "wb") as f:
                pickle.dump(precisions, f)
            with open("data/pkl/{}/recalls.pkl".format(frac), "wb") as f:
                pickle.dump(recalls, f)
            with open("data/pkl/{}/f1_scores.pkl".format(frac), "wb") as f:
                pickle.dump(f1_scores, f)
            with open("data/pkl/{}/timings.pkl".format(frac), "wb") as f:
                pickle.dump(timings, f)
        DB.close()
        FB.close()
        same_file.close()
    else:
        print("Loading precomputed pickle...")
        # TODO: Do something with loaded data...
        pass
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PARIS entity matching and compute metrics")

    parser.add_argument(
        "-no_paris", action="store_true", help="Use this flag to avoid running PARIS and load precomputed results from pickle instead. \
                                                If not set, the full algorithm will be executed 20 times for 3 different seed fractions (10/20/50%). \
                                                This may require about half an hour.",
    )
    args = parser.parse_args()
    main(args.no_paris)