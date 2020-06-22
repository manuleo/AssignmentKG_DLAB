# PARIS - Knowledge Graph Alignment

## Introduction
This assignment aims to analyze the performance of **PARIS**, a framework for Knowledge Graph (KG) Alignment (and more), based on a probabilistic method. Specifically, we test it against two small KGs (FB15k and DB15k) and analyze what fraction of the real alignment PARIS gets right. In order to do so, Precision, Recall, F1 score, and total running time are computed and elaborated. This README enables to easily run our evaluation code to produce the same results we achieved (some small difference are acceptable during the intrinsic randomization).

## Prerequisites
The minimum requirements for the `main.py` are:
- `Python` (tested on version **_3.8.2_**)
- [pip](https://pip.pypa.io/en/stable/) (tested on version *20.1.1*) (For package installation, if needed)
- `numpy` (tested on version *1.18.1*)
- `pandas` (tested on version *1.0.4*)
- `matplotlib` (tested on version *3.1.3*)

### Note about the plots
We used a LaTeX backend for our plots so that they have a "LaTeX style" and we can use its syntax for axes labeling, titles, and anything else. If you want to reproduce the plots (using the `--plots` flag, see *Usage instruction* below), you will need to have LaTeX installed locally on your machine. The installation really depends on your OS, so we recommend to go to [LaTeX official website](https://www.latex-project.org/get/) for more information. 
For Ubuntu users, we recommend using `apt-get` to install. We installed the full version on our machines (Note that this version is around 6GB big):

    sudo apt-get install texlive-full
You can try a smaller version if you don't have enough space, but after that, you may get some missing LaTeX packets, so Google for them is the best choice.
#### What you can do instead 
We provide a notebook `Analysis.ipynb` with disclosed output with all the plots we produce in the report and in the `main.py`. If you don't want to install LaTeX you can go over this notebook to see how the plots have been generated (*Note*: that might be some small differences in how these plots are done, especially regarding the figure size). This notebook provide also the same aggregated metrics you can obtain by running the script.

## Usage instruction
1. Open CMD/Bash
2. Move to the root folder, where the `main.py` is located
3. Execute the command ```python3 main.py```, eventually adding one or more of the following arguments:
```
Optional arguments:
  -h, --help  show this help message and exit
  --no_paris   Use this flag to avoid running PARIS and load precomputed
              results from pickle instead. If not set, the full algorithm will
              be executed 20 times for 3 different seed fractions
              (10%/20%/50%).This may require about an hour.
  --plots      Produce the same plots as shown in the report and in the
              notebook and save them to pdf for later use.
```
Whatever your choice, aggregated metrics for the last iteration only and 95% confidence interval will be computed. If you choose to run the complete PARIS algorithm intermediate metrics for each run will be printed to the console.

## Folder structure
```
    .
    ├── data 
    |   ├── original                           # Original txt of DB15k and FB15k
    |   |     ├── DB15K_EntityTriples.txt  
    |   |     ├── FB15K_EntityTriples.txt 
    │   |     └── DB15K_SameAsLink.txt
    │   ├── pkl                                # Precomputed pickles for all the possible seeds
    |   |     ├── 0.1 
    |   |     |    ├── f1_scores.pkl
    |   |     |    ├── precisions.pkl
    │   |     |    ├── recalls.pkl
    |   |     |    └── timings.pkl
    │   |     ├── 0.2 
    |   |     |    ├── f1_scores.pkl
    |   |     |    ├── precisions.pkl
    │   |     |    ├── recalls.pkl
    |   |     |    └── timings.pkl
    │   |     └── 0.5 
    |   |          ├── f1_scores.pkl
    |   |          ├── precisions.pkl
    │   |          ├── recalls.pkl
    |   |          └── timings.pkl
    |   ├── DB15K_EntityTriples.nt             # Converted in N-Triples format
    |   ├── FB15K_EntityTriples.nt             # Converted in N-Triples format
    |   └── DB15K_SameAsLink.nt                # Converted in N-Triples format
    |
    ├── plots                                  # Folder to store the plots
    |    ├── ....                                
    |    └── Different kind of plots .pdf                                
    |    
    ├── main.py                                 # Main entry point
    ├── paris_0_3.jar                           # PARIS JAR to execute the experiments
    ├── plot.py                                 # Plotting utility script
    ├── requirements.txt                        # Python requirements
    ├── Analysis.ipynb                          # Notebook with disclosed output 
    ├── Report.pdf                              # Report in PDF 
    └── README.md

```
*Note*: additional folders will be created during the code execution to store PARIS intermediate results and seeded N-Triples dataset. These folders are not showed here as they are not necessary to start the algorithm (The script will create them) and they will be overwritten at each run.

## Code reproducibility
Due to the probabilistic nature of PARIS, together with the random seed generation, the user is advised that **some small differences are possible among different runs**. However, we found PARIS to be a very stable algorithm so the overall conclusion remains the same.
*Our configuration*: especially for the timing analysis, it is important to consider the configuration we used to run our experiments. That is a Lenovo Thinkpad T450s with **12GB RAM** and Intel **I7-5600U CPU**

Moreover, a special note has to be added for the two jupyter notebooks: since we are using the Latex style for the plots, sometimes we have faced unexpected troubles at rendering such graphics (running the cells which should output the plot does not have any effect). We didn't manage to solve this bug, which occurred irregularily and is not well documented on the Internet. If you would ever run the code and face such trouble, we noticed that it is enough to re-run the `matplotlib` import and the latex style cells multiple times to solve such issues and render regularly the plots. 

## Authors
- Manuel Leone
- Stefano Huber
