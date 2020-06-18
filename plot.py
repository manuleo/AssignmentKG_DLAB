import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

#Set the style for latex-like plots
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def plot(seeds, metrics, metrics_names, timings):
    # Prepare folder to save folder if not existent
    if not os.path.isdir("plots/"):
        os.mkdir("plots/")
    mean_per_iter(seeds, metrics, metrics_names)
    timings_for_seed(seeds, timings)
    metrics_confidence(metrics, metrics_names, seeds)
    timings_confidence(seeds, timings)
    last_iter_goodness(seeds, metrics, metrics_names)


def compute_mean_and_std_by_metric(metric: list):
    """
    Compute the mean and the standard deviation for each metric,
    grouped by every iteration of the algorithm.
    
    Arguments:
        metric: the name of the metric, among 'precision', 'recall' and 'f1_score' 
    Returns:
        
    """
    metric_means = []
    metric_stds = []   
    last_iter = max(metric[0].keys()) + 1
    for i in range(last_iter):
        iter_values = [val[i] for val in metric]
        metric_means.append(np.mean(iter_values))
        metric_stds.append(np.std(iter_values))
    return metric_means, metric_stds


def mean_per_iter(seeds, metrics, metrics_names):
    title = "{} among the iterations, for different seeds percentages"
    fig, axarr = plt.subplots(3, 1, figsize=(25,15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    for i, metric in enumerate(metrics_names):
        for s in seeds:
            metric_means, metric_std = compute_mean_and_std_by_metric(metrics[metric][s])
            axarr[i].set_title(title.format(metric.capitalize()))
            if metric=="precision":
                axarr[i].set_ylim([0.7,1.01])
            else:
                axarr[i].set_ylim([0,1])
            axarr[i].set_xticks(range(10))
            axarr[i].set_xlabel("Iteration")
            axarr[i].set_ylabel(metric.capitalize())
            axarr[i].errorbar(range(len(metric_means)), metric_means, yerr=metric_std, label=str(float(s)*100)+"%")
        axarr[i].legend(title="Seed")
    fig.suptitle("Metrics behaviour among the iterations", y=0.95, size=16)
    fig.savefig("plots/mean_metric_per_iter.pdf")
    plt.close()


def timings_for_seed(seeds, timings):
    plt.figure(figsize=(15,8))
    plt.title("Time measuration per experiment", size=16)
    plt.boxplot([timings[s] for s in seeds], labels=[str(float(s)*100)+"% Seed" for s in seeds])
    plt.ylabel("Timing measurement from PARIS $[ ms ]$")
    plt.xlabel("Chosen seed percentage")
    plt.savefig("plots/timings_per_seed.pdf")
    plt.close()


def bootstrap_metric(metric_list, n_iter):
    """Compute bootstrap means list to be used for computing confidence intervals
        using bootstrap resample"""
    means = []
    if type(metric_list[0]) is dict:
        last_iter = max(metric_list[0].keys())
        metric_last = [val[last_iter] for val in metric_list]
    else:
        metric_last = metric_list
    for _ in range(n_iter):
        # Bootstrap
        metric_sample = np.random.choice(metric_last, size=len(metric_last), replace=True)
        means.append(np.mean(metric_sample))

    return means


def confidence_interval(means, conf_percent):
    # Computing low quantile 
    low_p = ((1.0 - conf_percent) / 2.0) * 100
    lower = np.percentile(means, low_p)

    # Computing high quantile
    high_p = (conf_percent + ((1.0 - conf_percent) / 2.0)) * 100
    upper = np.percentile(means, high_p)

    return [lower, upper]


def plot_confidence(means_metric, mean, interval, title, xlabel):
    
    # Plot scores
    plt.hist(means_metric, bins=25)

    # Plot of two interval lines
    plt.axvline(interval[0], color='k', linestyle='dashed', linewidth=1)
    plt.axvline(interval[1], color='k', linestyle='dashed', linewidth=1)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")


def metrics_confidence(metrics, metrics_names, seeds):
    index = 1
    fig, _ = plt.subplots(3, 3, figsize=(20, 15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.35)

    for m in metrics_names:
        for s in seeds:
            means_metric = bootstrap_metric(metrics[m][s], 1000)
            interval = confidence_interval(means_metric, 0.95)
            mean = np.mean(means_metric)
        
            plt.subplot(3, 3, index)
            plot_confidence(means_metric, mean, interval,\
                            "$95.0$ % confidence interval for {metric} with 1000 samples, seed {seed}".format(metric=m.capitalize(), seed = str(float(s)*100)+"%"),\
                            "Computed bootstrap means for {metric} with seed {seed}".format(metric=m, seed=str(float(s)*100)+"%"))
            index += 1 
    fig.suptitle("Computed confidence intervals at the last iteration, for different seeds/metrics", y = 0.95, size=16)
    fig.savefig("plots/confidence_metric.pdf")
    plt.close()


def timings_confidence(seeds, timings):
    index = 1
    fig, _ = plt.subplots(1, 3, figsize=(17, 10))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.35)

    for s in seeds:
        means_timings = bootstrap_metric(timings[s], 1000)
        interval = confidence_interval(means_timings, 0.95)
        mean = np.mean(means_timings)

        plt.subplot(1, 3, index)
        plot_confidence(means_timings, mean, interval,\
                        "$95.0$ % confidence interval for Timings using 1000 samples",\
                        "Computed bootstrap means for timings with seed {seed}".format(seed=str(float(s)*100)+"%"))
        index += 1 
    fig.suptitle("Computed confidence intervals for the total timing with different seeds", size=16)
    fig.savefig("plots/timings_metric.pdf")
    plt.close()

def last_iter_goodness(seeds, metrics, metrics_names):
    fig, _ = plt.subplots(3, 3, figsize=(20, 12))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)

    iterator = 1
    for m in metrics_names:
        for s in seeds:
            last_iter = max(metrics[m][s][0].keys())
            metric_last = [val[last_iter] for val in metrics[m][s]]
            plt.subplot(3, 3, iterator)
            plt.boxplot([metric_last], labels=["{metric} metric - Seed {seed}".format(metric=m.capitalize(), seed=str(float(s)*100)+"%")])
            plt.ylabel(m.capitalize())
            plt.title("{} metric behaviour at last iteration".format(m.capitalize()))
            iterator += 1 
    fig.suptitle("Last iteration goodness for different metric and seeds", y=0.95, size=16)
    fig.savefig("plots/last_iter_boxplots.pdf")
    plt.close()