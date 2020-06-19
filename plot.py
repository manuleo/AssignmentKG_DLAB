import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

# Set the style for latex-like plots -> Note that this requires LaTeX installed on the machine, otherwise you will get an exception
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)


def plot(seeds, metrics, metrics_names, timings):
    """Produce all the plots

    Args:
        seeds (list): list of seed used
        metrics (dict): dict of metrics performances
        metrics_names (list): list of computed metrics
        timings (dict): dict of list of timings for each seed
    """
    # Prepare folder to save folder if not existent
    if not os.path.isdir("plots/"):
        os.mkdir("plots/")
    
    # Create the plots
    print("\nProducing plot with mean/std of each metric by iteration")
    mean_per_iter(seeds, metrics, metrics_names)
    print("\nProducing plot with average time spent for each PARIS run")
    timings_for_seed(seeds, timings)
    print("\nProducing confidence intervals at 95% for the different metrics (only last iterations of PARIS)")
    metrics_confidence(metrics, metrics_names, seeds)
    print("\nProducing confidence intervals at 95% for the running time")
    timings_confidence(seeds, timings)
    print("\nProducing plot to analyze goodness of the different metrics through boxplots (only last iteration of PARIS)")
    last_iter_goodness(seeds, metrics, metrics_names)


def compute_mean_and_std_by_metric(metric):
    """
    Compute the mean and the standard deviation for each metric,
    grouped by every iteration of the algorithm.
    
    Args:
        metric (list): the name of the metric, among 'precision', 'recall' and 'f1_score' 
    Returns:
        metric_means (list): mean of the metric by iteration
        metric_stds (list): std of the metric by iteration
    """
    metric_means = []
    metric_stds = []
    # Get last iteration number
    last_iter = max(metric[0].keys()) + 1
    for i in range(last_iter):
        # Create a list of the values for iteration and compute mean/std
        iter_values = [val[i] for val in metric]
        metric_means.append(np.mean(iter_values))
        metric_stds.append(np.std(iter_values))
    return metric_means, metric_stds


def mean_per_iter(seeds, metrics, metrics_names):
    """Produce the plot with the mean of each metric by iteration and std

    Args:
        seeds (list): list of seed used
        metrics (dict): dict of metrics performances
        metrics_names (list): list of computed metrics
    """
    # Create figure
    title = "{} among the iterations, for different seeds percentages"
    fig, axarr = plt.subplots(3, 1, figsize=(10,15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    
    # Go over the metric/seed pairs
    for i, metric in enumerate(metrics_names):
        for s in seeds:
            # Create a subplot per metric with errorbar for std
            metric_means, metric_std = compute_mean_and_std_by_metric(metrics[metric][s])
            axarr[i].set_title(title.format(metric.capitalize()), fontsize=18)
            if metric=="precision":
                axarr[i].set_ylim([0.7,1.01])
            else:
                axarr[i].set_ylim([0,1])
            axarr[i].set_xticks(range(10))
            axarr[i].set_xlabel("Iteration", fontsize=14)
            axarr[i].set_ylabel(metric.capitalize(), fontsize=14)
            axarr[i].errorbar(range(len(metric_means)), metric_means, yerr=metric_std, label=str(float(s)*100)+"%")
        axarr[i].legend(title="Seed")
    fig.suptitle("Metrics behaviour among the iterations", y=0.95, fontsize=16)
    # Save
    fig.savefig("plots/mean_metric_per_iter.pdf")
    plt.close()


def timings_for_seed(seeds, timings):
    """Create boxplot with timings

    Args:
        seeds (list): list of seed used
        timings (dict): dict of list of timings for each seed
    """
    # Create one box for each seed percentage and save
    plt.figure(figsize=(10,7))
    plt.title("Time measuration per experiment", fontsize=20)
    plt.boxplot([timings[s] for s in seeds], labels=[str(float(s)*100)+"% Seed" for s in seeds])
    plt.ylabel("Timing measurement from PARIS $[ ms ]$", fontsize=14)
    plt.xlabel("Chosen seed percentage", fontsize=14)
    plt.savefig("plots/timings_per_seed.pdf")
    plt.tick_params(labelsize=10)
    plt.close()


def bootstrap_metric(metric_list, n_iter):
    """Compute bootstrap means list for a metric to be used for computing confidence intervals
        using bootstrap resample
    Args:
        metric_list (list): The list (in the case of timings) or list of dicts of the given metric
        n_iter (int): number of sample to do for bootstrap

    Returns:
        means (list): list of n_iter means computed from the samples
    """
    means = []
    # For the Precision/Recall/F1 metrics use only the last iteration, for timings we don't have to consider this case
    if type(metric_list[0]) is dict:
        last_iter = max(metric_list[0].keys())
        metric_last = [val[last_iter] for val in metric_list]
    else:
        metric_last = metric_list
    
    # Resample and compute mean
    for _ in range(n_iter):
        # Bootstrap
        metric_sample = np.random.choice(metric_last, size=len(metric_last), replace=True)
        means.append(np.mean(metric_sample))

    return means


def confidence_interval(means, conf_percent):
    """Get confidence intervals for the given percentage by getting the quantiles

    Args:
        means (list): list of means computed from the samples
        conf_percent (float): Percentage for the confidence interval (between 0-1)

    Returns:
        [lower, upper] (list): the lower and upper bound of the confidence interval
    """
    # Computing low quantile 
    low_p = ((1.0 - conf_percent) / 2.0) * 100
    lower = np.percentile(means, low_p)

    # Computing high quantile
    high_p = (conf_percent + ((1.0 - conf_percent) / 2.0)) * 100
    upper = np.percentile(means, high_p)

    return [lower, upper]


def plot_confidence(means_metric, mean, interval, title, xlabel):
    """Plot a histogram with the confidence interval

    Args:
        means_metric (list): list of means computed from the samples
        mean (float): mean of the list
        interval (list): the lower and upper bound of the confidence interval
        title (str): Title for the plot
        xlabel (str): Label for the x axis
    """
    
    # Plot the means
    plt.hist(means_metric, bins=25)

    # Plot of two interval lines + mean line
    plt.axvline(interval[0], color='k', linestyle='dashed', linewidth=1)
    plt.axvline(interval[1], color='k', linestyle='dashed', linewidth=1)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1)

    plt.title(title, fontsize=10)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel("Count", fontsize=10)


def metrics_confidence(metrics, metrics_names, seeds):
    """Compute and plot confidence interval at 95% for Precision, Recall and F1 score

    Args:
        metrics (dict): dict of metrics performances
        metrics_names (list): list of computed metrics
        seeds (list): list of seed used
    """
    # Create plot
    index = 1
    fig, _ = plt.subplots(3, 3, figsize=(10, 15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.35)

    for m in metrics_names:
        for s in seeds:
            # Compute interval for the metric/seed pair and plot them 
            means_metric = bootstrap_metric(metrics[m][s], 1000)
            interval = confidence_interval(means_metric, 0.95)
            mean = np.mean(means_metric)
        
            plt.subplot(3, 3, index)
            plot_confidence(means_metric,
                            mean,
                            interval,
                            "$95.0$ % confidence interval for {metric}\n with 1000 samples, seed {seed}".format(
                                metric=m.capitalize(), seed = str(float(s)*100)+"%"),\
                            "Computed bootstrap means for\n {metric} with seed {seed}".format(
                                metric=m, seed=str(float(s)*100)+"%"))
            index += 1 
    fig.suptitle("Computed confidence intervals at the last iteration, for different seeds/metrics",
                 y = 0.95,
                 fontsize=20)
    fig.savefig("plots/confidence_metric.pdf")
    plt.close()


def timings_confidence(seeds, timings):
    """Compute and plot confidence interval at 95% for the total timing

    Args:
        seeds (list): list of seed used
        timings (dict): dict of list of timings for each seed
    """
    index = 1
    fig, _ = plt.subplots(1, 3, figsize=(10, 8))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.35)

    for s in seeds:
        # Compute interval for the given seed and plot it
        means_timings = bootstrap_metric(timings[s], 1000)
        interval = confidence_interval(means_timings, 0.95)
        mean = np.mean(means_timings)

        plt.subplot(1, 3, index)
        plot_confidence(means_timings, mean, interval,
                        "$95.0$% confidence interval for\n timings using 1000 samples\n",
                        "Computed bootstrap means for\n timings with seed {seed}".format(seed=str(float(s)*100)+"%"))
        index += 1 
    fig.suptitle("Computed confidence intervals for the total timing with different seeds", fontsize=16)
    fig.savefig("plots/timings_metric.pdf")
    plt.close()

def last_iter_goodness(seeds, metrics, metrics_names):
    """Create boxplot to analyze the behaviour of the different metrics (Precision, Recall, F1 score) at the last iteration

    Args:
        seeds (list): list of seed used
        metrics (dict): dict of metrics performances
        metrics_names (list): list of computed metrics
    """
    # Create plot
    fig, _ = plt.subplots(3, 3, figsize=(10, 12))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.5)

    iterator = 1
    for m in metrics_names:
        # We want to share for the same metric the same y scale, so that the standard deviation
        # is visible graphically (otherwise all y scales adapt to the size of the boxplot).
        max_interval = 0    # Compute the max_interval: initially set to zero, so that it can always improve
        for s in seeds:
            last_iter = max(metrics[m][s][0].keys())
            metric_last = [val[last_iter] for val in metrics[m][s]]
            max_interval = max(max_interval,      # Update the max_interval
                               max(metric_last) - min(metric_last))
        # Add to the max interval a small fraction, so that the bigger interval box plot does not 
        # fill entirely the y axis (just for graphical purposes). 1/5 is just a good number.
        max_interval += max_interval / 5   
        for s in seeds:
            # Create a boxplot for each combination Metric/Seed
            last_iter = max(metrics[m][s][0].keys())
            metric_last = [val[last_iter] for val in metrics[m][s]]
            limit_bottom = min(metric_last)
            limit_top = max(metric_last)
            interval = limit_top - limit_bottom
            limit_bottom = limit_bottom - (max_interval - interval)/2
            limit_top = limit_top + (max_interval - interval)/2
            plt.subplot(3, 3, iterator)
            plt.boxplot([metric_last], labels=["{metric} metric - Seed {seed}".format(metric=m.capitalize(), seed=str(float(s)*100)+"%")])
            plt.ylabel(m.capitalize(), fontsize=15)
            plt.ylim((limit_bottom, limit_top))
            plt.title("{} metric behaviour\n at last iteration".format(m.capitalize()),
                      fontsize=15)
            iterator += 1 
    fig.suptitle("Last iteration goodness for different metric and seeds", y=0.95, fontsize=20)
    fig.savefig("plots/last_iter_boxplots.pdf")
    plt.close()