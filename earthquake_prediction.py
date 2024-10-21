"""Earthquake prediction using statistics"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt


def fit_data_and_make_prediction(x, y, x_pred, x_fit, n=10_000):
    """
    Fit the data and make a prediction
    :param x: x values
    :param y: y values
    :param x_pred: x value to predict
    :param x_fit: x values to fit
    :param n: number of samples
    :return: samples, y_pred, y_fit, y_fit_upper, y_fit_lower
    """
    log_y = np.log(y)
    slope, intercept = np.polyfit(x, log_y, 1)
    log_fit = intercept + slope * x
    std = np.std(log_y - log_fit)

    # generate N samples from a normal distribution
    mu = intercept + slope * x_pred
    samples = np.random.normal(mu, std, n)

    # apply the exponential function to the samples
    samples = np.exp(samples)

    y_fit = np.exp(intercept + slope * x_fit)
    y_fit_upper = y_fit * np.exp(std)
    y_fit_lower = y_fit / np.exp(std)

    y_pred = np.exp(intercept + slope * x_pred)

    return samples, y_pred, y_fit, y_fit_upper, y_fit_lower


def plot_graphs(x_fit, y_fit, y_fit_upper, y_fit_lower, y, y_pred,
                samples, median, median_date, this_year, last_year, bins=300):

    fig, ax = plt.subplots(ncols=2)

    # plot the fit of the data
    plt.sca(ax[0])
    plt.title("Fit of time between events")
    x_ids = list(range(len(y)))
    plt.scatter(x_ids, y, label="Data")
    plt.plot(x_fit, y_fit, "--", c='k', alpha=0.4, label="Fit")
    plt.plot(x_fit, y_fit_upper, "-", c='k', alpha=0.3, label="upper std")
    plt.plot(x_fit, y_fit_lower, "-", c='k', alpha=0.3, label="lowe std")
    plt.scatter(len(y), y_pred, c='r', label="Prediction")
    plt.xlabel("ID event")
    plt.ylabel("Years between events")
    plt.ylim(0, None)
    plt.legend()

    # plot the distribution of the samples
    plt.sca(ax[1])
    plt.title("Next earthquake\nprobability distribution")
    h, bins, *_ = plt.hist(samples, bins=bins, density=True, alpha=0.5, color="orange", label="Prediction")
    plt.plot((bins[1:] + bins[:-1])/2, h, color="orange", alpha=1., linewidth=4)
    plt.axvline(median, linestyle="--", color="k", alpha=0.5, label="Median")
    plt.xlabel("Year")
    plt.ylabel("Likelihood")

    # print median date on the median line
    y_plot = plt.gca().get_ylim()[-1]
    s_date = f"{median_date.year}"
    plt.text(median, y_plot*0.85, s_date, rotation=-90, verticalalignment="center")
    plt.xlim(this_year, last_year)

    # change the y-ticks to probabilities
    y_values = list(plt.gca().get_yticks())
    plt.gca().set_yticks(y_values)
    plt.gca().set_yticklabels([f"{i:.1%}" for i in plt.gca().get_yticks()])

    plt.legend()

    plt.show()


def main(n_bins=10**7, data_path=r"earthquake_data.txt"):

    data = pd.read_csv(data_path, sep="\t")
    # data = data[:-1]
    print(data)

    y = np.array(data["Year"])

    this_year = dt.datetime.now().year + dt.datetime.now().timetuple().tm_yday / 365
    # this_year = 1990
    last_year = this_year + 100

    y = y[1:] - y[:-1]
    x = np.arange(len(y))
    x_fit = np.linspace(-1, len(y)+1, 1000)

    samples, y_pred, y_fit, y_fit_upper, y_fit_lower = fit_data_and_make_prediction(x, y, x_pred=len(y),
                                                                                    x_fit=x_fit, n=n_bins)
    samples += np.array(data["Year"]).max()

    # cut the samples that occur before this year
    samples = samples[samples > this_year]
    assert len(samples) > 0

    # find the median of the samples
    median = np.median(samples)
    print(f"Median: {median:.1f}")

    # find the mode of the samples
    mode = np.argmax(np.bincount(samples.astype(int)))
    print(f"  Mode: {mode:.0f}")

    # convert median (in years) to date
    median_date = dt.datetime.fromordinal(int(median) * 365 + 1)

    plot_graphs(x_fit, y_fit, y_fit_upper, y_fit_lower, y, y_pred,
                samples, median, median_date, this_year, last_year)


if __name__ == "__main__":
    main()
