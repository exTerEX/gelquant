"""Source code for Gelquant."""

# Standard Python Modules
import shutil
from decimal import Decimal

# Third Party Python Modules
from matplotlib import pyplot as plt
import numpy
import numpy as np
import PIL
import pandas as pd
from scipy.integrate import trapz
from scipy.stats import norm
from scipy.optimize import curve_fit


def image_cropping(path: str, bbox: tuple, show: bool = False) -> numpy.ndarray:
    """Crop image in preparation for gel analysis.

    :param path: Path to image that will be cropped.
    :type path: str
    :param bbox: A tuple with crop points (x1, y1, x2, y2), where (x1, y1) point to top-leftmost
        value, while (x2, y2) point to bottom-rightmost value.
    :type bbox: tuple
    :param show: Whether or not to show original and cropped images with
        matplotlib, defaults to False.
    :type show: bool
    :return: Cropped image as numpy.ndarray.
    :rtype: numpy.ndarray
    """
    original = PIL.Image.open(path)
    crop = original.crop(bbox)

    if show:
        plt.figure(figsize=(7, 14))
        plt.subplot(121)
        plt.imshow(original)
        plt.title("original image")

        plt.subplot(122)
        plt.imshow(crop)
        plt.title("cropped image")

        plt.tight_layout()
        plt.show()

    return numpy.array(crop)


def lane_parser(img: numpy.ndarray, lanes: int, groups: int, baseline: list,
                tolerance: float = 0.1, show: bool = False) -> list:
    """[summary]

    :param img: Image of gel with lanes and protein bands to be analysed
    :type img: numpy.ndarray
    :param lanes: Amount of lanes in image
    :type lanes: int
    :param groups: Amount of different proteins
    :type groups: int
    :param baseline: y-value of protein lane.
    :type baseline: list
    :param tolerance: [description], defaults to 0.1
    :type tolerance: float, optional
    :param show: [description], defaults to False
    :type show: bool, optional
    :return: [description]
    :rtype: list
    """
    baseline1, baseline2 = baseline

    # TODO: Work with map(...)
    image_list = [
        img[:len(img), int(len(img[0]) * index / lanes):int(len(img[0]) * (index + 1) / lanes)]
        for index in range(lanes)]

    final_data = []

    for i, element1 in enumerate(image_list):

        all_intensities = []

        for j in range(len(element1)):
            row_intensities = []
            for k in range(len(element1[0])):
                pixel = element1[j, k]
                intensity = 1 - (0.2126 * pixel[0] / 255 + 0.7152 *
                                 pixel[1] / 255 + 0.0722 * pixel[2] / 255)
                row_intensities.append(intensity)
            all_intensities.append(row_intensities)

        final_intensities = []

        for k, element2 in enumerate(all_intensities):
            x = numpy.linspace(norm.ppf(0.01), norm.ppf(
                0.99), len(element2))
            weights = norm.pdf(x)
            ave_intensity = numpy.average(
                element2, weights=weights * sum(weights))
            final_intensities.append(ave_intensity)

        final_intensities = numpy.array(
            final_intensities) - numpy.mean(final_intensities[baseline1:baseline2])

        final_data.append(final_intensities)

    peakzero_xs, peakzero_ys = [], []

    for i in range(groups):
        initial_peak = max(final_data[int(i * len(final_data) / groups)])
        peakzero_ys.append(initial_peak)
        for j in range(len(final_data[int(i * len(final_data) / groups)])):
            if initial_peak == final_data[int(i * len(final_data) / groups)][j]:
                peakzero_xs.append(j)

    all_bounds = []

    for i in range(groups):

        peak = peakzero_ys[i]
        bounds = []

        for j in range(len(final_data[0])):
            if final_data[int(i * len(final_data) / groups)][j] < tolerance * peak:
                continue
            if final_data[int(i * len(final_data) / groups)][j] > tolerance * peak:
                bounds.append(j)

        lower_bound, upper_bound = bounds[0], bounds[-1]

        for k in range(int(len(final_data) / groups)):
            all_bounds.append([lower_bound, upper_bound])

    if show:
        for i, element in enumerate(final_data):
            plt.plot(numpy.arange(len(element)), element, "-")
            plt.plot([all_bounds[0], all_bounds[0]], [-0.1, 0.7], "--", color="green")
            plt.plot([all_bounds[1], all_bounds[1]], [-0.1, 0.7], "--", color="green")
            plt.ylim(-0.1, 0.7)
            plt.show()

    return final_data, all_bounds[0]


def area_integrator(data, bounds, groups, plot_output=False, percentages=True):

    def linear_baseline(x, m, b):
        return m*x+b

    baseline_xs = []
    baseline_ys = []

    for i in range(len(data)):

        x1 = bounds[0]
        x2 = bounds[1]
        y1 = data[i][bounds[0]]
        y2 = data[i][bounds[1]]
        m = (y2-y1)/(x2-x1)
        b = y1-m*x1

        baseline_x = np.arange(len(data[i][bounds[0]:bounds[1]]))
        baseline_x = baseline_x + bounds[0]

        baseline_y = []

        for i in range(len(baseline_x)):
            y = m*baseline_x[i] + b
            baseline_y.append(y)

        baseline_xs.append(baseline_x)
        baseline_ys.append(baseline_y)

    if plot_output == True:

        for i in range(len(data)):
            plt.plot(np.arange(len(data[i])), data[i], "-")
            plt.plot([bounds[0],bounds[0]], [-0.1, 0.7], "--", color="green")
            plt.plot([bounds[1],bounds[1]], [-0.1, 0.7], "--", color="green")
            plt.plot(baseline_xs[i], baseline_ys[i], "--", color="green")
            plt.ylim(-0.1, 0.7)
            plt.show()

    trunc_data = []

    for i in range(len(data)):
        d = data[i][bounds[0]:bounds[1]]
        d = d - baseline_ys[i]
        trunc_data.append(d)

    areas = []

    for i in range(len(trunc_data)):
        area_trapz = trapz(trunc_data[i])
        areas.append(area_trapz)

    sorted_areas = []

    for i in range(groups):

        index = int(len(data)/groups)
        group = areas[i*index:i*index+index]
        sorted_areas.append(group)

    sorted_areas = [item for sublist in sorted_areas for item in sublist]

    if percentages == True:
        return sorted_areas/sorted_areas[0]
    else:
        return sorted_areas

def summary_data(datasets, timepoints="", output="", p0=[7, 0.2], input_df = False):

    plt.figure(figsize=(2.5,2.5))
    plt.rcParams["font.family"] = "Times New Roman"

    if input_df == True:
        if type(datasets) != pd.core.frame.DataFrame:
            df = pd.read_json(datasets)
            plt.title(datasets.split(".")[0])
            df.to_json(output + ".json")
        else:
            df = datasets
            df.to_json (output + ".json")

    else:

        data = np.array(datasets).flatten()
        time = list(timepoints)*int((len(data)/len(timepoints)))
        time = [int(i) for i in time]
        df = pd.DataFrame({"timepoint":time, "value":data})
        df.to_json(output + ".json")

    def decay(x, a, k):
        return a * np.exp(-k * x)

    popt, pcov = curve_fit(decay, df.timepoint, df.value, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    plt.plot(df.timepoint,df.value, ".")
    plt.ylabel("Normalized \n pixel intensity", fontsize=10)
    plt.xlabel("Time (minutes)", fontsize=10)
    x_decay = np.linspace(0,1000,1000)
    plt.xlim(-1, max(df.timepoint)+5)
    plt.ylim(0,)
    plt.text(0.5,0.5,"k = " + f"{Decimal(str(popt[1])):.2E}" + "\n" + r' $\pm$ ' + f"{Decimal(str(perr[1])):.2E}" + r' min$^{-1}$', fontsize=10)
    plt.plot(x_decay, decay(x_decay, *popt))

    plt.tight_layout()
    plt.savefig(output + "_decay_curve.svg", dpi=100)
    plt.show()
    None

    return popt, perr


def fancy_plotter(dataset, ks, errs, colors, output, ylim=None,
                  ylabel=None, log=True, labeling=True):

    f, ax = plt.subplots(1, 1, figsize=(len(dataset)/1.6,5))

    ax.bar(np.arange(len(ks)), ks, yerr=[np.zeros(len(errs)), errs], color=colors,
           edgecolor="black", capsize=7)

    labels = [i.split(".")[0] for i in dataset]

    if log == True:
        ax.set_yscale('log')
    ax.set_ylim(ylim)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if labeling==True:
        plt.xticks(range(len(ks)), labels, rotation=90, fontsize=20)
    else:
        plt.xticks(range(len(ks)), "", rotation=90, fontsize=20)
    plt.ylabel(ylabel)
    f.savefig(output, bbox_inches = "tight", dpi=1000)
    None
