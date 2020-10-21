"""Source code for Gelquant."""

# Standard Python Modules
from decimal import Decimal
import itertools

# Third Party Python Modules
import matplotlib.pyplot
import numpy
import PIL.Image
import pandas
import scipy
import scipy.stats
import scipy.integrate


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
    :return: Cropped image as numpy.ndarray
    :rtype: numpy.ndarray
    """
    original = PIL.Image.open(path)
    crop = original.crop(bbox)

    if show:
        matplotlib.pyplot.figure(figsize=(7, 14))
        matplotlib.pyplot.subplot(121)
        matplotlib.pyplot.imshow(original)
        matplotlib.pyplot.title("original image")

        matplotlib.pyplot.subplot(122)
        matplotlib.pyplot.imshow(crop)
        matplotlib.pyplot.title("cropped image")

        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()

    return numpy.array(crop)


# TODO: Move to utils.py
def crop_to_lane(obj: numpy.ndarray, lane: int, lanes: int) -> numpy.ndarray:
    """Crop a numpy.ndarray to selected lane.

    :param obj: Array to be cropped
    :type obj: numpy.ndarray
    :param lane: Selected lane
    :type lane: int
    :param lanes: Total lanes
    :type lanes: int
    :return: Return the cropped object representing lane data
    :rtype: numpy.ndarray
    """
    x_1 = int(len(obj[0]) * lane / lanes)
    x_2 = int(len(obj[0]) * (lane + 1) / lanes)

    y_1 = 0
    y_2 = len(obj)

    return obj[y_1:y_2, x_1:x_2]


# TODO: Move to utils.py
def pixel_intensity(pixel: numpy.ndarray) -> numpy.float64:
    """Calculate the intensity of pixel value.

    :param obj: Pixel value
    :type obj: numpy.ndarray
    :return: Intensity value
    :rtype: numpy.float64
    """
    return 1 - (0.2126 * pixel[0] / 255 + 0.7152 * pixel[1] / 255 + 0.0722 * pixel[2] / 255)


def lane_parser(img: numpy.ndarray, lanes: int, groups: int, baseline: list,
                tolerance: float = 0.1, show: bool = False) -> list:
    """Gel images are parsed and each lane is converted into an array of pixel intensity.

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
    #baseline1, baseline2 = baseline

    final_data = []
    for image in map(crop_to_lane, itertools.repeat(img), range(lanes), itertools.repeat(lanes)):
        final_intensities = []
        for pixel in image:
            intensity = list(map(pixel_intensity, pixel))

            weights = scipy.stats.norm.pdf(numpy.linspace(
                scipy.stats.norm.ppf(0.01),
                scipy.stats.norm.ppf(0.99),
                len(intensity)))

            final_intensities.append(numpy.average(intensity, weights=weights * sum(weights)))

        final_data.append(numpy.array(final_intensities) -
                          numpy.mean(final_intensities))#[baseline1:baseline2]))

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
            matplotlib.pyplot.plot(numpy.arange(len(element)), element, "-")
            matplotlib.pyplot.plot([all_bounds[0], all_bounds[0]], [-0.1, 0.7], "--", color="green")
            matplotlib.pyplot.plot([all_bounds[1], all_bounds[1]], [-0.1, 0.7], "--", color="green")
            matplotlib.pyplot.ylim(-0.1, 0.7)
            matplotlib.pyplot.show()

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

        baseline_x = numpy.arange(len(data[i][bounds[0]:bounds[1]]))
        baseline_x = baseline_x + bounds[0]

        baseline_y = []

        for i in range(len(baseline_x)):
            y = m*baseline_x[i] + b
            baseline_y.append(y)

        baseline_xs.append(baseline_x)
        baseline_ys.append(baseline_y)

    if plot_output == True:

        for i in range(len(data)):
            matplotlib.pyplot.plot(numpy.arange(len(data[i])), data[i], "-")
            matplotlib.pyplot.plot([bounds[0],bounds[0]], [-0.1, 0.7], "--", color="green")
            matplotlib.pyplot.plot([bounds[1],bounds[1]], [-0.1, 0.7], "--", color="green")
            matplotlib.pyplot.plot(baseline_xs[i], baseline_ys[i], "--", color="green")
            matplotlib.pyplot.ylim(-0.1, 0.7)
            matplotlib.pyplot.show()

    trunc_data = []

    for i in range(len(data)):
        d = data[i][bounds[0]:bounds[1]]
        d = d - baseline_ys[i]
        trunc_data.append(d)

    areas = []

    for i in range(len(trunc_data)):
        area_trapz = scipy.integrate.trapz(trunc_data[i])
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

    matplotlib.pyplot.figure(figsize=(2.5, 2.5))
    matplotlib.pyplot.rcParams["font.family"] = "Times New Roman"

    if input_df == True:
        if type(datasets) != pandas.core.frame.DataFrame:
            df = pandas.read_json(datasets)
            matplotlib.pyplot.title(datasets.split(".")[0])
        else:
            df = datasets

    else:

        data = numpy.array(datasets).flatten()
        time = list(timepoints)*int((len(data)/len(timepoints)))
        time = [int(i) for i in time]
        df = pandas.DataFrame({"timepoint":time, "value":data})
        df.to_json(output + ".json")

    def decay(x, a, k):
        return a * numpy.exp(-k * x)

    popt, pcov = scipy.optimize.curve_fit(decay, df.timepoint, df.value, p0=p0)
    perr = numpy.sqrt(numpy.diag(pcov))

    matplotlib.pyplot.plot(df.timepoint,df.value, ".")
    matplotlib.pyplot.ylabel("Normalized \n pixel intensity", fontsize=10)
    matplotlib.pyplot.xlabel("Time (minutes)", fontsize=10)
    x_decay = numpy.linspace(0,1000,1000)
    matplotlib.pyplot.xlim(-1, max(df.timepoint)+5)
    matplotlib.pyplot.ylim(0,)
    matplotlib.pyplot.text(0.5,0.5,"k = " + f"{Decimal(str(popt[1])):.2E}" + "\n" + r' $\pm$ ' + f"{Decimal(str(perr[1])):.2E}" + r' min$^{-1}$', fontsize=10)
    matplotlib.pyplot.plot(x_decay, decay(x_decay, *popt))

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(output + "_decay_curve.svg", dpi=100)
    matplotlib.pyplot.show()

    return popt, perr


def fancy_plotter(dataset, ks, errs, colors, path: str = None, ylim=None,
                  ylabel=None, log=True, labeling=True):

    fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(len(dataset) / 1.6, 5))

    ax.bar(numpy.arange(len(ks)), ks, yerr=[numpy.zeros(len(errs)), errs], color=colors,
           edgecolor="black", capsize=7)

    labels = [i.split(".")[0] for i in dataset]

    if log:
        ax.set_yscale('log')

    ax.set_ylim(ylim)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if labeling:
        matplotlib.pyplot.xticks(range(len(ks)), labels, rotation=90, fontsize=20)
    else:
        matplotlib.pyplot.xticks(range(len(ks)), "", rotation=90, fontsize=20)

    matplotlib.pyplot.ylabel(ylabel)

    if path is not None:
        fig.savefig(path, bbox_inches="tight", dpi=1000)
