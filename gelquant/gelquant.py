"""Source code for Gelquant."""

# Standard Python Modules
import decimal
import itertools

# Third Party Python Modules
import matplotlib.pyplot as plt
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


def pixel_intensity(pixel: numpy.ndarray) -> numpy.float64:
    """Calculate the RGB intensity of pixel value.

    :param obj: Pixel value
    :type obj: numpy.ndarray
    :return: Intensity value
    :rtype: numpy.float64
    """
    return 1 - (0.2126 * pixel[0] / 255 + 0.7152 * pixel[1] / 255 + 0.0722 * pixel[2] / 255)


def lane_parser(img: numpy.ndarray, lanes: int, groups: int,
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
    :param tolerance: Tolerance value, defaults to 0.1
    :type tolerance: float, optional
    :param show: Whether to show plot or not, defaults to False
    :type show: bool, optional
    :return: Return processed intensity data and bounds
    :rtype: list
    """
    final_data = []
    for image in map(crop_to_lane, itertools.repeat(img), range(lanes), itertools.repeat(lanes)):
        final_intensities = []
        for pixel in image:
            intensity = list(map(pixel_intensity, pixel))

            weights = scipy.stats.norm.pdf(numpy.linspace(
                scipy.stats.norm.ppf(0.01),
                scipy.stats.norm.ppf(0.99),
                len(intensity)))

            final_intensities.append(numpy.average(
                intensity, weights=weights * sum(weights)))

        final_data.append(numpy.array(final_intensities) -
                          numpy.mean(final_intensities))

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
            plt.plot([all_bounds[0], all_bounds[0]],
                     [-0.1, 0.7], "--", color="green")
            plt.plot([all_bounds[1], all_bounds[1]],
                     [-0.1, 0.7], "--", color="green")
            plt.ylim(-0.1, 0.7)
            plt.show()

    return final_data, all_bounds[0]


def area_integrator(data: list, bounds: list, groups: int,
                    show: bool = False, percentages: bool = True) -> numpy.ndarray:
    """Capture the peak areas as of the first band in each experiment.

    :param data: List of individual lanes RGB intensity data
    :type data: list
    :param bounds: Initial peak boundaries
    :type bounds: list
    :param groups: Amount of different proteins
    :type groups: int
    :param show: Whether to show plot or not, defaults to False
    :type show: bool, optional
    :param percentages: Whether to return peak areas as a percentage or not, defaults to True
    :type percentages: bool, optional
    :return: Sorted area
    :rtype: numpy.ndarray
    """
    baseline_xs, baseline_ys = [], []
    for element in data:
        x_1 = bounds[0]
        x_2 = bounds[1]
        y_1 = element[bounds[0]]
        y_2 = element[bounds[1]]

        m = (y_2 - y_1) / (x_2 - x_1)
        b = y_1 - m * x_1

        baseline_x = numpy.arange(len(element[bounds[0]:bounds[1]]))
        baseline_x = baseline_x + bounds[0]

        baseline_y = m * baseline_x + b

        baseline_xs.append(baseline_x)
        baseline_ys.append(baseline_y)

    if show:
        for i, element in enumerate(data):
            plt.plot(numpy.arange(len(element)), element, "-")
            plt.plot([bounds[0], bounds[0]], [-0.1, 0.7], "--", color="green")
            plt.plot([bounds[1], bounds[1]], [-0.1, 0.7], "--", color="green")
            plt.plot(baseline_xs[i], baseline_ys[i], "--", color="green")
            plt.ylim(-0.1, 0.7)
            plt.show()

    trunc_data = [element[bounds[0]:bounds[1]] - baseline_ys[i]
                  for i, element in enumerate(data)]

    areas = [scipy.integrate.trapz(element) for element in trunc_data]

    sorted_areas = []
    for i in range(groups):
        index = int(len(data) / groups)
        group = areas[i * index:i * index + index]
        sorted_areas.append(group)

    sorted_areas = [item for sublist in sorted_areas for item in sublist]

    if percentages:
        return sorted_areas / sorted_areas[0]

    return sorted_areas


def summary_data(datasets: numpy.ndarray, timepoints: str = "", fp: str = "",
                 p_0: list = None, input_df: bool = False) -> None:
    """Make a summeryplot from timepoints and RGB pixel intensity.

    :param datasets: RGB pixel intensity
    :type datasets: numpy.ndarray
    :param timepoints: Timepoints to be used in summary, defaults to ""
    :type timepoints: str, optional
    :param fp: Filename for summary data image or json output, defaults to ""
    :type fp: str, optional
    :param p_0: Initial guess for the parameters for curve_fit(), defaults to [7, 0.2]
    :type p_0: list, optional
    :param input_df: Whether input is dataframe or not, defaults to False
    :type input_df: bool, optional
    :return: Show an image summarizing the data, and if output is defined saves the image to disk.
    :rtype: None
    """
    if p_0 is None:
        p_0 = [7, 0.2]

    plt.figure(figsize=(2.5, 2.5))
    plt.rcParams["font.family"] = "Times New Roman"

    if input_df:
        if isinstance(datasets, pandas.core.frame.DataFrame):
            df = datasets
        else:
            df = pandas.read_json(datasets)
            plt.title(datasets.split(".")[0])
    else:
        data = numpy.array(datasets).flatten()
        time = list(timepoints) * int((len(data) / len(timepoints)))
        time = [int(i) for i in time]
        df = pandas.DataFrame({"timepoint": time, "value": data})
        df.to_json(fp.split(".")[0] + ".json")

    def decay(x, a, k):
        return a * numpy.exp(-k * x)

    popt, pcov = scipy.optimize.curve_fit(
        decay, df.timepoint, df.value, p0=p_0)
    perr = numpy.sqrt(numpy.diag(pcov))

    plt.plot(df.timepoint, df.value, ".")
    plt.ylabel("Normalized \n pixel intensity", fontsize=10)
    plt.xlabel("Time (minutes)", fontsize=10)
    x_decay = numpy.linspace(0, 1000, 1000)
    plt.xlim(-1, max(df.timepoint) + 5)
    plt.ylim(0,)
    plt.text(
        0.5,
        0.5,
        "k = " +
        f"{decimal.Decimal(str(popt[1])):.2E}" +
        "\n" +
        r' $\pm$ ' +
        f"{decimal.Decimal(str(perr[1])):.2E}" +
        r' min$^{-1}$',
        fontsize=10)
    plt.plot(x_decay, decay(x_decay, *popt))

    plt.tight_layout()
    plt.savefig(fp, dpi=100)
    plt.show()

    return popt, perr


def fancy_plotter(dataset, ks, errs, colors, fp: str = None, ylim=None,
                  ylabel=None, log=True, labeling=True):

    fig, ax = plt.subplots(1, 1, figsize=(len(dataset) / 1.6, 5))

    ax.bar(numpy.arange(len(ks)), ks, yerr=[numpy.zeros(len(errs)), errs], color=colors,
           edgecolor="black", capsize=7)

    labels = [i.split(".")[0] for i in dataset]

    if log:
        ax.set_yscale('log')

    ax.set_ylim(ylim)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if labeling:
        plt.xticks(range(len(ks)), labels, rotation=90, fontsize=20)
    else:
        plt.xticks(range(len(ks)), "", rotation=90, fontsize=20)

    plt.ylabel(ylabel)

    if fp is not None:
        fig.savefig(fp, bbox_inches="tight", dpi=1000)
