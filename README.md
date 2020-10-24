# Gelquant

Simple SDS-PAGE/western blot gel quantification software.

## About The Project

Simple band quantification software for protein and DNA gels/blots. Takes an input image which gets cropped by the user. Then splits image into user-specified vertical lanes, which are temporarily saved. These lanes are then converted to numpy arrays, and the average RGB intensity of each row in the image slice is calculated. Intensities are gaussian-weighted so that the middle of the lane contributes more to the average than the outer edges, as other lanes sometimes bleed into the lane of interest. Data for each slice is baselined given a user-specified baselining region. This data is plotted and can be quantified for band intensities.

## Getting Started

### Prerequisites

All provided installation methods in [Installation](###installation) rely on ```pip```. If you want to clone/pull directly from Github you also need ```git```.

Project dependencies: ```matplotlib, numpy, PIL, pandas, scipy```

### Installation

To install a development version, clone this repo and pip install:

```
pip install -e .
```

Or pull directly from this project:

```
pip install git+git://github.com/exterex/gelquant.git
```

## Usage

See notebook-example.ipynb and gel-example.png for details/example usage.

## Roadmap

Work on extending functionality, and streamline usage experience. Have any ideas, create an issue.

## Contributing

Create an issue. Fork this project, provide changes and make a pull request.

## License

This project is licensed under ```MIT```. See [LICENSE](LICENSE) for more information. The original project was developed by ```Joseph Harman``` with several changes by myself, ```Andreas Sagen```. The changes are centered around documentation, formatting (PEP8 compliance), optimization (vectorization, memory reduction), etc.

## Contact

Andreas Sagen - [@andreas_sagen](https://twitter.com/andreas_sagen)

## Acknowledgements

* [jharman25/gelquant](https://github.com/jharman25/gelquant)
* [numpy/numpy](https://github.com/numpy/numpy)
