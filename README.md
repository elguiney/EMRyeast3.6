# YMPy


 ![alt text](https://github.com/elguiney/YMPy/blob/master/YMPy-logo.png "Yeast Morphology Pypeline")
 ## **YMPY** is a  **Y**east **M**orphology **Py**peline



Image processing pipeline for quantitative analysis of yeast (_Saccharomyces cerevisiae_) microscopy.
Uses edge detection based image segmentation combined with yeast specific heuristics to substantially outperform freely available but general purpose analysis tools (eg [CellProfiler](http://cellprofiler.org/)).
Written in Python3

#### Requires:
- **Python3.6** or greater

The standard scientific python stack:
- numpy
- scipy
- pandas
- matplotlib
- scikit-image
- Ipython and ipywidgets
- Jupyter Notebook

For the best experience, it is highly reccomended to install python3 and its standard
dependences with [Anaconda](https://www.anaconda.com/distribution/)

For use with DeltaVision microscopes and the .dv file format, requires
- mrcfile ([github site](https://github.com/ccpem/mrcfile))

#### Features:
- Takes advantage of bright-field microscopy to find accurate cell outlines without requiring a fluorescent marker
- Edges detection via laplace of gaussian transformation, followed by heuristic based morphology cleanup yielding high quality cell outlines suitable for precise protein localization and cell morphology measuremnts
- integrates a blinded quality control step to ensure all cells used for further analysis are segmented and measured at publication quality
- Built from modular core functions, combined into simple pipelines to process a folder of images in a single batch
- All code in a pipeline is run from a single Jupyter notebook, ensuring complete documentation and reproducibility for publication grade analyses.
