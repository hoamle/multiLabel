# Multi-label image classification

Source code for paper [Fully automated multi-label image annotation by Convolutional Neural Network and Adaptive Thresholding](http://dl.acm.org/citation.cfm?id=3011118) at [SoICT'16](https://soict.hust.edu.vn/~soict2016/)

*NB: The code was intended for internal development, we have not refactored to make it ready-to-run in user-friendly fashion yet. Sorry for the inconvenience.*

In essence, the "`main`" code is as in `./src/fullmodel_msrcv2_fixed.py`. The other lengthy sources are just embarrassingly copy-and-paste copycats with appropriate amending from the `main` code (Yes, we were total novices at source code management and collaboration back then)

* `cd` to `./src/` before running the code
* `t_train` takes a while to generate, and will be stored in `./src/t/` to reuse. 
* all evaluation metrics to be stored in `./src/metrics/`
* snapshots of scoring module to be stored in `./snapshot_models/`