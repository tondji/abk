# abk
This repository contains the Python code that generates the figures in the paper "Adaptive Bregman-Kaczmarz: An Approach to Solve Linear Inverse Problems with Independent Noise Exactly".

##### Authors:
- Lionel Tondji  (<tngoupeyou@aimsammi.org>)
- Idriss Tondji  (<itondji@aimsammi.org>)
- Dirk Lorenz    (<d.lorenz@tu-braunschweig.de>)

Contents
--------

##### Drivers (run these to generate figures):

  example_overdetermined.ipynb                  notebook to generate figure 2 and 3
  exmaple_variation_block_number.ipynb          notebook to generate figure 4
  example_CT.ipynb                              notebook to generate figure 5 and 6

##### Routines called by the drivers:
	tools.py                      Python packages containing functions like The adaptive Bregman-Kaczmarz, myphantom, soft_skrinkage.

The myphantom function generates the data for the CT example.
