# kalman-filter
Three Python 3 Kalman Filter implementations for GNSS, Massive-MIMO, and a Combined Solution for East and North coordinates. 

Includes Massive-MIMO data in the .txt files, with the base station at [250,300] so files 245_295.txt are 5m away, 240_290.txt is 10m away etc.

Use the print_variance_mimo.py file to gather the variance of the Massive-MIMO data, and then use generate.py to generate coordinates with this variance at the location of the GNSS coordinates. 

gnss_only_filter.py and combined_filter.py can both have multipath added/removed by comment/uncommenting "self.gnss_multipath(lower_lim, upper_lim)" in the constructor. 

Plots for two q values currently, 10,000 and 0.0000001. 

Define the Massive-MIMO file to use in the init_filter call at the bottom of the filter files.

Any other questions feel free to ask - ryan.c4rter@gmail.com
