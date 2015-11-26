#Implementation of the Mean Shift Algorithm#
***gen_raw_img.cpp***
- Generates a 240x240 boolean image "dots.bmp" with random dots. The density of the dots is highest at (100,100) and decreases further from that point.
***mean_shift.cpp***
- The original mean shift algorithm for finding the highest density region of the dots image. It starts from a random location and converges to a local maximum by following the direction of gradient.
***mean_shift_tracker.cpp***
- This file implements the mean shift object tracking algorithm. We use hue histogram as feature.