import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from pynslr import nslr2d


# Create some kind of nasty signal
# representing eye movements here
ts = np.arange(0, 10, 1/60.0)
eye = scipy.signal.sawtooth(ts*10)
eyes = np.array([eye, eye]).T

# Add some noise
noise_level = 0.5
signal = eyes + np.random.randn(*eyes.shape)*noise_level

# Plot the signal and ground thruth
plt.plot(ts, eyes[:,0])
plt.plot(ts, signal[:,0], '.')


# Get the regression
split_penalty = 4.0 # This is generally a good value for eye movement signals
reconstruction = nslr2d(ts, signal, noise_level, split_penalty)

# Plot the reconstruction
plt.plot(ts, reconstruction(ts)[:,0])

# We can also split the independent segments
# This is what one probably wants for saccade/fixation/pursuit analysis
# NOTE: The segments can look a bit weird as they are overlapping,
# in the segmentation there are actually two values for the midpoints.
# The reconstruction averages those
for segment in reconstruction.segments:
    t = np.array(segment.t) # Start and end times of the segment
    x = np.array(segment.x) # Start and end points of the segment
    plt.plot(t, x[:,0], 'ro-', alpha=0.3)

plt.show()
