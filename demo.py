import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import cppimport
pynslr = cppimport.imp('pynslr')
from pynslr import fit_gaze


# Create some kind of nasty signal
# representing eye movements here
ts = np.arange(0, 10, 1/60.0)
eye = scipy.signal.sawtooth(ts*10)
eyes = np.array([eye, eye]).T

# Add some noise
noise_level = 0.5
signal = eyes + np.random.randn(*eyes.shape)*noise_level

# Plot the signal and ground thruth
plt.plot(ts, signal[:,0], '.')


# Get the regression
# Estimates noise automatically.
# If the noise estimate seems wrong, it can be specified
# by calling
# reconstruction = fit_gaze(ts, signal, structural_error=noise_level, optimize_error=False).
reconstruction = fit_gaze(ts, signal)

# Check noise estimation accuracy
est_noise = np.mean(np.std(reconstruction(ts) - signal, axis=0))
plt.title("True noise %f, estimated noise %f"%(noise_level, est_noise,))

plt.plot(ts, eyes[:,0])

# Plot the reconstruction
# We can also split the independent segments
# This is what one probably wants for saccade/fixation/pursuit analysis
for segment in reconstruction.segments:
    t = np.array(segment.t) # Start and end times of the segment
    x = np.array(segment.x) # Start and end points of the segment
    plt.plot(t, x[:,0], 'ro-', alpha=0.5)

plt.show()
