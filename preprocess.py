import numpy as np
import signal_function as sf

filename = "1n_angle_zeroADC"
prefix = ""
suffix = "_rms"

before = "./emg/" + filename + ".txt"
after = "./emg/" + prefix + filename + suffix + ".txt"
WINDOW_SIZE = 10

raw = np.loadtxt(before, delimiter=" ")

d0 = sf.MovRMS(WINDOW_SIZE)
d1 = sf.MovRMS(WINDOW_SIZE)
d2 = sf.MovRMS(WINDOW_SIZE)
d3 = sf.MovRMS(WINDOW_SIZE)

data = []

for row in raw:
    d0.update(row[0])
    d1.update(row[1])
    d2.update(row[2])
    d3.update(row[3])
    #angle = round(row[4] / 11.3778, 0)
    data.append( [d0.rms, d1.rms, d2.rms, d3.rms, row[4]] )

    # data.append( [row[0]-2048, row[1]-2048, row[2]-2048, row[3]-2048, row[4] ] )

np.savetxt(after, data, '%d')