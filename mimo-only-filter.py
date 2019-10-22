import numpy as np
import sys
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# reads file, outputs kalamn filter instance with mimo_measurements, deviation, and q values passed
def init_filter(file_name, q, plot):
    try:
        file_measurements = open(file_name, "r")
        lines = file_measurements.readlines()
    except IOError:
        print ("Could not find \'"+file_name+"\'")
        sys.exit()
    mimo_measurements = []
    for i in range(0, len(lines)):
        line = lines[i].split(" ")
        line = [x for x in line if x]
        try:
            x = float(line[0].strip())
            y = float(line[1].strip())
            #z = float(line[2].strip())
        except:
            print("Wasn't expecting this file format.\n")
        coords = []
        coords.append(x)
        coords.append(y)
        #coords.append(z)
        mimo_measurements.append(coords)

    return kalman_filter(mimo_measurements, q, plot)

# Kalman filter class
class kalman_filter(object):
    def __init__(self, mimo_measurements, q, plot):
        self.mimo_measurements = np.array(mimo_measurements)
        self.q = q
        self.plot = plot
        self.mimo_measurements = self.mimo_measurements - np.mean(self.mimo_measurements, axis=0)

        self.A_matrix = np.identity(n=2)
        self.variance = np.sum(np.square(self.mimo_measurements - np.mean(self.mimo_measurements, axis=0)), axis=0)/len(self.mimo_measurements)

        # initial state, pcm, and covariance matrix
        last_state = np.array([[self.mimo_measurements[0][0]], [self.mimo_measurements[0][1]]])
        last_pcm = np.diag(np.var(self.mimo_measurements, axis=0))
        self.covar = np.diag(self.variance)

        # start iteration
        self.iterate(last_pcm, last_state)

    def iterate(self, last_pcm, last_state):
        x = np.zeros((len(self.mimo_measurements), 2))
        y = np.zeros((len(self.mimo_measurements), 2))
        gx = np.zeros((len(self.mimo_measurements), 2))
        gy = np.zeros((len(self.mimo_measurements), 2))

        for k in range(1, len(self.mimo_measurements)):
            # predicted pcm and state

            pred_state = np.dot(self.A_matrix, last_state) # Xkp = AXk-1 + BU
            pred_pcm = np.dot(np.dot(self.A_matrix, last_pcm), self.A_matrix.T) + np.full((2,2), self.q) #self.q[k]) # Pkp = APk-1 At + Qk

            # calculate gain and current state
            gain = np.dot(np.dot(pred_pcm, np.identity(n=2).T), np.linalg.inv((np.dot(np.dot(np.identity(n=2), pred_pcm), np.identity(n=2).T) + self.covar))) # K = PkpH / H Pkp Ht + R
            curr_state = pred_state + np.dot(gain, (self.mimo_measurements[k] - pred_state.T).T) # X = Pkp + K[Yk - Pkp]

            # uncomment this to print the current state each epoch
            # print("State: "+str(curr_state)+"\n")

            # current becomes previous
            last_pcm = np.dot(np.identity(n=2) -  np.dot(gain,self.A_matrix), pred_pcm) # Pk-1 = (I - KH) Pkp
            last_state = curr_state # Xk-1 = Xk

            # save xyz of each epoch for plotting
            x[k] = [curr_state[0], k]
            y[k] = [curr_state[1], k]
            gx[k] = [self.mimo_measurements[k][0], k]
            gy[k] = [self.mimo_measurements[k][1], k]

        if self.plot:
            plt.figure(figsize=(15,5))
            plt.title("North Measurement, Q value: "+str(self.q))
            plt.ylabel("Error (m)")
            plt.xlabel("Epochs (30s)")
            plt.plot(np.hsplit(x[1:], 2)[1], np.hsplit(x[1:], 2)[0], label="NF, STD: "+str(round(np.sqrt(np.var(np.hsplit(x[1:], 2)[0])), 5)), c='red', zorder=2)
            plt.scatter(np.hsplit(gx[1:], 2)[1], np.hsplit(gx[1:], 2)[0], label="NM, STD: "+str(round(np.sqrt(np.var(np.hsplit(gx[1:], 2)[0])), 5))+" ("+str(round(np.sqrt(self.covar[0][0]), 4))+") m", s=12, c='blue', zorder=1)
            plt.legend(loc='upper right')
            axes = plt.gca()
            axes.set_ylim([-0.025, 0.025])
            plt.show()

            plt.figure(figsize=(15,5))
            plt.title("East Measurement, Q value: "+str(self.q))
            plt.ylabel("Error (m)")
            plt.xlabel("Epochs (30s)")
            plt.plot(np.hsplit(y[1:], 2)[1], np.hsplit(y[1:], 2)[0], label="EF, STD: "+str(round(np.sqrt(np.var(np.hsplit(y[1:], 2)[0])), 5)), c='red', zorder=2)
            plt.scatter(np.hsplit(gy[1:], 2)[1], np.hsplit(gy[1:], 2)[0], label="EM, STD: "+str(round(np.sqrt(np.var(np.hsplit(gy[1:], 2)[0])), 5))+" ("+str(round(np.sqrt(self.covar[1][1]), 4))+") m", s=12, c='blue', zorder=1)
            plt.legend(loc='upper right')
            axes = plt.gca()
            axes.set_ylim([-0.025, 0.025])
            plt.show()

# main
kalman_filter2 = init_filter(file_name="245_295.txt", q=10000, plot=True)
kalman_filter1 = init_filter(file_name="245_295.txt", q=0.0000001, plot=True)
