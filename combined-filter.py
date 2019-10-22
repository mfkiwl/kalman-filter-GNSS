import numpy as np
import sys
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random

# reads file, outputs kalman filter instance with measurements, deviation, and q values passed
def init_filter(gnss_file_name, mimo_file_name, q, plot):
    try:
        file_measurements = open(gnss_file_name, "r")
        lines = file_measurements.readlines()
    except IOError:
        print ("Could not find \'"+file_name+"\'")
        sys.exit()
    gnss_measurements = []
    gnss_deviation = []
    for i in range(24, len(lines)):
        line = lines[i].split(" ")
        line = [x for x in line if x]
        try:
            x = float(line[2].strip())
            y = float(line[3].strip())
            sdx = float(line[7].strip())
            sdy = float(line[8].strip())
        except:
            print("Wasn't expecting this file format.\n")
        coords = []
        std_dev = []
        coords.append(x)
        coords.append(y)
        std_dev.append(sdx)
        std_dev.append(sdy)
        gnss_measurements.append(coords)
        gnss_deviation.append(std_dev)

    try:
        file_measurements = open(mimo_file_name, "r")
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
        except:
            print("Wasn't expecting this file format.\n")
        coords = []
        coords.append(x)
        coords.append(y)
        mimo_measurements.append(coords)

    return kalman_filter(mimo_measurements, gnss_measurements, gnss_deviation, q, plot)

# Kalman filter class
class kalman_filter(object):
    def __init__(self, mimo_measurements, gnss_measurements, gnss_deviation, q, plot):
        self.gnss_measurements = np.array(gnss_measurements)
        self.mimo_measurements = np.array(mimo_measurements)
        self.q = q
        self.plot = plot

        self.mimo_measurements = self.mimo_measurements - np.mean(self.gnss_measurements, axis=0)
        self.gnss_measurements = self.gnss_measurements - np.mean(self.gnss_measurements, axis=0)

        # comment/uncomment this to add multipath
        self.gnss_multipath(lower_lim=0.5, upper_lim=1.5)

        self.mimo_variance = np.sum(np.square(self.mimo_measurements - np.mean(self.mimo_measurements, axis=0)), axis=0)/len(self.mimo_measurements)

        self.H_matrix = np.concatenate((np.identity(n=2),np.identity(n=2)), axis=0)
        self.A_matrix = np.identity(n=2)
        self.gnss_variance = np.square(np.array(gnss_deviation))

        # initial state, pcm, and covariance matrix
        last_state = np.array([(self.gnss_measurements[0][0]+self.mimo_measurements[0][0])/2, (self.gnss_measurements[0][1]+self.mimo_measurements[0][1])/2])

        last_pcm = np.diag(np.var(np.concatenate((self.gnss_measurements, self.mimo_measurements), axis=0), axis=0))
        self.covar = np.diag(np.concatenate((self.gnss_variance[1], self.mimo_variance)))

        # start iteration
        self.iterate(last_pcm, last_state)

    def iterate(self, last_pcm, last_state):
        x = np.zeros((len(self.gnss_measurements), 2))
        y = np.zeros((len(self.gnss_measurements), 2))
        gxm = np.zeros((len(self.gnss_measurements), 2))
        gym = np.zeros((len(self.gnss_measurements), 2))
        mxm = np.zeros((len(self.gnss_measurements), 2))
        mym = np.zeros((len(self.gnss_measurements), 2))

        for k in range(1, len(self.gnss_measurements)):
            # predicted pcm and state
            pred_state = np.dot(self.A_matrix, last_state) # Xkp = AXk-1 + BU
            pred_pcm = np.dot(np.dot(self.A_matrix, last_pcm), self.A_matrix.T) + np.full((2,2), self.q) # Pkp = APk-1 At + Qk

            # calculate gain and current state
            gain = np.dot(np.dot(pred_pcm, self.H_matrix.T), np.linalg.inv((np.dot(np.dot(self.H_matrix, pred_pcm), self.H_matrix.T) + self.covar))) # K = PkpH / H Pkp Ht + R

            curr_state = pred_state + np.dot(gain, (np.concatenate((self.gnss_measurements[k],self.mimo_measurements[k]),axis=0)) - np.dot(self.H_matrix, pred_state)) # X = Pkp + K[Yk - Pkp]

            # uncomment this to print the current state each epoch
            # print("State: "+str(curr_state)+"\n")

            # current becomes previous
            last_pcm = np.dot(np.identity(n=2) -  np.dot(gain,self.H_matrix), pred_pcm) # Pk-1 = (I - KH) Pkp
            last_state = curr_state # Xk-1 = Xk
            self.covar = np.diag(np.concatenate((self.gnss_variance[k], self.mimo_variance)))

            # save xy of each epoch for plotting
            x[k] = [curr_state[0], k]
            y[k] = [curr_state[1], k]
            gxm[k] = [self.gnss_measurements[k][0], k]
            gym[k] = [self.gnss_measurements[k][1], k]
            mxm[k] = [self.mimo_measurements[k][0], k]
            mym[k] = [self.mimo_measurements[k][1], k]
        if self.plot:
            plt.figure(figsize=(15,5))
            plt.title("North Measurement, Q value: "+str(self.q))
            plt.ylabel("Error (m)")
            plt.xlabel("Epochs (30s)")
            plt.plot(np.hsplit(x[1:], 2)[1], np.hsplit(x[1:], 2)[0], label="NF, STD: "+str(round(np.sqrt(np.var(np.hsplit(x[1:], 2)[0])), 5)), c='red', zorder=3)
            plt.scatter(np.hsplit(gxm[1:], 2)[1], np.hsplit(gxm[1:], 2)[0], label="NG, STD: "+str(round(np.sqrt(np.var(np.hsplit(gxm[1:], 2)[0])),5))+" ("+str(round(np.sqrt(self.covar[0][0]), 5))+") m", s=15, c='green', zorder=2)
            plt.scatter(np.hsplit(mxm[1:], 2)[1], np.hsplit(mxm[1:], 2)[0], label="NM, STD: "+str(round(np.sqrt(np.var(np.hsplit(mxm[1:], 2)[0])),5))+" ("+str(round(np.sqrt(self.covar[2][2]), 5))+") m", s=1, c='blue', zorder=1)
            plt.legend(loc='upper right')
            axes = plt.gca()
            axes.set_ylim([-0.25, 2])
            plt.show()

            plt.figure(figsize=(15,5))
            plt.title("East Measurement, Q value: "+str(self.q))
            plt.ylabel("Error (m)")
            plt.xlabel("Epochs (30s)")
            plt.plot(np.hsplit(y[1:], 2)[1], np.hsplit(y[1:], 2)[0], label="EF, STD: "+str(round(np.sqrt(np.var(np.hsplit(y[1:], 2)[0])), 5)), c='red', zorder=3)
            plt.scatter(np.hsplit(gym[1:], 2)[1], np.hsplit(gym[1:], 2)[0], label="EG, STD: "+str(round(np.sqrt(np.var(np.hsplit(gym[1:], 2)[0])),5))+" ("+str(round(np.sqrt(self.covar[1][1]), 5))+") m", s=15, c='green', zorder=2)
            plt.scatter(np.hsplit(mym[1:], 2)[1], np.hsplit(mym[1:], 2)[0], label="EM, STD: "+str(round(np.sqrt(np.var(np.hsplit(mym[1:], 2)[0])),5))+" ("+str(round(np.sqrt(self.covar[3][3]), 5))+") m", s=1, c='blue', zorder=1)
            plt.legend(loc='upper right')
            axes = plt.gca()
            axes.set_ylim([-0.25, 2])
            plt.show()

    def gnss_multipath(self, lower_lim, upper_lim):
        for i in range(0, 514):
            if i % 6 == 0:
                self.gnss_measurements[i] += random.uniform(lower_lim, upper_lim)
        for i in range(514, 1027):
            if i % 3 == 0:
                self.gnss_measurements[i] += random.uniform(lower_lim, upper_lim)
        for i in range(1027, 1542):
            if i % 4 == 0:
                self.gnss_measurements[i] += random.uniform(lower_lim, upper_lim)
        for i in range(1542, 2055):
            if i % 2 == 0:
                self.gnss_measurements[i] += random.uniform(lower_lim, upper_lim) #6402640
        for i in range(2055, 2569):
            if i % 12 == 0:
                self.gnss_measurements[i] += random.uniform(lower_lim, upper_lim)
        for i in range(2569, 3082):
            if i % 4 == 0:
                self.gnss_measurements[i] += random.uniform(lower_lim, upper_lim)

# main
kalman_filter2 = init_filter(gnss_file_name="baserover.pos", mimo_file_name="245_295.txt", q=10000, plot=True)
kalman_filter1 = init_filter("baserover.pos", mimo_file_name="245_295.txt", q=0.0000001, plot=True)
