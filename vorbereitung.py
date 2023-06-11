"""
Verarbeitung der ausgelesenen Sensordaten aus:
    https://team-e-hcu.github.io/Sensor-Data-Collection/
"""

import numpy as np
import matplotlib.pyplot as plt

filename = "sensorData_L_in_XY.txt"

#%% Vorbereitung

# Textdatei einlesen und splitten in zwei str variablen
with open(filename) as data:
    accData = ""
    orData = ""
    dtype = "ACC"
    for row in data:
        if row == "BREAK\n":
            dtype = "OR"
        # Zeilen mit Text aussortieren und Rest in entsprechende Variablen schreiben
        try:
            int(row[0])
            if dtype == "ACC":
                accData += row
            else:
                orData += row
        except:
            None

# Umwandlung Beschleunigungsdaten in numpy array
lines = accData.split('\n')[:-1]
accData = [line.split(',') for line in lines]
accData = [[float(value) for value in line] for line in accData]
accData = np.array(accData)

# Umwandlung Orientierungsdaten in numpy array
lines = orData.split('\n')[:-1]
orData = [line.split(',') for line in lines]
orData = [[float(value) for value in line] for line in orData]
orData = np.array(orData)

#%% Plots der Rohdaten anfertigen

# Einzelwerte extrahieren
acc_time = accData[:,0]
acc_x = accData[:,1]
acc_y = accData[:,2]
acc_z = accData[:,3]

o_time = orData[:,0]
o_alpha = orData[:,1]
o_beta = orData[:,2]
o_gamma = orData[:,3]

# Resample Zeit (Start bei 0)
acc_time -= acc_time[0]
o_time -= o_time[0]

# Plot Beschleunigung
fig = plt.plot(acc_time, acc_x, label="x-Achse")
fig = plt.plot(acc_time, acc_y, label="y-Achse")
fig = plt.plot(acc_time, acc_z, label="z-Achse")
plt.xlabel("Zeit [ms]")
plt.ylabel("Beschleunigung [m/s²]")
plt.legend()
plt.savefig("Beschleunigung.png")
plt.show()

# Plot Orientierung
fig = plt.plot(o_time, o_alpha, label="Alpha")
fig = plt.plot(o_time, o_beta, label="Beta")
fig = plt.plot(o_time, o_gamma, label="Gamma")
plt.xlabel("Zeit [ms]")
plt.ylabel("Orientierung [°]")
plt.legend()
plt.savefig("Orientierung.png")
plt.show()


