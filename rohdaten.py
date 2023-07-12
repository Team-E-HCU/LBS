# -*- coding: utf-8 -*-
"""
Untersuchung der Rohdaten
"""
import numpy as np
import matplotlib.pyplot as plt

def data_preparation(filename):
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
    
    return(accData, orData)

def synchronisierung(accData, orData, interp = False):
    
    if interp == True:
        # Interpolation der Werte aus accData für jeden Zeitstempel orData
        accDatainterp_x = np.interp(orData[:,0], accData[:,0], accData[:,1])
        accDatainterp_y = np.interp(orData[:,0], accData[:,0], accData[:,2])
        accDatainterp_z = np.interp(orData[:,0], accData[:,0], accData[:,3])
        
        acc_time = orData[:,0]
        acc_x = accDatainterp_x
        acc_y = accDatainterp_y
        acc_z = accDatainterp_z
    else:
        acc_time = accData[:,0]
        acc_x = accData[:,1]
        acc_y = accData[:,2]
        acc_z = accData[:,3]
        
        

    o_time = orData[:,0]
    o_alpha = orData[:,1]
    o_beta = orData[:,2]
    o_gamma = orData[:,3]
    
    # Zeitstemepl auf 0 skalieren (Bei 0 Anfangen)
    acc_time -= acc_time[0]
    o_time -= o_time[0]
    
    return(acc_time, acc_x, acc_y, acc_z, o_time, o_alpha, o_beta, o_gamma)

def smoothing(data, neighbours):
    new_data = []
    for i in range(0+neighbours,len(data)-neighbours,1):
        new_data.append( np.mean(data[i-neighbours:i+neighbours]) )
    return(new_data)

def reduktion(time, data, length = 200):
    """
    Diese Funktion bringt eine Datensatz auf eine vorgegebene Anzahl an Messwerten.
    Dafür wird die lineare Interpolation benutzt

    Parameters
    ----------
    time : Datensatz mit Zeitstempeln als np.array.
    data : Datensatz mit Messdaten als np.array.
    length : Gewünschte Länge des neuen Datensatzes als Integer (Standard = 4)

    Returns
    -------
    new_time : np.array mit neuen Zeitstempeln
    new_data : np.array mit interpolierten Messwerten

    """
    
    time_start = time[0]
    time_end = time[-1]
    
    
    delta_time = (time_end - time_start) / (length-1)
    new_time = [i*delta_time for i in range(0,length,1)]
    
    new_data = np.interp(new_time, time, data)
    return(new_time, new_data)

def berechne_korrelation(file, mittel, plot = False):
    """
    Berechnen der Korrelation eines gegeben Datsensatzes mit einem Mittelwert Datensatz


    """
    accData, orData = data_preparation(file)
    acc_time, acc_x, acc_y, acc_z, o_time, o_alpha, o_beta, o_gamma = synchronisierung(accData, orData, True)
    new_time, data = reduktion(acc_time, acc_x, anzahl_zu_betrachtender_werte)
    data = smoothing(data, glaettung)
    
    if plot == True:
        plt.plot(data)
        plt.plot(mittel, label = "Mittel")
        plt.legend()
        plt.show()

    
    correlation_matrix = np.corrcoef(mittel, data)
    correlation_coefficient = correlation_matrix[0, 1]
    return(correlation_coefficient)

#%%
"""
MAIN
"""
filelist_kreise = [f"data/Kreise/SensorDataKreis ({i}).txt" for i in range(0,15)]   # Maximal 250 
filelist_kreuze = [f"data/Kreuze/SensorDataKreuz ({i}).txt" for i in range(0,250)]
filelist_andere = [f"data/Andere/SensorData ({i}).txt" for i in range(0,24)]
anzahl_zu_betrachtender_werte = 200
glaettung = 10


mittel = np.array([float(0) for i in range(anzahl_zu_betrachtender_werte-2*glaettung)])

for file in filelist_kreise:
    accData, orData = data_preparation(file)
    acc_time, acc_x, acc_y, acc_z, o_time, o_alpha, o_beta, o_gamma = synchronisierung(accData, orData, True)
    

    new_time, data = reduktion(acc_time, acc_x, anzahl_zu_betrachtender_werte)
    data = smoothing(data, glaettung)
    
    
    plt.plot(data,'-', linewidth=.8)
    mittel += data
    
mittel = mittel/len(filelist_kreise)
plt.plot(mittel,'-',linewidth=4, color='Black')
plt.show()

for file in filelist_kreise:
    print("Korrelationskoeffizient:", berechne_korrelation(file, mittel))
