# -*- coding: utf-8 -*-
import numpy as np

"""Parameter"""
glaettung = 8
gewicht_acc = 2/3
gewicht_or = 1/3
grenzwert_c_kreis = 0.68
grenzwert_c_kreuz = 0.55

"""Funktionen"""
def split_data(data):
    accData = ""
    orData = ""
    dtype = "ACC"
    for row in data.split("\n"):
        #print(row)
        if row == "BREAK":
            dtype = "OR"
        # Zeilen mit Text aussortieren und Rest in entsprechende Variablen schreiben
        try:
            int(row[0])
            if dtype == "ACC":
                accData += row + "\n"
            else:
                orData += row + "\n"
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


def process_file(file, glaettung = 8):
    """
    Included functions: split_data | synchronisierung | smoothing

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    glaettung : TYPE, optional
        DESCRIPTION. The default is 8.

    Returns
    -------
    Extracted Values as single arrays

    """
    # Daten einlesen
    accData, orData = split_data(file)
    acc_time, acc_x, acc_y, acc_z, o_time, o_alpha, o_beta, o_gamma = synchronisierung(accData, orData, False)
    
    
    # Glättung des Datensatzes um das Gewicht von Ausreißern zu minimieren
    acc_x = smoothing(acc_x, glaettung)
    acc_y = smoothing(acc_y, glaettung)
    acc_z = smoothing(acc_z, glaettung)
    
    # Skalierung auf -1 bis 1
    acc_x = (acc_x - min(acc_x)) / ( max(acc_x) - min(acc_x) )
    acc_y = (acc_y - min(acc_y)) / ( max(acc_y) - min(acc_y) )
    acc_z = (acc_z - min(acc_z)) / ( max(acc_z) - min(acc_z) )
     
    return(acc_x, acc_y, acc_z, o_alpha, o_beta, o_gamma)

#%% Funktionen zur Berechnung der Korrelation

def phasenkorrektur(timestamps, vergleichsdaten, neumessung):
    
    try:
        idx_max_mittel = np.where(vergleichsdaten == np.max(vergleichsdaten))
        idx_max = np.where(neumessung == np.max(neumessung))
        
        idx_min_mittel = np.where(vergleichsdaten == np.min(vergleichsdaten))
        idx_min = np.where(neumessung == np.min(neumessung))
        
        idx_diff_max = idx_max_mittel[0] - idx_max[0]
        idx_diff_min = idx_min_mittel[0] - idx_min[0]
    
        mean_diff = int((idx_diff_max + idx_diff_min)/2)
        timestamps += mean_diff
    except:
        None # Bedeutet Differenz ist bereits Minimal

    return(timestamps)

def korrelationskoeffizient(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_dev_x = (sum((xi - mean_x)**2 for xi in x) / n)**0.5
    std_dev_y = (sum((yi - mean_y)**2 for yi in y) / n)**0.5
    correlation_coefficient = covariance / (std_dev_x * std_dev_y)
    
    return correlation_coefficient


def berechne_korrelation(neumessung, vergleichsmessung, __plots = False):
    
    timestamps = np.array([i for i in range(len(neumessung))])
    timestamps_mittel = np.array([i for i in range(len(vergleichsmessung))])
    
    # Verschieben  
    timestamps = phasenkorrektur(timestamps, vergleichsmessung, neumessung)
    
    # Bis wohin verschoben?
    # Timestamps dürfen nicht negativ sein
    if timestamps[0] < 0:
        startwert = 0
    else:
        startwert = timestamps[0]
    endwert = timestamps[-1]
    
    
    timestamps_mittel = timestamps_mittel[startwert:endwert]
    vergleichsmessung = vergleichsmessung[startwert:endwert]
 
    neumessung = np.interp(timestamps_mittel, timestamps, neumessung)
    
    # Korrelation berechnen
    try: 
            correlation_coefficient = korrelationskoeffizient(vergleichsmessung, neumessung)
    except:
        print("Berechnung der Korrelation nicht möglich")
        None
    

    return(correlation_coefficient)

def umwandlung_orientierungsdaten(data):    
    data_diff = np.diff(data)
    data_diff[data_diff < -180] += 360
    data_diff[data_diff > 180] -= 360
    data_smoothed = np.cumsum(data_diff)
    return(data_smoothed)


def korrelation_orientierung(vergleichsmessung, neumessung):
    
    neumessung = umwandlung_orientierungsdaten(neumessung)
    vergleichsmessung = umwandlung_orientierungsdaten(vergleichsmessung)
    
    neumessung = smoothing(neumessung, glaettung)[30:]
    vergleichsmessung = smoothing(vergleichsmessung, glaettung)[30:]
    
    # Skalierung auf -1 bis 1
    try:
        neumessung = (neumessung - min(neumessung)) / ( max(neumessung) - min(neumessung) )
        vergleichsmessung = (vergleichsmessung - min(vergleichsmessung)) / ( max(vergleichsmessung) - min(vergleichsmessung) )
        correlation_coefficient = berechne_korrelation(neumessung, vergleichsmessung)
    except:
        correlation_coefficient = 0.0
    return(correlation_coefficient)

#%% Main 

def klassifizierung(vergleichsmessung_kreis, vergleichsmessung_kreuz, neumessung, grenzwert_c_kreis = 0.68, grenzwert_c_kreuz = 0.55):

    acc_x_kreis, acc_y_kreis, acc_z_kreis, o_alpha_kreis, o_beta_kreis, o_gamma_kreis = process_file(vergleichsmessung_kreis, glaettung)
    acc_x_kreuz, acc_y_kreuz, acc_z_kreuz, o_alpha_kreuz, o_beta_kreuz, o_gamma_kreuz = process_file(vergleichsmessung_kreuz, glaettung)
    acc_x, acc_y, acc_z, o_alpha, o_beta, o_gamma = process_file(neumessung, glaettung) 
    
    c_acc_x_kreis = berechne_korrelation(acc_x, acc_x_kreis)
    c_acc_y_kreis = berechne_korrelation(acc_y, acc_y_kreis)
    c_acc_z_kreis = berechne_korrelation(acc_z, acc_z_kreis)

    c_o_alpha_kreis = korrelation_orientierung(o_alpha_kreis, o_alpha)
    c_o_beta_kreis = korrelation_orientierung(o_beta_kreis, o_beta)
    c_o_gamma_kreis = korrelation_orientierung(o_gamma_kreis, o_gamma)
    
    
    c_acc_x_kreuz = berechne_korrelation(acc_x, acc_x_kreuz)
    c_acc_y_kreuz = berechne_korrelation(acc_y, acc_y_kreuz)
    c_acc_z_kreuz = berechne_korrelation(acc_z, acc_z_kreuz)
    
    c_o_alpha_kreuz = korrelation_orientierung(o_alpha_kreuz, o_alpha)
    c_o_beta_kreuz = korrelation_orientierung(o_beta_kreuz, o_beta)
    c_o_gamma_kreuz = korrelation_orientierung(o_gamma_kreuz, o_gamma)
    
    correlations = {"Kreise_Acc": [np.array([c_acc_x_kreis,c_acc_y_kreis,c_acc_z_kreis])],
                    "Kreise_O": [np.array([c_o_alpha_kreis,c_o_beta_kreis,c_o_gamma_kreis])],
                    "Kreuze_Acc": [np.array([c_acc_x_kreuz,c_acc_y_kreuz,c_acc_z_kreuz])],
                    "Kreuze_O": [np.array([c_o_alpha_kreuz,c_o_beta_kreuz,c_o_gamma_kreuz])]}
    
    
    mean_correlation_kreise_acc = np.mean( np.abs(correlations["Kreise_Acc"]))
    mean_correlation_kreise_o= np.mean( np.abs(correlations["Kreise_O"]))
    mean_correlation_kreuze_acc = np.mean( np.abs(correlations["Kreuze_Acc"]))
    mean_correlation_kreuze_o = np.mean( np.abs(correlations["Kreuze_O"]))
    
    mean_correlation_kreise = (gewicht_acc * mean_correlation_kreise_acc + gewicht_or * mean_correlation_kreise_o)
    mean_correlation_kreuze = (gewicht_acc * mean_correlation_kreuze_acc + gewicht_or * mean_correlation_kreuze_o)

    # Wenn es mehr mit Kreis korreliert als mit Kreuz
    if mean_correlation_kreise > mean_correlation_kreuze:
        # Prüfen ob Grenzwert eingehalten wird
        if mean_correlation_kreise >= grenzwert_c_kreis:
            result = "Kreis erkannt!"
        else:
            result = "Nichts erkannt!"
            
    elif mean_correlation_kreise < mean_correlation_kreuze:
        # Prüfen ob Grenzwert eingehalten wird
        if mean_correlation_kreuze >= grenzwert_c_kreuz:
            result = "Kreuz erkannt!"
        else:
            result = "Nichts erkannt!"
            
    else:
        result = "Nichts erkannt!"

    return(result)

    
    


