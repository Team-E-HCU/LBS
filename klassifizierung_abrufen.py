# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 14:14:21 2023

@author: lukas
"""
from klassifizierung_2D_ausVariable import klassifizierung
import numpy as np


#%% 

##### Abruf des Hauptprogramms für alle Dateien #####


grenzwert_kreis = 0.0085 * 2


filelist_kreise = [f"data/Kreise/SensorDataKreis ({i}).txt" for i in range(0,250)]
filelist_kreuze = [f"data/Kreuze/SensorDataKreuz ({i}).txt" for i in range(0,250)]


kreis_counter = 0
for file in filelist_kreise:
    r = klassifizierung(open(file).read())
    if r == "Kreis erkannt":
        kreis_counter += 1


kreuz_counter = 0
for file in filelist_kreuze:
    r = klassifizierung(open(file).read())
    if r == "Nichts erkannt":
        kreuz_counter += 1
        

print(f"{kreis_counter} von {len(filelist_kreise)} Kreisen erkannt ({np.round((kreis_counter/len(filelist_kreise))*100,2)}%)")      
print(f"{kreuz_counter} von {len(filelist_kreuze)} Kreuzen erkannt ({np.round((kreuz_counter/len(filelist_kreuze))*100,2)}%)")
print(f"Gesamt: {kreis_counter+kreuz_counter} von {len(filelist_kreise)+len(filelist_kreuze)} korrekt erkannt ({np.round(((kreis_counter+kreuz_counter)/(len(filelist_kreuze)+len(filelist_kreuze)))*100,2)}%)")


#%%

##### Abruf des Hauptprogramms für eine Dateie #####
"""
print(klassifizierung("data/Kreuze/SensorDataKreuz (16).txt"))
"""

#%% Datensatz als Variable mit Textübergeben

data = open("data/Kreise/SensorDataKreis (16).txt").read()
print(klassifizierung(data))