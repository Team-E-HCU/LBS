# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def data_preparation(filename):
    """
    Hiermit wird die .txt Datei, welcher mit Hilfe unser GitHub Page erstellt wurde eingelesen und verarbeitet.
    
    Parameters
    ----------
    filename : dateiname

    Returns
    -------
    accData (Daten des Beschleunigungssensors)
    orData (Daten des Orientierungssensors)

    """
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

def synchronisierung(accData, orData):
    # Zeitliche Synchronisation der Daten (Durch Interpolation)
    """
    Gyroskop- und Beschleunigungswerte haben nicht exakt identische Zeitstempel
    Daher folgt die Synchronisierung über Interpolation
    
    Parameters
    ----------
    accData (Daten des Beschleunigungssensors)
    orData (Daten des Orientierungssensors)

    Returns
    -------
    Beschleunigungs- und Orientierungssensordaten als einzelnes Array für jede achse mit identischen Zeitstempeln
    """
    
    # Interpolation der Werte aus accData für jeden Zeitstempel orData
    accDatainterp_x = np.interp(orData[:,0], accData[:,0], accData[:,1])
    accDatainterp_y = np.interp(orData[:,0], accData[:,0], accData[:,2])
    accDatainterp_z = np.interp(orData[:,0], accData[:,0], accData[:,3])
    
    # Einzelwerte extrahieren 
        # Beschleunigung
    acc_time = orData[:,0]
    acc_x = accDatainterp_x
    acc_y = accDatainterp_y
    acc_z = accDatainterp_z
    
        # Orientieriung
    o_time = orData[:,0]
    o_alpha = orData[:,1]
    o_beta = orData[:,2]
    o_gamma = orData[:,3]
    
    # Zeitstemepl auf 0 skalieren (Bei 0 Anfangen)
    acc_time -= acc_time[0]
    o_time -= o_time[0]
    
    return(acc_time, acc_x, acc_y, acc_z, o_time, o_alpha, o_beta, o_gamma)


def reduktion(time, data, length = 300):
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

#%%
# Schritte 1 und 2: Datensatz vorbereiten und Merkmale extrahieren
filelist_kreise = [f"data/Kreise/SensorDataKreis ({i}).txt" for i in range(0,250)]
filelist_kreuze = [f"data/Kreuze/SensorDataKreuz ({i}).txt" for i in range(0,250)]
filelist_validierung = [f"data/Kreuze/SensorDataKreuz ({i}).txt" for i in range(0,59)]



X, Y = np.array([99 for i in range(2100)]), np.array([99])

for file in filelist_kreise:
    accData, orData = data_preparation(file)
    acc_time, acc_x, acc_y, acc_z, o_time, o_alpha, o_beta, o_gamma = synchronisierung(accData, orData)
    
    new_time_acc, acc_x = reduktion(acc_time, acc_x)
    new_time_acc, acc_y = reduktion(acc_time, acc_y)
    new_time_acc, acc_z = reduktion(acc_time, acc_z)
    new_time_o, o_alpha = reduktion(o_time, o_alpha)
    new_time_o, o_beta = reduktion(o_time, o_beta)
    new_time_o, o_gamma = reduktion(o_time, o_gamma)
    datensatz = np.hstack((new_time_acc, acc_x, acc_y, acc_z, o_alpha, o_beta, o_gamma))
    
    X = np.vstack((X,datensatz))
    Y1 = ["Kreis" for i in filelist_kreise]
    
X = X[1:]

for file in filelist_kreuze:
    accData, orData = data_preparation(file)
    acc_time, acc_x, acc_y, acc_z, o_time, o_alpha, o_beta, o_gamma = synchronisierung(accData, orData)
    
    new_time_acc, acc_x = reduktion(acc_time, acc_x)
    new_time_acc, acc_y = reduktion(acc_time, acc_y)
    new_time_acc, acc_z = reduktion(acc_time, acc_z)
    new_time_o, o_alpha = reduktion(o_time, o_alpha)
    new_time_o, o_beta = reduktion(o_time, o_beta)
    new_time_o, o_gamma = reduktion(o_time, o_gamma)
    datensatz = np.hstack((new_time_acc, acc_x, acc_y, acc_z, o_alpha, o_beta, o_gamma))
    
    X = np.vstack((X,datensatz))
    Y2 = ["Kreuz" for i in filelist_kreuze]
    Y = Y1 + Y2

#%%

# Label-Encoding der Klassenlabels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# Schritt 3: Trainingsdaten vorbereiten
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Schritt 4: Modell erstellen und trainieren
model = Sequential()
model.add(Dense(120, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=5)

# Schritt 5: Modell evaluieren
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)

#%%
# Schritt 6: Vorhersagen treffen und Testen
# Jeweils 25 (Kreis, Kreuz müll) für Vorhersagen machen und wahrscheinlichkeiten prüfen
# 0 bis 19 Kreise / 20 bis 39 Kreute /  40 bis 49 Müll 
filelist_validierung = [f"data/Kreuze/SensorDataKreuz ({i}).txt" for i in range(0,60)]

X_valid = np.array([99 for i in range(2100)])
for file in filelist_validierung:
    accData, orData = data_preparation(file)
    acc_time, acc_x, acc_y, acc_z, o_time, o_alpha, o_beta, o_gamma = synchronisierung(accData, orData)
    
    new_time_acc, acc_x = reduktion(acc_time, acc_x)
    new_time_acc, acc_y = reduktion(acc_time, acc_y)
    new_time_acc, acc_z = reduktion(acc_time, acc_z)
    new_time_o, o_alpha = reduktion(o_time, o_alpha)
    new_time_o, o_beta = reduktion(o_time, o_beta)
    new_time_o, o_gamma = reduktion(o_time, o_gamma)
    datensatz = np.hstack((new_time_acc, acc_x, acc_y, acc_z, o_alpha, o_beta, o_gamma))
    
    X_valid = np.vstack((X_valid,datensatz))

X_valid = X_valid[1:]   
Y_valid = ["Kreis" * 20] + ["Kreuz" * 20] + ["Andere" * 20]


predictions = model.predict(X_valid)

# Extrahieren der Wahrscheinlichkeiten für jede Klasse
probabilities = np.max(predictions, axis=1)

# Konvertierung der Vorhersagen in Klassenindizes
predicted_labels = np.argmax(predictions, axis=1)

# Ausgabe der vorhergesagten Klasse und zugehörigen Wahrscheinlichkeiten
for i, label in enumerate(predicted_labels):
    probability = probabilities[i]
    print(f'Vorhersage: {label_encoder.inverse_transform([label])[0]}, Wahrscheinlichkeit: {round(probability*100,2)}%')

