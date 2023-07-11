"""
Klassifizierung der ausgelesenen Sensordaten aus:
    https://team-e-hcu.github.io/Sensor-Data-Collection/
    
Input: Textdatei mit Gyroskop- und Sensordaten (unverändert)

Output: Klassifizierung
    
"""
import numpy as np
import math as m

#%%
"""
##### Benötigte Funktionen deklarieren #####
"""

""" Funktionen zur Vorbereitung der Daten """

# Textdatei einlesen und splitten in 2 Variablen für Beschleunigungs und Orientierungsdaten
def data_preparation(data):
    """
    Hiermit wird die variable mit den Messdaten.
    
    Parameters
    ----------
    filename : dateiname

    Returns
    -------
    accData (Daten des Beschleunigungssensors)
    orData (Daten des Orientierungssensors)

    """

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

def calc_v(acc,time):
    """Berechnung der Geschwindigkeitswerte aus Beschleunigung und Zeit

    Parameters
    ----------
    acc : Beschleunigung [m/s²]
    time : Zeitdifferenzen [s]

    Returns
    -------
    speed: Geschwindigkeit [m/2]

    """
    vi = 0
    acc_v = np.array([])
    for ai,ti in zip(acc[1:],time):
        vi = ai * ti + vi
        acc_v = np.append(acc_v,vi)
        
    return(acc_v)

def calc_s(time,speed):
    """Berechnung der Strecke nach der Formel s = v*t
    
    Parameters
    ----------
    time : Zeitdifferenzen [s].
    speed: Geschwindigkeit [m/2]

    Returns
    -------
    s: Strecke [m]

    """
    si = 0
    acc_s = np.array([])
    for (ti,vi) in zip(time,speed):
        si = vi * ti
        acc_s = np.append(acc_s,si)
    return(acc_s)

def calc_trajectory2D(position_data, rotation_data):
    """
    This function calculates a 2D trajectory of an IMU using
    distance and rotation values. [Integrated Navigation EX2]

    Input:
        position_data ({"x": [float], "y": [float], "z": [float]}): position data as lists in a dictionary
        rotation_data ({"x": [float], "y": [float], "z": [float]}): rotation data as lists in a dictionary
    """
    last_value = {"x": 0.0, "y": 0.0, "z": 0.0}
    position_change = {"x": [], "y": [], "z": []}
    for xyz in ["x", "y", "z"]:
        for i, e in enumerate(position_data[xyz]):
            position_change[xyz].append(e-last_value[xyz])
            last_value[xyz] = e
    trajectory = {"x": [0.0], "y": [0.0]}
    rho = m.pi/180
    for i, e in enumerate(position_change["x"]):
        trajectory["x"].append(trajectory["x"][i] + (m.sin(rotation_data["z"][i]*rho)*position_change["x"][i]) + (m.cos(rotation_data["z"][i]*rho)*position_change["y"][i]))
        trajectory["y"].append(trajectory["y"][i] + (m.sin(rotation_data["z"][i]*rho)*position_change["y"][i]) + (m.cos(rotation_data["z"][i]*rho)*position_change["x"][i]))
    return(trajectory)


def filter_trajektorie_2D(dataset, limit=2):
    """Mit dieser Funktion werden die Daten in XY Ebene nach Ausreißern gefiltert.
    Kriterium ist der Abstand zum Mittelpunkt des Datensatzes

    Parameters
    ----------
    dataset: Dictionary mit x und y
    limit: limit * mittlerer Abstand als Grenzwert für Filterung

    Returns
    -------
    filtered_dataset: Gefilterter Datensatz
    mean_x, mean_y: Zentrum des Datensatzes
    """
    # Mittelpunkt des Datensatzes berechnen
    mean_x, mean_y = np.mean(dataset["x"]), np.mean(dataset["y"])
    
    # Abstand zum Mittelpunkt für jeden Einzelpunkt berechnen:
    dataset["d"] = np.sqrt( (mean_x - dataset["x"])**2 + (mean_y - dataset["y"])**2 )
    
    # Mittleren Abstand berechnen
    d_mean = np.mean(dataset["d"])
    
    # Alle Punkte mit Abstand > limit*Mittelwert entfernen
    filter_dataset = {"x":[], "y":[]}
    for x,y,d in zip(dataset["x"],dataset["y"],dataset["d"]):
        if d < limit * d_mean:
            filter_dataset["x"].append(x)
            filter_dataset["y"].append(y)
        
    
    return(filter_dataset,mean_x,mean_y)

def smooth_trajektorie_2D(dataset, neighbours=3):
    """Glättung der Trajektorie durch gleitendes Mittel unter Einbezug einer gegeben Anzahl von Nachbarpunkten

    Parameters
    ----------
    dataset : Zu glättende Trajektorie
    neighbours : Einzubeziehende Nachbarpunkte

    Returns
    -------
    smooth_dataset: Geglättete Trajektorie
    """
    smooth_dataset = {"x":[], "y":[]}
    for i in range(0+neighbours,len(dataset["x"])-neighbours,1):
        smooth_dataset["x"].append( np.mean(dataset["x"][i-neighbours:i+neighbours]) )
        smooth_dataset["y"].append( np.mean(dataset["y"][i-neighbours:i+neighbours]) )
    return(smooth_dataset)

def berechne_kreis_XY(dataset):
    """Diese Funktion berechnet den Mittelpunkt einer gefilterten und geglätteten Trajektorie und berechnet einen Kreis
    aus mit mittlerem Abstand zu allen Punkten als Radius

    Parameters
    ----------
    dataset: Trajektorie, in die ein Kreis gelegt werden soll

    Returns
    -------
    kreis: Punktdatensatz des Kreises
    speedmean_x, mean_y: Kreismittelpunkt
    """
    
    # Mittelpunkt des Datensatzes berechnen
    mean_x, mean_y = np.mean(dataset["x"]), np.mean(dataset["y"])
    
    # Abstand zum Mittelpunkt für jeden Einzelpunkt berechnen:
    dataset["d"] = np.sqrt( (mean_x - dataset["x"])**2 + (mean_y - dataset["y"])**2 )
    
    # Mittleren Abstand berechnen
    d_mean = np.mean(dataset["d"])
    
    # Kreispunkte berechnen 
    kreis = {"x":[], "y":[]}
    rho = m.pi/180
    for i in range(1,361):
        kreis["x"].append( mean_x + m.sin(i*rho) * d_mean )
        kreis["y"].append( mean_y + m.cos(i*rho) * d_mean )
    return(kreis, mean_x, mean_y)
        
def diff_kreis_XY(trajektorie, kreis, mx, my):
    """
    Diese Funktion berechnet den mittleren Abstand aller Punkte einer Trajektorie zu einem Kreis
    
    GGf. hier auf andere Methode umstellen --> evtl. Stabw? oder Varianz?
    ----------
    trajektorie: Datensatz 
    kreis: Kreispunkte
    mx, my: Kreismittelpunkt

    Returns
    -------
    diff_mean: Mittlere, absolute Abweichung

    """
    radius = np.sqrt( (kreis["x"][0]-mx)**2+(kreis["y"][0]-my)**2 ) 
    distanz_zur_mitte =  np.array([])
    for x,y in zip(trajektorie["x"],trajektorie["y"]):
        distanz_zur_mitte = np.append(distanz_zur_mitte, np.sqrt( (x-mx)**2+(y-my)**2) )
    diff = np.abs(distanz_zur_mitte - radius)
    diff_mean = np.mean(diff)
    return(diff_mean)


#%% 
"""
##### Hauptprogramm #####
"""

def klassifizierung(data, grenzwert_kreis = 0.0085 * 1.75):
    
    # Aufbereitung der Daten
    accData, orData = data_preparation(data)
    acc_time, acc_x, acc_y, acc_z, o_time, o_alpha, o_beta, o_gamma = synchronisierung(accData, orData)
    
    # Berechnungen:
        
    # Zunächst Delta t der Einzelschritte aufstellen:
    delta_t_acc = (np.diff(acc_time))/1000 # Umrechnung zu Sekunden 
    
    # Berechnung der Geschwindigkeit für jede Achse
    acc_v_x = calc_v(acc_x , delta_t_acc)
    acc_v_y = calc_v(acc_y , delta_t_acc)
    acc_v_z = calc_v(acc_z , delta_t_acc)

    
    # Berechnung der zurückgelegten Strecke für jede Achse
    s_x = calc_s(delta_t_acc, acc_v_x)
    s_y = calc_s(delta_t_acc, acc_v_y)
    s_z = calc_s(delta_t_acc, acc_v_z)
    
    # Berechnung der Trajektorie in 2D (XY-Richtung)
    pos = {"x": s_x, "y": s_y, "z": s_z}
    rot = {"x": np.diff(o_alpha), "y": np.diff(o_beta), "z": np.diff(o_gamma)} 
    trajectorie_xy = calc_trajectory2D(pos, rot)
    
    
    
    # Filterung, Glättung und Visualisierung der 2D Trajektorie
    dataset,mx,my = filter_trajektorie_2D(trajectorie_xy, limit=3)
    dataset = smooth_trajektorie_2D(dataset,3)
    

    
    # Berechnung und Visualisierung eines Kreises in die XY-Trajektorie
    kreis, mx, my = berechne_kreis_XY(dataset)
    
    
    # Berechnung der Standardabweichung zum Kreis
    mittlere_abweichung_kreis = diff_kreis_XY(trajectorie_xy, kreis, mx, my)
    
    result = ""
    if mittlere_abweichung_kreis <= grenzwert_kreis:
        result = "Kreis erkannt"
    else:
        result = "Nichts erkannt"
    
    return(result)
