"""
Verarbeitung der ausgelesenen Sensordaten aus:
    https://team-e-hcu.github.io/Sensor-Data-Collection/
    
Input: Textdatei mit Gyroskop- und Sensordaten (unverändert)

Anmerkung:
Die Funktionen diff_kreis und berechne_kreis sind redundant, eine dient aber mehr der visualisierung und die andere mehr der berechnung
"""
import numpy as np
import matplotlib.pyplot as plt
import math as m


filelist_kreise = [f"data_new/Kreise/SensorData ({i}).txt" for i in range(0,10)]
filelist_kreuze = [f"data_new/Kreuze/SensorData ({i}).txt" for i in range(10,20)]

create_plots = True # True/False --> Sollen plots gemacht werden? (Programmlaufzeit +)

#%%
"""
##### Benötigte Funktionen deklarieren #####
"""

""" Funktionen zur Vorbereitung der Daten """

# Textdatei einlesen und splitten in 2 Variablen für Beschleunigungs und Orientierungsdaten
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
    """Berechnung der Geschwindigkeit nach der Formel s = v*t
    
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

def calc_trajectory3D(position_data, rotation_data):
    """
    This function calculates a 3D trajectory of an IMU using
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
    trajectory = {"x": [0.0], "y": [0.0], "z":[0.0]}
    rho = m.pi/180
    for i, e in enumerate(position_change["x"]):
        trajectory["x"].append(trajectory["x"][i] + (m.sin(rotation_data["z"][i]*rho)*position_change["x"][i]) + (m.cos(rotation_data["z"][i]*rho)*position_change["y"][i]))
        trajectory["y"].append(trajectory["y"][i] + (m.sin(rotation_data["z"][i]*rho)*position_change["y"][i]) + (m.cos(rotation_data["z"][i]*rho)*position_change["x"][i]))
        trajectory["z"].append(trajectory["z"][i] + (m.sin(rotation_data["z"][i]*rho)*position_change["z"][i]) + (m.cos(rotation_data["z"][i]*rho)*position_change["z"][i]))
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


""" Funktionen zum Plotten von Ergebnissen """

def plot_beschleunigung(acc_time, acc_x, acc_y, acc_z, fname):
    if create_plots == True:
        # Plot Beschleunigungsmesswerte
        fig = plt.plot(acc_time, acc_x, label="x-Achse")
        fig = plt.plot(acc_time, acc_y, label="y-Achse")
        fig = plt.plot(acc_time, acc_z, label="z-Achse")
        plt.xlabel("Zeit [ms]")
        plt.ylabel("Beschleunigung [m/s²]")
        plt.legend()
        plt.title("Beschleunigung")
        plt.savefig(f"plots/{fname}_Beschleunigung.png")
        plt.show()
    
def plot_orientierung(o_time, o_alpha, o_beta, o_gamma, fname):
    if create_plots == True:
        # Plot Orientierung
        fig = plt.plot(o_time, o_alpha, label="Alpha")
        fig = plt.plot(o_time, o_beta, label="Beta")
        fig = plt.plot(o_time, o_gamma, label="Gamma")
        plt.xlabel("Zeit [ms]")
        plt.ylabel("Orientierung [°]")
        plt.legend()
        plt.title("Orientierung")
        plt.savefig(f"plots/{fname}_Orientierung.png")
        plt.show()
    
def plot_geschwindigkeit(delta_t_acc, acc_v_x, acc_v_y, acc_v_z, fname):
    if create_plots == True:
        # Plot der Geschwindigkeit für jede Achse
        fig = plt.plot(np.cumsum(delta_t_acc), acc_v_x, label="X-Achse")
        fig = plt.plot(np.cumsum(delta_t_acc), acc_v_y, label="Y-Achse")
        fig = plt.plot(np.cumsum(delta_t_acc), acc_v_z, label="Z-Achse")
        plt.xlabel("Zeit [s]")
        plt.ylabel("Geschwindigkeit [m/s]")
        plt.legend()
        plt.title("Geschwindigkeit")
        plt.savefig(f"plots/{fname}_Geschwindigkeit.png")
        plt.show()
        
def plot_trajektorie_xy(dataset, mx, my, fname="XY-Trajektorie.png"):
    if create_plots == True:
        fig = plt.plot(dataset["x"], dataset["y"], '.',label="Positionen")
        fig = plt.plot(mx, my, '.', label="Mittelpunkt", color="red")
        #plt.axis('scaled')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("XY-Trajektorie")
        plt.savefig(fname)
        plt.show()
    
def plot_trajektorie_xyz(trajectorie_xyz, fname):
    if create_plots == True:
        fig = plt.plot()
        ax = plt.axes(projection='3d')
        # Data for three-dimensional scattered points
        zdata = trajectorie_xyz["z"]
        xdata = trajectorie_xyz["x"]
        ydata = trajectorie_xyz["y"]
        #plt.xlabel("X [m]")
        #plt.ylabel("Y [m]")
        #plt.zlabel("Z [m]")
        plt.title("XYZ-Trajektorie")
        ax.scatter3D(xdata, ydata, zdata, c = zdata)
        #ax.plot3D(xdata, ydata, zdata)
        ax.view_init(20, 45)
        plt.show()
        plt.savefig(f"plots/{fname}_XYZ_Trajektorie.png")
    
def plot_kreis_XY(dataset, kreis, mx, my, fname):
    if create_plots == True:
        ax = plt.axes()
        fig_kreis = plt.plot(dataset["x"], dataset["y"], '.',label="Positionen")
        fig_kreis = plt.plot(mx, my, '.', label="Mittelpunkt", color="red")
        fig_kreis = plt.plot(kreis["x"], kreis["y"], label="Kreis", color="red")
        plt.axis('scaled')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Kreis in XY")
        plt.savefig(f"plots/{fname}_XY_Trajektorie.png")
        plt.show()

#%% 
"""
##### Hauptprogramm #####
"""

def processing(filename):
    
    # Aufbereitung der Daten
    accData, orData = data_preparation(filename)
    acc_time, acc_x, acc_y, acc_z, o_time, o_alpha, o_beta, o_gamma = synchronisierung(accData, orData)
    
    # Entfernung der Dateiendung, damit plots problemfrei benannt werden
    filename = (filename[:-4]).split("/")[2]
        
    # Plotten der rohen Messwerte
    plot_beschleunigung(acc_time, acc_x, acc_y, acc_z, filename)
    plot_orientierung(o_time, o_alpha, o_beta, o_gamma, filename)
    
    # Berechnungen:
        
    # Zunächst Delta t der Einzelschritte aufstellen:
    delta_t_acc = (np.diff(acc_time))/1000 # Umrechnung zu Sekunden 
    
    # Berechnung der Geschwindigkeit für jede Achse
    acc_v_x = calc_v(acc_x , delta_t_acc)
    acc_v_y = calc_v(acc_y , delta_t_acc)
    acc_v_z = calc_v(acc_z , delta_t_acc)
    
    # Plotten der errechneten Geschwindigkeiten
    plot_geschwindigkeit(delta_t_acc, acc_v_x, acc_v_y, acc_v_z, filename)
    
    # Berechnung der zurückgelegten Strecke für jede Achse
    s_x = calc_s(delta_t_acc, acc_v_x)
    s_y = calc_s(delta_t_acc, acc_v_y)
    s_z = calc_s(delta_t_acc, acc_v_z)
    
    # Berechnung der Trajektorie in 2D (XY-Richtung)
    pos = {"x": s_x, "y": s_y, "z": s_z}
    rot = {"x": np.diff(o_alpha), "y": np.diff(o_beta), "z": np.diff(o_gamma)} 
    trajectorie_xy = calc_trajectory2D(pos, rot)
    
    # Berechnung der Trajektorie in 3D (XYZ-Richtung)
    pos = {"x": s_x, "y": s_y, "z": s_z}
    rot = {"x": np.diff(o_alpha), "y": np.diff(o_beta), "z": np.diff(o_gamma)} 
    trajectorie_xyz = calc_trajectory3D(pos, rot)
    
    
    # Filterung, Glättung und Visualisierung der 2D Trajektorie
    dataset,mx,my = filter_trajektorie_2D(trajectorie_xy, limit=3)
    dataset = smooth_trajektorie_2D(dataset,3)
    plot_trajektorie_xy(dataset, mx, my, filename)
    
    # Visualisierung der Trajektorie in 3D
    plot_trajektorie_xyz(trajectorie_xyz, filename)
    
    # Berechnung und Visualisierung eines Kreises in die XY-Trajektorie
    kreis, mx, my = berechne_kreis_XY(dataset)
    plot_kreis_XY(dataset, kreis, mx, my, filename)
    
    # Berechnung der Standardabweichung zum Kreis
    mittlere_abweichung = diff_kreis_XY(trajectorie_xy, kreis, mx, my)
    print(f"{filename}, Abweichung: {np.round(mittlere_abweichung,3)}[m]")
    return(mittlere_abweichung)

#%% 
"""
##### Abruf des Hauptprogramms für alle Dateien #####
"""
d_kreise = []
d_kreuze = []
    
print("Berechnung der mittleren Abweichung vom Kreis für Kreise: ")
d_kreise = np.mean([processing(filename) for filename in filelist_kreise])
    
print("\nBerechnung der mittleren Abweichung vom Kreis für Kreuze: ")
d_kreuze = np.mean([processing(filename) for filename in filelist_kreuze])

print(f"\nMittlere Abweichungen:\nKreise: {np.mean(d_kreise)}[m]\nKreuze: {np.mean(d_kreuze)}[m]")

#%% 
"""
##### Abruf des Hauptprogramms für eine einzelne Datei #####
"""
#processing("sensorData_KreisXYGross.txt")