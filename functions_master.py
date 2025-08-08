import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
import scipy.signal.windows as win
import scipy as sp
import warnings
import re
from ipywidgets import interact,widgets
from scipy.optimize import curve_fit
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.misc import derivative


# Función para extraer la tempertura del nombre del archivo
c = 0.299792458 # speed of light mm/ps
def extraer_temperatura(nombre_archivo):
    match = re.search(r'(\d+\.\d+|\d+)K', nombre_archivo)
    if match:
        return float(match.group(1))
    else:
        return None
    
def apply_window(params):
    """
    Aplica una ventana de scipy a una señal.

    Parámetros:
    - params (list): Lista con los siguientes elementos:
      1. Nombre de la ventana (str).
      2. Tamaño de la ventana (M) (int).
      3. Parámetros adicionales de la ventana (opcional).

    Retorna:
    - ventana (array): La ventana calculada.
    """
    try:
        # Extrae el nombre de la ventana y el tamaño M de la lista
        window_name = params[0]
        M = params[1]
        extra_params = params[2:]  # Parámetros adicionales para la ventana

        # Obtiene la función de ventana a partir del nombre
        window_func = getattr(win, window_name)
        
        # Aplica la ventana con M como primer argumento y los parámetros adicionales
        window = window_func(M, *extra_params)
        return window
    except AttributeError:
        raise ValueError(f"Ventana '{window_name}' no está disponible en scipy.signal.windows.")
    except TypeError as e:
        raise ValueError(f"Error al pasar los parámetros a la ventana: {e}")

    
def FourierT(f,N):
    return (sp.fft.fft(f,n=N))

def FourierI(f,N):
    return (sp.fft.ifft(f,n=N))


def loadData(name):
    # Leer el archivo CSV
    nombre_archivo = name
    df = pd.read_csv(nombre_archivo)

    # Asignar cada columna a una variable
    Temp = df['Temp'].values
    ep_inf = df['ep_inf'].values
    ep_s = df['ep_s'].values
    nu_to = df['nu_to'].values
    Gamma = df['Gamma'].values
    nu_p = df['nu_p'].values
    gamma = df['gamma'].values

    return Temp, ep_inf,ep_s,nu_to,Gamma,nu_p,gamma


def epsilon_drude(nu,ep_inf,ep_s,nu_to,Gamma,nu_p,gamma):

    return ep_inf+(ep_s-ep_inf)*nu_to**2/(nu_to**2-nu**2-1j*nu*Gamma)-nu_p**2/(nu**2+1j*nu*gamma)

def epsilon(nu,ep_inf,ep_s,nu_to,Gamma):

    return ep_inf+(ep_s-ep_inf)*nu_to**2/(nu_to**2-nu**2-1j*nu*Gamma)

def T_crit(T,a,Tcrit):

    return a*(T-Tcrit)**0.5

def epsilonboth_drude(nu,ep_inf,ep_s,nu_to, Gamma,nu_p,gamma):
    N = len(nu)
    nu_real = nu[:N//2]
    nu_imag = nu[N//2:]
    y_real = np.real(epsilon_drude(nu_real, ep_inf,ep_s,nu_to,Gamma,nu_p,gamma))
    y_imag = np.imag(epsilon_drude(nu_imag, ep_inf,ep_s,nu_to,Gamma,nu_p,gamma))
    return np.hstack([y_real, y_imag])

def epsilonboth(nu,ep_inf,ep_s,nu_to,Gamma):
    N = len(nu)
    nu_real = nu[:N//2]
    nu_imag = nu[N//2:]
    y_real = np.real(epsilon(nu_real, ep_inf,ep_s,nu_to,Gamma))
    y_imag = np.imag(epsilon(nu_imag, ep_inf,ep_s,nu_to,Gamma))
    return np.hstack([y_real, y_imag])


def fit_complex_drude(x,y,p0=[]):

    #Return ep_inf,ep_s,nu_to,Gamma,nu_p,gamma
    yReal = np.real(y)
    yImag = np.imag(y)
    yBoth = np.hstack([yReal, yImag])
    #lim_inf = [0, 0, 0, 0.00000, 0.00000, 0.00000] # 0.77868 # 0.06% -> 0.54868[::-1] # 0.10% -> 0.28562
    #lim_sup = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf] # 1.29225

    bounds = ([0, 0, 0.4, 0, 0, 0], [np.inf, np.inf, 2, np.inf, np.inf,np.inf])
    if not p0 :
        poptBoth, pcovBoth = curve_fit(epsilonboth_drude, np.hstack([x, x]), yBoth,bounds=bounds,maxfev=20000)
        return poptBoth

    else:
        poptBoth, pcovBoth = curve_fit(epsilonboth_drude, np.hstack([x, x]), yBoth,p0=p0,bounds=bounds,maxfev=20000)
        return poptBoth

def fit_complex(x,y,p0):

    #Return ep_inf,ep_s,nu_to,Gamma,nu_p,gamma
    yReal = np.real(y)
    yImag = np.imag(y)
    yBoth = np.hstack([yReal, yImag])
    #lim_inf = [0, 0, 0, 0.00000, 0.00000, 0.00000] # 0.77868 # 0.06% -> 0.54868[::-1] # 0.10% -> 0.28562
    #lim_sup = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf] # 1.29225
    bounds = ([0, 0, 0.2,0], [np.inf, np.inf, 2, np.inf])
    if not p0 :
  
        poptBoth, pcovBoth = curve_fit(epsilonboth, np.hstack([x, x]), yBoth,bounds=bounds,maxfev=10000)
        return poptBoth

    else:
        poptBoth, pcovBoth = curve_fit(epsilonboth, np.hstack([x, x]), yBoth,maxfev=10000)
        return poptBoth

# def FourierT(f,N):
#     return np.conj(sp.fft.fft(f.values,n=N))

def agrupar_por_rango_temperatura(archivos_por_temp, rangos):
    archivos_por_rango = {rango: [] for rango in rangos}
    for temp, archivos in archivos_por_temp.items():
        for rango in rangos:
            if rango[0] <= temp < rango[1]:  # Verifica si la temperatura está dentro del rango
                archivos_por_rango[rango].extend(archivos)
                break  # Detiene el ciclo una vez que encuentra el rango correcto

    archivos_por_rango  = {k: v for k, v in archivos_por_rango .items() if v}
    
    return archivos_por_rango

def FourierT2(f,N):
    return np.conj(sp.fft.fft(f.values,n=N))

def generar_rangos(min_temp, max_temp, paso):
    rangos = []
    if paso == 0:
        return None
    else:
        for i in np.arange(min_temp, max_temp+paso, paso):
            rangos.append((i, i + paso))
        return rangos

# Función para procesar archivos .dat
def convert_dats(carpeta,N):
    nueva_carpeta = os.path.join(carpeta, 'carpeta1')
    os.makedirs(nueva_carpeta, exist_ok=True)
    
    archivos = [archivo for archivo in os.listdir(carpeta) if archivo.endswith('.dat')]
    print(archivos)
    # Agrupar archivos por temperatura
    archivos_por_temp = {}
    temperaturas = []

    for archivo in archivos:
        temp = extraer_temperatura(archivo)
        if temp is not None:
            temperaturas.append(temp)
            archivos_por_temp.setdefault(temp, []).append(archivo)

    if not temperaturas:
        print("No se encontraron temperaturas en los archivos.")
        return
    
    archivos_por_temp = dict(sorted(archivos_por_temp.items()))
    #FIN AGRUPACION ARCHIVOS POR TEMP
    
    min_temp = min(temperaturas)
    max_temp = max(temperaturas)

    # Generar rangos de temperatura desde la mínima hasta la máxima, de 10 en 10
    rangos_temperatura = generar_rangos(min_temp, max_temp, N)

    if rangos_temperatura is None:
        archivos_por_rango = archivos_por_temp
    else: 
    # Agrupar archivos por rango de temperatura
        archivos_por_rango = agrupar_por_rango_temperatura(archivos_por_temp, rangos_temperatura)


    # Procesar archivos por cada temperatura
    for rango, lista_archivos in archivos_por_rango.items():
        try:
            # Inicializar variables de acumulación
            suma_col1 = None
            suma_col2 = None
            n_archivos = len(lista_archivos)
            
            # Iterar sobre los archivos con la misma temperatura
            temps_arch = []

            for archivo in lista_archivos:
                temps_arch.append(extraer_temperatura(archivo))
                df = pd.read_csv(os.path.join(carpeta, archivo), delim_whitespace=True)
                
                # Acumular las columnas
                if suma_col1 is None:
                    suma_col1 = df['pos']
                    suma_col2 = df['X']
                else:
                    suma_col1 += df['pos']
                    suma_col2 += df['X']
            
            # Calcular el promedio
            promedio_col1 = suma_col1 / n_archivos * (2/c)
            promedio_col2 = suma_col2 / n_archivos
            
            # Crear un DataFrame con los promedios
            df_promedio = pd.DataFrame({'pos': promedio_col1, 'X': promedio_col2})
            mean_temp = (max(temps_arch)+min(temps_arch))/2
            # Guardar el archivo resultante en la nueva carpeta
            archivo_salida = os.path.join(nueva_carpeta, f'Average_{round(mean_temp,2)}K.dat')
            df_promedio.to_csv(archivo_salida, index=False, sep=' ')
            print(f"Archivo {archivo_salida} generado en {nueva_carpeta}.")

        except Exception as e:
            print(f"Error al procesar los archivos con temperatura {round(mean_temp,2)}: {e}")


def getFilterdata(path_signal,right,left):

    df1 = pd.read_csv(path_signal, delim_whitespace=True)
    df1 = df1.loc[(df1.iloc[:, 0] >= left) & (df1.iloc[:, 0] <=right)]
    
    return df1['pos'] ,df1['X']


def getSignal(path_signal,right,left):

    df1 = pd.read_csv(path_signal, delim_whitespace=True)
    df1 = df1.loc[(df1.iloc[:, 0] >= left) & (df1.iloc[:, 0] <=right)]
    return df1['X']


def getSignalWindowed(path_signal, 
                     path_ref, 
                     left, 
                     right_signal,
                     right_subs, 
                     params_window=None):
    '''
    Alinea la señal al máximo del substrato y aplica una ventana centrada en ese máximo.

    Parameters
    ----------
    path_signal : str
        Ruta del archivo con la señal.
    path_ref : str
        Ruta del archivo con el substrato.
    left : float
        Límite izquierdo para la señal.
    right_signal : float
        Límite derecho de la señal.
    right_subs : float
        Límite derecho del substrato.
    params_window : list, optional
        Parámetros para aplicar la ventana. Si es None o vacío, no se aplica ventana.

    Returns
    -------
    tuple
        Si no hay ventana: (y, y_substrate)
        Si hay ventana: (desplazamiento, y_alineada, y_substrate, ventana_alineada)
    '''
    y = getSignal(path_signal, right_signal, left)
    y_substrate = getSignal(path_ref, right_subs, left)
    # print(f"Señal: {len(y)} puntos, Substrato: {len(y_substrate)} puntos")
    y = np.asarray(y)
    y_substrate = np.asarray(y_substrate)

    # Encontrar máximos
    idx_max_y = np.argmax(y)
    idx_max_subs = np.argmax(y_substrate)

    # Desplazar y para alinear su máximo al del substrato
    desplazamiento = idx_max_subs - idx_max_y
    
    # Función para balancear puntos izquierda/derecha del máximo
    def balance_signal(signal):
        idx_max = np.argmax(signal)
        left_points = idx_max  # Puntos desde inicio hasta máximo
        right_points = len(signal) - idx_max - 1  # Puntos desde máximo hasta final
        
        # Si hay más puntos a la derecha, rellenar con ceros a la izquierda
        if right_points > left_points:
            pad_size = right_points - left_points
            signal = np.pad(signal, (pad_size, 0), 'constant')
        else:
            pad_size = abs(right_points - left_points)
            signal = np.pad(signal, (0, pad_size), 'constant')

        
        return signal

    # Balancear ambas señales individualmente
    y = balance_signal(y)
    y_substrate = balance_signal(y_substrate)
    
    if not params_window:
        return y, y_substrate

    # # Asegurar mismos tamaños (por si hay diferencias residuales)
    # len_diff = len(y) - len(y_substrate)
    # if len_diff > 0:
    #     y_substrate = np.pad(y_substrate, (len_diff, 0), 'constant')
    # elif len_diff < 0:
    #     y = np.pad(y, (-len_diff, 0), 'constant')

    # # Encontrar máximos
    # idx_max_y = np.argmax(y)
    idx_max_subs = np.argmax(y_substrate)

    # # Desplazar y para alinear su máximo al del substrato
    # desplazamiento = idx_max_subs - idx_max_y
    
    y_alineada = np.roll(y, desplazamiento)

    # Aplicar ventana
    params_window_copia = list(params_window)
    params_window_copia.insert(1, len(y_substrate))  # insertar tamaño de la señal
    ventana = apply_window(params_window_copia)

    # Desplazar ventana para que su máximo coincida con el máximo común
    idx_max_ventana = np.argmax(ventana)
    desp_ventana = idx_max_subs - idx_max_ventana
    ventana_alineada = np.roll(ventana, desp_ventana)

    return desplazamiento, y_alineada, y_substrate, ventana_alineada
    
def getSignalWindowed2(path_signal, 
                     left, 
                     right_signal,
                     params_window=None):
    '''
    Alinea la señal al máximo del substrato y aplica una ventana centrada en ese máximo.

    Parameters
    ----------
    path_signal : str
        Ruta del archivo con la señal.
    path_ref : str
        Ruta del archivo con el substrato.
    left : float
        Límite izquierdo para la señal.
    right_signal : float
        Límite derecho de la señal.
    right_subs : float
        Límite derecho del substrato.
    params_window : list, optional
        Parámetros para aplicar la ventana. Si es None o vacío, no se aplica ventana.

    Returns
    -------
    tuple
        Si no hay ventana: (y, y_substrate)
        Si hay ventana: (desplazamiento, y_alineada, y_substrate, ventana_alineada)
    '''
    y = getSignal(path_signal, right_signal, left)
    # print(f"Señal: {len(y)} puntos, Substrato: {len(y_substrate)} puntos")
    y = np.asarray(y)

    # Encontrar máximos
    idx_max_y = np.argmax(y)

    
    # Función para balancear puntos izquierda/derecha del máximo
    def balance_signal(signal):
        idx_max = np.argmax(signal)
        left_points = idx_max  # Puntos desde inicio hasta máximo
        right_points = len(signal) - idx_max - 1  # Puntos desde máximo hasta final
        
        # Si hay más puntos a la derecha, rellenar con ceros a la izquierda
        if right_points > left_points:
            pad_size = right_points - left_points
            signal = np.pad(signal, (pad_size, 0), 'constant')
        else:
            pad_size = abs(right_points - left_points)
            signal = np.pad(signal, (0, pad_size), 'constant')
 
        return signal

    # Balancear ambas señales individualmente
    y = balance_signal(y)

    if not params_window:
        return y

    # # Desplazar y para alinear su máximo al del substrato
    # desplazamiento = idx_max_subs - idx_max_y

    # Aplicar ventana
    params_window_copia = list(params_window)
    params_window_copia.insert(1, len(y))  # insertar tamaño de la señal
    ventana = apply_window(params_window_copia)

    # Desplazar ventana para que su máximo coincida con el máximo común
    idx_max_ventana = np.argmax(ventana)
    desp_ventana = idx_max_y - idx_max_ventana
    ventana_alineada = np.roll(ventana, desp_ventana)

    return desp_ventana, y, ventana_alineada

def trans_model(nu,b,a,x0,gamma):

    return b-a/(1+(nu-x0)**2/gamma**2)

def fit_trans_model(x,y,x1,x2,p0=[]):

    filt = (x>x1) & (x<x2)
    x = x[filt]
    y = y[filt]

    bounds = ([0, 0, 0,0], [1, np.inf, np.inf, np.inf])

    if not p0 :
        poptBoth, pcovBoth = curve_fit(trans_model, x, y,maxfev=10000)
        return poptBoth

    else:
        poptBoth, pcovBoth = curve_fit(trans_model,x, y,p0=p0,bounds=bounds,maxfev=5000)
        return poptBoth
    
def E_THz(t,τs,τc,τp):
    # term1 = 
    term2 = τc * erfc((-t / τp) + (τp / (2 * τs))) * np.exp(-(t / τs) + (τp**2 / (4 * τs**2)))
    term3 = -τs * erfc((-t / τp) + (τp / (2 * τc))) * np.exp(-(t / τc) + (τp**2 / (4 * τc**2)))
    
    return (term2 + term3)


def extraer_humedad(nombre_archivo):
    match = re.search(r'(\d+\.\d+|\d+)RH', nombre_archivo)
    if match:
        return float(match.group(1))
    else:
        return None

def extrac_data_freq(nu1,nu2,path_air):

    df1 = pd.read_csv(path_air, delim_whitespace=True)
    c = 0.299792458 # speed of light mm/ps
    # suma_col1 = df1['pos']* (2/c)
    suma_col2 = df1['X'].values
    N = 2**12
    k = 15
    nu = sp.fft.fftfreq(N, 1/30)
    fourier = FourierT(suma_col2,N)[1:len(nu)//k]
    nu = nu[1:len(nu)//k]
    xmin, xmax = nu1, nu2
    mask = (nu >= xmin) & (nu <= xmax)
    nu_filtradas = nu[mask]
    fourier = fourier[mask]

    return nu_filtradas, fourier

def concat_trans(nu,parameters,index):

    s = 0

    for k in range(len(parameters)):

        s += trans_model(nu,*parameters[k][index])

        

    return s


def extrac_data_time(path_air):

    df1 = pd.read_csv(path_air, delim_whitespace=True)
    c = 0.299792458 # speed of light mm/ps
    suma_col1 = df1['pos']* (2/c)
    suma_col2 = df1['X'].values
   
    return suma_col1, suma_col2