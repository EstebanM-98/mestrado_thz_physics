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
from glob import glob
import matplotlib.gridspec as gridspec
from ipywidgets import interact, FloatSlider
from ipywidgets import FloatSlider, Button, HBox, VBox, HTML, interactive_output
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.animation as animation

# Función para extraer la tempertura del nombre del archivo
c = 0.299792458 # speed of light mm/ps
def extraer_temperatura(nombre_archivo):
    match = re.search(r'(\d+\.\d+|\d+)K', nombre_archivo)
    if match:
        return float(match.group(1))
    else:
        return None

def create_frequency_frame(
    index,
    archivos_dat_samp,
    archivos_dat_ref,
    getFilterdata,
    extraer_temperatura,
    FourierT2,
    left,
    right_sample,
    right_subs,
    axes,
    norm,
    cmap=cm.coolwarm,
    N=2**13,
    fft_fs=30,                  # frecuencia de muestreo para fftfreq
    fft_k=15,                   # truncamiento de FFT
    fft_xlim=(0.25, 1.0),       # ventana de frecuencias
    time_xlabel='Time (ps)',
    fft_xlabel=r"$\nu$ (THz)",
    fft_ylabel=r"$|FFT|^{2}$",
    fft_yscale='log',
    samp_label='Sam',
    legend_loc='lower right'
):
    ax1, ax2 = axes
    for ax in axes:
        ax.clear()

    try:
        # Paths y color
        path_signal = archivos_dat_samp[int(index)]
        temp = extraer_temperatura(path_signal)
        color = cmap(norm(temp)) if temp is not None else 'blue'
        path_ref = archivos_dat_ref[0]

        # Datos filtrados
        x, y = getFilterdata(path_signal, right_sample, left)
        x_ref, y_ref = getFilterdata(path_ref, right_subs, left)

        y = np.array(y)
        y_ref = np.array(y_ref)

        if np.max(np.abs(y)) == 0 or np.max(np.abs(y_ref)) == 0:
            raise ValueError("Señales vacías")

        # --- Subplot 1 ---
        ax1.plot(x, y, color=color, label=samp_label)
        ax1.set_xlabel(time_xlabel)
        ax1.legend(loc=legend_loc)

        # --- Subplot 2 ---
        y_subs = pd.Series(y_ref)
        y_signal = pd.Series(y)

        nu = sp.fft.fftfreq(N, 1/fft_fs)[:N//2]
        nu = nu[1:N//fft_k]

        fft_y_signal = FourierT2(y_signal, N)[1:len(nu)+1]
        fft_y_subs = FourierT2(y_subs, N)[1:len(nu)+1]

        xmin, xmax = fft_xlim
        mask = (nu >= xmin) & (nu <= xmax)
        nu_filtradas = nu[mask]
        fft_y_signal = fft_y_signal[mask]
        fft_y_subs = fft_y_subs[mask]

        if np.max(fft_y_signal) == 0 or np.max(fft_y_subs) == 0:
            raise ValueError("FFT vacía")

        ax2.plot(nu_filtradas, np.abs(fft_y_signal)**2, color=color, label=f'{temp}')
        ax2.set_ylabel(fft_ylabel)
        ax2.set_xlabel(fft_xlabel)
        ax2.set_yscale(fft_yscale)
        ax2.legend()

    except Exception as e:
        print(f"[WARN] Frame {index} falló: {e}")
        ax1.plot([], [])
        ax2.plot([], [])

    return ax1.lines + ax2.lines

def plot_windowed_frequency_samples(
    # datos y funciones (requeridos)
    archivos_dat_samp,
    archivos_dat_ref,
    getFilterdata,          # callable(path, right, left) -> (x, y)
    extraer_temperatura,    # callable(path) -> float|None
    FourierT2,              # callable(serie_pd, N) -> np.array FFT
    # recortes en tiempo
    left=None,
    right_sample=None,
    right_subs=None,
    # FFT / frecuencia
    N=2**13,
    fs=30.0,                # Hz para sp.fft.fftfreq
    k_trunc=15,             # recorte de altas: nu = nu[1:len(nu)//k_trunc]
    freq_xlim=(0.25, 1.0),  # ventana (THz) para mostrar/buscar mínimos
    # figura 1 (|T|^2 vs nu)
    figsize_fft=(5, 5),
    dpi_fft=200,
    cmap=cm.coolwarm,
    show_colorbar=True,
    cbar_pos=(0.88, 0.15, 0.02, 0.7),
    cbar_label='Temperature (K)',
    # mínimos a buscar
    min1_range=(0.25, 0.40),   # rango THz del 1er mínimo
    min2_range=(0.50, 1.00),   # rango THz del 2do mínimo
    # offsets/estética (por si quieres apilar señales en el tiempo más adelante)
    val_offset_signal=0.8,
    val_offset_kappa=0.03,
    # figura 2 (resúmenes vs temperatura)
    figsize_summary=(14, 8),
    dpi_summary=200,
    tn_line=None,              # e.g., 23.5 para dibujar T_N; None para ocultar
    tn_label=r"$T_N$",
    # etiquetas
    y_label_fft=r"$|T|^{2}$",
    x_label_fft=r"$\nu$ (THz)",
    y_label_min1="Transmittance Minimum",
    y_label_min2="Amplitude Minimum",
    x_label_temp="Temperature (K)",
    params_window1 = ['nuttall']
    
):
    """
    Grafica |T|^2(ν) por archivo (coloreado por temperatura) y extrae dos mínimos en
    rangos configurables, luego grafica frecuencia y amplitud de esos mínimos vs temperatura.

    Devuelve un dict con: temps, max1_freqs, max1_values, max2_freqs, max2_values.
    """

    # --- temperaturas válidas (en el mismo orden que los archivos) ---
    temps = [extraer_temperatura(p) for p in archivos_dat_samp if extraer_temperatura(p) is not None][::-1]
    if not temps:
        raise ValueError("No se encontraron temperaturas válidas en los archivos")

    vmin, vmax = min(temps), max(temps)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # --- Figura 1: |T|^2 (THz) para todas las muestras ---
    fig1, ax_fft = plt.subplots(1, 1, figsize=figsize_fft, dpi=dpi_fft)
    ax_fft.set_xlabel(x_label_fft)
    ax_fft.set_ylabel(y_label_fft)

    if show_colorbar:
        fig1.subplots_adjust(right=0.85)
        cbar_ax = fig1.add_axes(cbar_pos)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig1.colorbar(sm, cax=cbar_ax, label=cbar_label)

    # resultados
    max1_freqs, max1_values = [], []
    max2_freqs, max2_values = [], []

    # recorremos en el mismo orden (invertido como tu original)
    for index, path_signal in enumerate(archivos_dat_samp[::-1]):
        temp = temps[index]
        color = cmap(norm(temp)) if temp is not None else 'blue'
        path_ref = archivos_dat_ref[0]

        # recortes temporales (right primero, luego left — como tu función)
        phase, y_signal_ventaneada, y_substrate_padding, ventana = getSignalWindowed(
        path_signal, path_ref, left, right_sample, right_subs, params_window1)

        y_subs_ventana = pd.Series(y_substrate_padding * ventana)
        y_signal_ventaneada = pd.Series(y_signal_ventaneada * ventana)


        # FFT y kappa
        k = 15
        nu = sp.fft.fftfreq(N, 1 / 30)
        fft_y_signal_ventaneada = FourierT2(y_signal_ventaneada, N)[1:len(nu)//k]
        fft_y_subs_ventaneada = FourierT2(y_subs_ventana, N)[1:len(nu)//k]

        nu = nu[1:len(nu)//k]
        xmin, xmax = 0.25, 1.0
        mask = (nu >= xmin) & (nu <= xmax)

        fft_y_signal_ventaneada = fft_y_signal_ventaneada[mask]
        fft_y_subs_ventaneada = fft_y_subs_ventaneada[mask]
        nu_f = nu[mask]

        val_offset_kappa = 0.8
        offset_kappa = val_offset_kappa * index
        
        T = np.abs(fft_y_signal_ventaneada)**2/np.max(np.abs(fft_y_signal_ventaneada)**2)
        
        ax_fft.plot(nu_f, T, color=color)
        ax_fft.set_yscale('log')

        # mínimos en rangos configurables
        # 1er mínimo
        r1_min, r1_max = min1_range
        mask1 = (nu_f >= r1_min) & (nu_f <= r1_max)
        if np.any(mask1):
            idx1 = np.argmin(T[mask1])
            max1_freq = nu_f[mask1][idx1]
            max1_val  = T[mask1][idx1]
            ax_fft.scatter(max1_freq, max1_val, color='black', edgecolors=color, s=50, zorder=5)
        else:
            max1_freq, max1_val = np.nan, np.nan

        # 2do mínimo
        r2_min, r2_max = min2_range
        mask2 = (nu_f >= r2_min) & (nu_f <= r2_max)
        if np.any(mask2):
            idx2 = np.argmin(T[mask2])
            max2_freq = nu_f[mask2][idx2]
            max2_val  = T[mask2][idx2]
            ax_fft.scatter(max2_freq, max2_val, color='black', edgecolors=color, s=50, zorder=5)
        else:
            max2_freq, max2_val = np.nan, np.nan

        max1_freqs.append(max1_freq)
        max1_values.append(max1_val)
        max2_freqs.append(max2_freq)
        max2_values.append(max2_val)

    ax_fft.set_yscale('log')
    # trazas guía (opcionales)
    ax_fft.plot(max1_freqs, max1_values, 'k--', linewidth=1, alpha=0.5)
    ax_fft.plot(max2_freqs, max2_values, 'k--', linewidth=1, alpha=0.5)
    ax_fft.legend(loc='best')

    # --- Figura 2: resúmenes vs temperatura ---
    # === Dos filas, cada una con doble eje y ===
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=figsize_summary, dpi=dpi_summary, sharex=True)

    temps_plot = temps[::-1]
    ckw = dict(c=temps_plot, cmap='coolwarm', edgecolors='k', s=80, zorder=3)

    # -------- Fila 1: primer mínimo --------
    ax1 = ax_top
    ax1.scatter(temps_plot, max1_freqs[::-1], **ckw, label='First Frequency Min.')
    ax1.plot(temps_plot, max1_freqs[::-1], 'k--', alpha=0.3, linewidth=1)
    ax1.set_ylabel('First Frequency Min. (THz)')

    ax1b = ax1.twinx()
    ax1b.scatter(temps_plot, max1_values[::-1], **ckw, marker='s', label=y_label_min1)
    ax1b.plot(temps_plot, max1_values[::-1], 'k:', alpha=0.3, linewidth=1)
    ax1b.set_yscale('log')
    ax1b.set_ylabel(y_label_min1)

    if tn_line is not None:
        ax1.axvline(x=tn_line, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        ax1b.axvline(x=tn_line, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

    # Combinar leyendas de ambos ejes (fila 1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize='small')

    # -------- Fila 2: segundo mínimo --------
    ax3 = ax_bottom
    ax3.scatter(temps_plot, max2_freqs[::-1], **ckw, label='Second Frequency Min.')
    ax3.plot(temps_plot, max2_freqs[::-1], 'k--', alpha=0.3, linewidth=1)
    ax3.set_ylabel('Second Frequency Min. (THz)')
    ax3.set_xlabel(x_label_temp)

    ax3b = ax3.twinx()
    ax3b.scatter(temps_plot, max2_values[::-1], **ckw, marker='s', label=y_label_min2)
    ax3b.plot(temps_plot, max2_values[::-1], 'k:', alpha=0.3, linewidth=1)
    ax3b.set_yscale('log')
    ax3b.set_ylabel(y_label_min2)

    if tn_line is not None:
        ax3.axvline(x=tn_line, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        ax3b.axvline(x=tn_line, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

    # Combinar leyendas de ambos ejes (fila 2)
    h3, l3 = ax3.get_legend_handles_labels()
    h4, l4 = ax3b.get_legend_handles_labels()
    ax3.legend(h3 + h4, l3 + l4, loc='lower left', fontsize='small')

    plt.tight_layout()

    return {
        "temps": temps,
        "min1_freqs": max1_freqs,
        "min1_vals": max1_values,
        "min2_freqs": max2_freqs,
        "min2_vals": max2_values,
        "fig_fft": fig1,
        "fig_summary": fig,
    }


def plot_frequency_samples(
    # datos y funciones (requeridos)
    archivos_dat_samp,
    archivos_dat_ref,
    getFilterdata,          # callable(path, right, left) -> (x, y)
    extraer_temperatura,    # callable(path) -> float|None
    FourierT2,              # callable(serie_pd, N) -> np.array FFT
    # recortes en tiempo
    left=None,
    right_sample=None,
    right_subs=None,
    # FFT / frecuencia
    N=2**13,
    fs=30.0,                # Hz para sp.fft.fftfreq
    k_trunc=15,             # recorte de altas: nu = nu[1:len(nu)//k_trunc]
    freq_xlim=(0.25, 1.0),  # ventana (THz) para mostrar/buscar mínimos
    # figura 1 (|T|^2 vs nu)
    figsize_fft=(5, 5),
    dpi_fft=200,
    cmap=cm.coolwarm,
    show_colorbar=True,
    cbar_pos=(0.88, 0.15, 0.02, 0.7),
    cbar_label='Temperature (K)',
    # mínimos a buscar
    min1_range=(0.25, 0.40),   # rango THz del 1er mínimo
    min2_range=(0.50, 1.00),   # rango THz del 2do mínimo
    # offsets/estética (por si quieres apilar señales en el tiempo más adelante)
    val_offset_signal=0.8,
    val_offset_kappa=0.03,
    # figura 2 (resúmenes vs temperatura)
    figsize_summary=(14, 8),
    dpi_summary=200,
    tn_line=None,              # e.g., 23.5 para dibujar T_N; None para ocultar
    tn_label=r"$T_N$",
    # etiquetas
    y_label_fft=r"$|T|^{2}$",
    x_label_fft=r"$\nu$ (THz)",
    y_label_min1="Transmittance Minimum",
    y_label_min2="Amplitude Minimum",
    x_label_temp="Temperature (K)",
):
    """
    Grafica |T|^2(ν) por archivo (coloreado por temperatura) y extrae dos mínimos en
    rangos configurables, luego grafica frecuencia y amplitud de esos mínimos vs temperatura.

    Devuelve un dict con: temps, max1_freqs, max1_values, max2_freqs, max2_values.
    """

    # --- temperaturas válidas (en el mismo orden que los archivos) ---
    temps = [extraer_temperatura(p) for p in archivos_dat_samp if extraer_temperatura(p) is not None][::-1]
    if not temps:
        raise ValueError("No se encontraron temperaturas válidas en los archivos")

    vmin, vmax = min(temps), max(temps)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # --- Figura 1: |T|^2 (THz) para todas las muestras ---
    fig1, ax_fft = plt.subplots(1, 1, figsize=figsize_fft, dpi=dpi_fft)
    ax_fft.set_xlabel(x_label_fft)
    ax_fft.set_ylabel(y_label_fft)

    if show_colorbar:
        fig1.subplots_adjust(right=0.85)
        cbar_ax = fig1.add_axes(cbar_pos)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig1.colorbar(sm, cax=cbar_ax, label=cbar_label)

    # resultados
    max1_freqs, max1_values = [], []
    max2_freqs, max2_values = [], []

    # recorremos en el mismo orden (invertido como tu original)
    for index, path_signal in enumerate(archivos_dat_samp[::-1]):
        temp = temps[index]
        color = cmap(norm(temp)) if temp is not None else 'blue'
        path_ref = archivos_dat_ref[0]

        # recortes temporales (right primero, luego left — como tu función)
        x, y = getFilterdata(path_signal, right_sample, left)
        x_ref, y_ref = getFilterdata(path_ref, right_subs, left)

        # series para FFT
        y_signal = pd.Series(np.asarray(y, dtype=float))
        y_subs   = pd.Series(np.asarray(y_ref, dtype=float))

        # FFT
        nu = sp.fft.fftfreq(N, 1.0/fs)
        fft_y_signal = FourierT2(y_signal, N)
        fft_y_subs   = FourierT2(y_subs,   N)

        # recorte DC y truncado altas
        nu = nu[1:len(nu)//k_trunc]
        fft_y_signal = fft_y_signal[1:len(nu)+1]
        fft_y_subs   = fft_y_subs[1:len(nu)+1]

        # ventana de frecuencias para ploteo/búsqueda
        xmin, xmax = freq_xlim
        mask = (nu >= xmin) & (nu <= xmax)
        nu_f = nu[mask]
        T = np.abs(fft_y_signal[mask] / fft_y_subs[mask])**2
        ax_fft.plot(nu_f, T, color=color)

        # mínimos en rangos configurables
        # 1er mínimo
        r1_min, r1_max = min1_range
        mask1 = (nu_f >= r1_min) & (nu_f <= r1_max)
        if np.any(mask1):
            idx1 = np.argmin(T[mask1])
            max1_freq = nu_f[mask1][idx1]
            max1_val  = T[mask1][idx1]
            ax_fft.scatter(max1_freq, max1_val, color='black', edgecolors=color, s=50, zorder=5)
        else:
            max1_freq, max1_val = np.nan, np.nan

        # 2do mínimo
        r2_min, r2_max = min2_range
        mask2 = (nu_f >= r2_min) & (nu_f <= r2_max)
        if np.any(mask2):
            idx2 = np.argmin(T[mask2])
            max2_freq = nu_f[mask2][idx2]
            max2_val  = T[mask2][idx2]
            ax_fft.scatter(max2_freq, max2_val, color='black', edgecolors=color, s=50, zorder=5)
        else:
            max2_freq, max2_val = np.nan, np.nan

        max1_freqs.append(max1_freq)
        max1_values.append(max1_val)
        max2_freqs.append(max2_freq)
        max2_values.append(max2_val)

    ax_fft.set_yscale('log')
    # trazas guía (opcionales)
    ax_fft.plot(max1_freqs, max1_values, 'k--', linewidth=1, alpha=0.5)
    ax_fft.plot(max2_freqs, max2_values, 'k--', linewidth=1, alpha=0.5)
    ax_fft.legend(loc='best')

    # --- Figura 2: resúmenes vs temperatura ---
    fig2, axs = plt.subplots(2, 2, figsize=figsize_summary, dpi=dpi_summary)
    ax1, ax2, ax3, ax4 = axs.flatten()

    # Usamos el mismo mapeo de color por temperatura
    temps_plot = temps[::-1]
    ckw = dict(c=temps_plot, cmap='coolwarm', edgecolors='k', s=80, zorder=3)

    # Frecuencia del primer mínimo
    ax1.scatter(temps_plot, max1_freqs[::-1], **ckw)
    ax1.plot(temps_plot, max1_freqs[::-1], 'k--', alpha=0.3, linewidth=1)
    ax1.set_ylabel('First Frequency Min. (THz)')
    ax1.set_xlabel(x_label_temp)
    if tn_line is not None:
        ax1.axvline(x=tn_line, label=f'{tn_label} = {tn_line}', color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        ax1.legend(loc='upper right', fontsize='small')

    # Amplitud del primer mínimo
    ax2.scatter(temps_plot, max1_values[::-1], **ckw)
    ax2.plot(temps_plot, max1_values[::-1], 'k--', alpha=0.3, linewidth=1)
    ax2.set_yscale('log')
    ax2.set_ylabel(y_label_min1)
    ax2.set_xlabel(x_label_temp)
    if tn_line is not None:
        ax2.axvline(x=tn_line, label=f'{tn_label} = {tn_line}', color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        ax2.legend(loc='upper right', fontsize='small')

    # Frecuencia del segundo mínimo
    ax3.scatter(temps_plot, max2_freqs[::-1], **ckw)
    ax3.plot(temps_plot, max2_freqs[::-1], 'k--', alpha=0.3, linewidth=1)
    ax3.set_ylabel('Second Frequency Min. (THz)')
    ax3.set_xlabel(x_label_temp)
    if tn_line is not None:
        ax3.axvline(x=tn_line, label=f'{tn_label} = {tn_line}', color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        ax3.legend(loc='upper right', fontsize='small')

    # Amplitud del segundo mínimo
    ax4.scatter(temps_plot, max2_values[::-1], **ckw)
    ax4.plot(temps_plot, max2_values[::-1], 'k--', alpha=0.3, linewidth=1)
    ax4.set_yscale('log')
    ax4.set_ylabel(y_label_min2)
    ax4.set_xlabel(x_label_temp)
    if tn_line is not None:
        ax4.axvline(x=tn_line, label=f'{tn_label} = {tn_line}', color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        ax4.legend(loc='upper right', fontsize='small')

    plt.tight_layout()

    return {
        "temps": temps,
        "min1_freqs": max1_freqs,
        "min1_vals": max1_values,
        "min2_freqs": max2_freqs,
        "min2_vals": max2_values,
        "fig_fft": fig1,
        "fig_summary": fig2,
    }



def generate_frequency_gif(
    n0,
    n1,
    n2,
    archivos_dat_samp,
    archivos_dat_ref,
    getFilterdata,
    extraer_temperatura,
    FourierT2,
    save_path='animacion_espectro.gif',
    figsize=(12, 5),
    dpi=200,
    cmap=cm.coolwarm,
    cbar_pos=[0.92, 0.15, 0.02, 0.7],
    cbar_label='Temperatura (K)',
    interval=400,
    blit=False,
    fps=2,
    **frame_kwargs            # argumentos adicionales que se pasan a create_frame
):
    # Calcular rango de temperaturas
    temps = [extraer_temperatura(p) for p in archivos_dat_samp if extraer_temperatura(p) is not None]
    if not temps:
        raise ValueError("No se encontraron temperaturas válidas.")

    min_temp, max_temp = min(temps), max(temps)
    norm = Normalize(vmin=min_temp, vmax=max_temp)

    left = n0
    right_sample = n1
    right_subs = n2

    # Figura y ejes
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    axes = [ax1, ax2]

    # Barra de color
    cbar_ax = fig.add_axes(cbar_pos)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cbar_ax, label=cbar_label)

    def animate(i):
        return create_frequency_frame(
            index=i,
            archivos_dat_samp=archivos_dat_samp,
            archivos_dat_ref=archivos_dat_ref,
            getFilterdata=getFilterdata,
            extraer_temperatura=extraer_temperatura,
            FourierT2=FourierT2,
            left=left,
            right_sample=right_sample,
            right_subs=right_subs,
            axes=axes,
            norm=norm,
            cmap=cmap,
            **frame_kwargs
        )

    # Número de frames
    num_frames = len(archivos_dat_samp)
    if num_frames == 0:
        raise ValueError("No hay archivos para animar")

    # Animación
    ani = animation.FuncAnimation(
        fig, animate,
        frames=num_frames,
        interval=interval,
        blit=blit
    )

    # Guardar
    ani.save(save_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close()
    

def create_temporal_frame(
    index,
    archivos_dat_samp,
    archivos_dat_ref,
    getFilterdata,
    extraer_temperatura,
    left,
    right_sample,
    right_subs,
    axes,
    norm,
    cmap=cm.coolwarm,
    offset_val=1.0,
    ref_index=0,
    ref_style='--k',
    ref_label='Ref',
    samp_label='Sam',
    vline_color='red',
    vline_style='--',
    vline_alpha=0.5,
    legend_loc='lower right'
):
    ax1 = axes[0]  # Solo trabajamos con el primer eje
    ax1.clear()

    # Paths y colores
    path_signal = archivos_dat_samp[int(index)]
    temp = extraer_temperatura(path_signal)
    color = cmap(norm(temp)) if temp is not None else 'blue'
    path_ref = archivos_dat_ref[ref_index]

    # Datos (sin filtrar en este ejemplo)
    x, y = getFilterdata(path_signal)
    x_ref, y_ref = getFilterdata(path_ref)

    # Señales en dominio temporal
    line1 = ax1.plot(x, y / max(y) + offset_val, color=color, label=samp_label)[0]
    line2 = ax1.plot(x_ref, y_ref / max(y_ref), ref_style, label=ref_label)[0]

    # Líneas verticales
    line_a = ax1.axvline(right_sample, color=vline_color, linestyle=vline_style, alpha=vline_alpha)
    line_b = ax1.axvline(left, color=vline_color, linestyle=vline_style, alpha=vline_alpha)

    ax1.set_title(f"{int(extraer_temperatura(path_signal))} K")
    ax1.legend(loc=legend_loc)

    return [line1, line2, line_a, line_b]


def generate_temporal_gif(
    n0,
    n1,
    n2,
    archivos_dat_samp,
    archivos_dat_ref,
    getFilterdata,
    extraer_temperatura,
    save_path='animacion.gif',
    figsize=(7, 4),
    dpi=200,
    cmap=cm.coolwarm,
    cbar_pos=[0.92, 0.15, 0.02, 0.7],
    cbar_label='Temperature (K)',
    interval=400,
    blit=True,
    fps=2,
    offset_val=1.0,
    ref_index=0,
    ref_style='--k',
    ref_label='Ref',
    samp_label='Sam',
    vline_color='red',
    vline_style='--',
    vline_alpha=0.5,
    legend_loc='lower right'
):
    # Rango de temperaturas
    temps = [extraer_temperatura(p) for p in archivos_dat_samp if extraer_temperatura(p) is not None]
    if not temps:
        raise ValueError("No se encontraron temperaturas válidas en los archivos")

    min_temp, max_temp = min(temps), max(temps)
    norm = Normalize(vmin=min_temp, vmax=max_temp)

    left = n0
    right_sample = n1
    right_subs = n2

    # Figura y ejes
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0, 0])
    axes = [ax1]

    # Colorbar
    cbar_ax = fig.add_axes(cbar_pos)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cbar_ax, label=cbar_label)

    # Función de animación
    def animate(i):
        return create_temporal_frame(
            i,
            archivos_dat_samp=archivos_dat_samp,
            archivos_dat_ref=archivos_dat_ref,
            getFilterdata=getFilterdata,
            extraer_temperatura=extraer_temperatura,
            left=left,
            right_sample=right_sample,
            right_subs=right_subs,
            axes=axes,
            norm=norm,
            cmap=cmap,
            offset_val=offset_val,
            ref_index=ref_index,
            ref_style=ref_style,
            ref_label=ref_label,
            samp_label=samp_label,
            vline_color=vline_color,
            vline_style=vline_style,
            vline_alpha=vline_alpha,
            legend_loc=legend_loc
        )

    # Número de frames
    num_frames = len(archivos_dat_samp)
    if num_frames == 0:
        raise ValueError("No hay archivos para animar")

    # Animación
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=num_frames,
        interval=interval,
        blit=blit
    )

    # Guardar
    ani.save(save_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close()

def plot_all_samples(
    left,
    right_sample,
    archivos_dat_samp,
    getFilterdata,
    extraer_temperatura,
    figsize=(8, 4),
    dpi=200,
    cmap=cm.coolwarm,
    offset_factor=0.3,         # factor de desplazamiento vertical
    field_label="Normalized field + offset",
    xlabel="Time (arb. u.)",
    title="Transmitted field",
    colorbar_label="Temperature (K)",
    invert_order=True          # invertir el orden de los archivos
):
    """
    Dibuja todas las señales procesadas con desplazamiento y coloreadas por temperatura.

    Parámetros
    ----------
    left : float
        Parámetro 'left' para getFilterdata.
    right_sample : float
        Parámetro 'right_sample' para getFilterdata.
    archivos_dat_samp : list[str]
        Lista de rutas de archivos .dat de las muestras.
    getFilterdata : callable
        Función para obtener (x, y) desde un archivo y parámetros.
    extraer_temperatura : callable
        Función que devuelve la temperatura desde el nombre del archivo.
    figsize : tuple, opcional
        Tamaño de la figura.
    dpi : int, opcional
        Resolución en DPI de la figura.
    cmap : matplotlib colormap, opcional
        Mapa de colores para la temperatura.
    offset_factor : float, opcional
        Multiplicador para el desplazamiento vertical entre trazas.
    field_label : str, opcional
        Etiqueta del eje Y.
    xlabel : str, opcional
        Etiqueta del eje X.
    title : str, opcional
        Título de la gráfica.
    colorbar_label : str, opcional
        Etiqueta de la barra de colores.
    invert_order : bool, opcional
        Si True, invierte el orden de los archivos y temperaturas.
    """

    # Extraer temperaturas válidas
    temps = [extraer_temperatura(p) for p in archivos_dat_samp if extraer_temperatura(p) is not None]
    if not temps:
        raise ValueError("No se encontraron temperaturas válidas en los archivos")

    if invert_order:
        temps = temps[::-1]
        archivos_dat_samp_plot = archivos_dat_samp[::-1]
    else:
        archivos_dat_samp_plot = archivos_dat_samp

    # Configuración de colores según temperatura
    min_temp, max_temp = min(temps), max(temps)
    norm = Normalize(vmin=min_temp, vmax=max_temp)

    # Crear figura
    fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax1.set_ylabel(field_label)
    ax1.set_xlabel(xlabel)
    ax1.set_title(title)

    val = 0  # desplazamiento acumulado
    for index, path_signal in enumerate(archivos_dat_samp_plot):
        temp = temps[index]
        color = cmap(norm(temp)) if temp is not None else 'blue'

        # Obtener datos filtrados
        x, y = getFilterdata(path_signal, right_sample, left)

        # Graficar señal desplazada
        ax1.plot(x, y + val, color=color)
        val += offset_factor * max(y)  # incremento desplazamiento

    # Barra de color lateral
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cbar_ax, label=colorbar_label)

    plt.show()

    
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



def make_anim_widget(
    archivos_dat_samp,
    archivos_dat_ref,
    getFilterdata,          # callable(path, right, left) -> (x, y)
    FourierT2,              # callable(serie, N) -> FFT (array-like)
    extraer_temperatura,    # callable(path_str) -> float | None
    # ---- parámetros “quemados” ahora ajustables ----
    N=2**15,
    fs=30.0,                # Hz (para sp.fft.fftfreq)
    k_trunc=15,             # recorte de alta frecuencia: nu = nu[1:len(nu)//k_trunc]
    freq_window=(0.15, 1.0),# (xmin, xmax) THz
    figsize=(10, 8),
    dpi=200,
    # sliders (rangos y valores por defecto)
    left_range=(320.0, 423.0, 396.7),         # (min, max, default)
    right_sample_range=(380.0, 450.0, 406.0),
    right_subs_range=(380.0, 450.0, 403.2),
    index_default=0,
    # visual
    use_log_absorption=True,
    norm_time_traces=True,  # normaliza señales temporales por su máximo
    color_samp='b',
    color_ref='r',
    label_samp='Sam',
    label_ref='Ref',
    # --- guardado de selección ---
    save_globals=True,
    var_names=('n0','n1','n2'),   # nombres para left, right_sample, right_subs
    on_save=None,                 # callback opcional: on_save(dict(left=..., right_sample=..., right_subs=...))
    # --- NUEVO destino de escritura ---
    write_to="user_ns",           # "user_ns" | "module" | "custom"
    custom_ns=None,               # dict si write_to="custom"
):
    """
    Crea y muestra un widget interactivo. Incluye botón "Guardar selección" que escribe
    n0, n1, n2 en:
      - user_ns del notebook (por defecto), o
      - globals del módulo (write_to="module"), o
      - un dict custom (write_to="custom", pasar custom_ns)

    Devuelve (ui, sliders_dict).
    """

    xmin, xmax = freq_window

    # -------- sliders explícitos --------
    opc = dict(continuous_update=False, readout_format=".3f")
    left_slider  = FloatSlider(min=left_range[0],  max=left_range[1],  value=left_range[2],  step=0.1, description='left', **opc)
    rs_slider    = FloatSlider(min=right_sample_range[0], max=right_sample_range[1], value=right_sample_range[2], step=0.1, description='right_sample', **opc)
    rr_slider    = FloatSlider(min=right_subs_range[0],   max=right_subs_range[1],   value=right_subs_range[2],   step=0.1, description='right_subs', **opc)
    idx_slider   = FloatSlider(min=0, max=max(0, len(archivos_dat_samp)-1), value=index_default, step=1, description='index', **opc)

    status = HTML(value="")
    btn_save = Button(description="Guardar selección", button_style='success', icon='save')

    # -------- función de dibujo --------
    def _anim2(left, right_sample, right_subs, index):
        right_ref = right_subs

        f = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(2, 2)

        ax1 = plt.subplot(gs[0, :])  # señales temporales
        ax2 = plt.subplot(gs[1, 0])  # FFTs
        ax3 = plt.subplot(gs[1, 1])  # |T|^2

        # paths seleccionados
        idx = int(index)
        idx = max(0, min(idx, len(archivos_dat_samp) - 1))
        path_signal = archivos_dat_samp[idx]
        path_ref = archivos_dat_ref[0]  # usa el primero por defecto
        print(path_signal)

        # datos filtrados
        x, y = getFilterdata(path_signal, right_sample, left)
        x_ref, y_ref = getFilterdata(path_ref, right_ref, left)

        y_signal = pd.Series(y, dtype=float)
        y_subs = pd.Series(y_ref, dtype=float)

        # trazas temporales
        if norm_time_traces:
            max_y = np.max(np.abs(y)) or 1.0
            max_r = np.max(np.abs(y_ref)) or 1.0
            ax1.plot(x,     y/max_y,  color_samp, label=label_samp)
            ax1.plot(x_ref, y_ref/max_r, color_ref, label=label_ref)
        else:
            ax1.plot(x,     y,     color_samp, label=label_samp)
            ax1.plot(x_ref, y_ref, color_ref, label=label_ref)

        ax1.set_xlabel('Time')
        ax1.set_ylabel('field')
        ax1.set_title(f"{extraer_temperatura(str(path_signal))}")
        ax1.legend(loc='lower right')

        # ejes de frecuencia
        nu = sp.fft.fftfreq(N, d=1.0/fs)
        nu = nu[1:len(nu)//k_trunc]  # descarta DC y trunca altas
        # FFT (mismo recorte)
        fft_y_signal = FourierT2(y_signal, N)[1:len(nu)+1]
        fft_y_subs   = FourierT2(y_subs,   N)[1:len(nu)+1]

        # ventana de frecuencias
        mask = (nu >= xmin) & (nu <= xmax)
        nu_f = nu[mask]
        fft_s = fft_y_signal[mask]
        fft_r = fft_y_subs[mask]

        # subplot FFT
        ax2.plot(nu_f, np.abs(fft_s)**2, color_samp, label=label_samp)
        ax2.set_ylabel(r"$|FFT|^{2}$")
        ax2.set_xlabel(r"$\nu$ (THz)")
        ax2.legend()

        # Transmitancia y |T|^2
        T = fft_s / fft_r
        ax3.plot(nu_f, np.abs(T)**2, color_samp)
        ax3.set_ylabel(r"$|T|^2$")
        ax3.set_xlabel(r"$\nu$ (THz)")
        ax3.set_title("Absorption Spectrum")
        if use_log_absorption:
            ax3.set_yscale('log')

        plt.tight_layout()

    out = interactive_output(
        _anim2,
        {
            'left': left_slider,
            'right_sample': rs_slider,
            'right_subs': rr_slider,
            'index': idx_slider
        }
    )

    # -------- lógica de guardado --------
    def _write_values(vals):
        # decide destino
        if write_to == "user_ns":
            try:
                from IPython import get_ipython
                ip = get_ipython()
            except Exception:
                ip = None
            if ip is not None and hasattr(ip, "user_ns"):
                ip.user_ns[var_names[0]] = vals['left']
                ip.user_ns[var_names[1]] = vals['right_sample']
                ip.user_ns[var_names[2]] = vals['right_subs']
                return True
            # si no hay IPython, cae al módulo
        if write_to == "custom" and isinstance(custom_ns, dict):
            custom_ns[var_names[0]] = vals['left']
            custom_ns[var_names[1]] = vals['right_sample']
            custom_ns[var_names[2]] = vals['right_subs']
            return True

        # por defecto escribe en el módulo (functions_master)
        globals()[var_names[0]] = vals['left']
        globals()[var_names[1]] = vals['right_sample']
        globals()[var_names[2]] = vals['right_subs']
        return False  # indicó que fue al módulo

    def _on_save(_):
        vals = dict(
            left=left_slider.value,
            right_sample=rs_slider.value,
            right_subs=rr_slider.value
        )
        wrote_to_user_ns = False
        if save_globals and isinstance(var_names, (tuple, list)) and len(var_names) == 3:
            wrote_to_user_ns = _write_values(vals)
        if callable(on_save):
            on_save(vals)
        destino = "notebook (user_ns)" if wrote_to_user_ns else ("custom_ns" if write_to=="custom" else "módulo")
        status.value = f"<b>Guardado en {destino}:</b> {vals}"

    btn_save.on_click(_on_save)

    ui = VBox([
        HBox([left_slider, rs_slider, rr_slider, idx_slider]),
        HBox([btn_save, status]),
        out
    ])

    # Muestra y también devuelve referencias por si quieres leer los sliders desde Python
    display(ui)
    sliders = dict(left=left_slider, right_sample=rs_slider, right_subs=rr_slider, index=idx_slider)
    return ui, sliders


def make_anim_window(
    # --- datos y funciones requeridas ---
    archivos_dat_samp,
    archivos_dat_ref,
    getSignalWindowed,      # callable(path_signal, path_ref, left, right_sample, right_subs, params_window1) -> (phase, y_signal_vent, y_subs_pad, ventana)
    FourierT2,              # callable(serie_pd, N) -> FFT (array-like)
    extraer_temperatura,    # callable(path_str) -> float | None

    # --- parámetros “quemados” ahora configurables ---
    N=2**15,
    fs=30.0,                # para sp.fft.fftfreq
    k_trunc=15,             # truncar altas: nu = nu[1:len(nu)//k_trunc]
    freq_window=(0.25, 1.0),
    figsize=(10, 8),
    dpi=200,
    offset_val=2.0,         # desplazamiento en el trazo de la muestra en tiempo
    yscale_fft='log',       # 'linear' o 'log'
    show_title=True,

    # ventana (lo que antes era ['gaussian', desv])
    params_window=['blackman'],

    # --- rangos de sliders (min, max, default, step) ---
    left_range=(320.0, 423.0, 392.5, 0.1),
    right_sample_range=(380.0, 450.0, 424.1, 0.1),
    right_subs_range=(380.0, 450.0, 433.4, 0.1),
    index_default=0,   # se ajusta internamente al [0, len(archivos_dat_samp)-1]
    desv_range=(0.0, 1000.0, 150.0, 0.001),
    d_range=(0.01, 1.0, 0.627, 0.01),
    # global variables
    # --- guardado de selección ---
    save_globals=True,
    var_names=('nw0','nw1','nw2'),   # nombres para left, right_sample, right_subs
    on_save=None,                 # callback opcional: on_save(dict(left=..., right_sample=..., right_subs=...))
        # --- NUEVO destino de escritura ---
    write_to="user_ns",           # "user_ns" | "module" | "custom"
    custom_ns=None,               # dict si write_to="custom"
):
    """
    Crea y muestra un widget interactivo equivalente a tu 'anim3', pero parametrizable.

    Retorna (ui, sliders) para poder leer los valores desde Python si lo necesitas:
        sliders['left'].value, sliders['right_sample'].value, ...
    """

    # ---- sliders ----
    opc = dict(continuous_update=False, readout_format=".3f")
    left_slider        = FloatSlider(min=left_range[0],  max=left_range[1],  value=left_range[2],  step=left_range[3],  description='left', **opc)
    right_samp_slider  = FloatSlider(min=right_sample_range[0], max=right_sample_range[1], value=right_sample_range[2], step=right_sample_range[3], description='right_sample', **opc)
    right_subs_slider  = FloatSlider(min=right_subs_range[0],   max=right_subs_range[1],   value=right_subs_range[2],   step=right_subs_range[3], description='right_subs', **opc)
    idx_slider         = FloatSlider(min=0, max=max(0, len(archivos_dat_samp)-1), value=index_default, step=1, description='index', **opc)
    desv_slider        = FloatSlider(min=desv_range[0], max=desv_range[1], value=desv_range[2], step=desv_range[3], description='desv', **opc)
    d_slider           = FloatSlider(min=d_range[0],    max=d_range[1],    value=d_range[2],    step=d_range[3],    description='d', **opc)

    status = HTML(value="")
    btn_save = Button(description="Guardar selección", button_style='success', icon='save')

    # ---- función de dibujo: misma lógica que tu anim3, pero usando args ----
    def _anim3(left, right_sample, right_subs, index, desv, d):
        right_ref = right_subs

        f = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(1, 2)

        ax1 = plt.subplot(gs[0, 0])  # temporal
        ax2 = plt.subplot(gs[0, 1])  # FFT
        # ax3 = plt.subplot(gs[1, :])  # (reservado por si quieres algo extra luego)

        # archivos
        idx = int(index)
        idx = max(0, min(idx, len(archivos_dat_samp)-1))
        path_signal = archivos_dat_samp[idx]
        path_ref = archivos_dat_ref[0]

        # ventana/filtrado
        # Mantengo tu interfaz: params_window1 = ['gaussian', desv]
        # Si quieres usar 'd' también en tu getSignalWindowed, pásalo aquí:
        params_window1 = params_window

        phase, y_signal_ventaneada, y_substrate_padding, ventana = getSignalWindowed(
            path_signal, path_ref, left, right_sample, right_ref, params_window1
        )

        # series
        y_subs_ventana = pd.Series(y_substrate_padding * ventana)
        y_signal_vent = pd.Series(y_signal_ventaneada * ventana)

        # --- subplot 1: dominio temporal ---
        ax1.plot(y_subs_ventana / (np.max((y_subs_ventana)) or 1.0), 'r', label='Ref')
        ax1.plot(ventana / (np.max(np.abs(ventana)) or 1.0), 'k--', label='Window')
        ax1.plot(y_signal_vent / (np.max((y_signal_vent)) or 1.0) + offset_val, 'b',
                 label=f'Sam+{extraer_temperatura(str(path_signal))} K')
        ax1.plot(ventana / (np.max(np.abs(ventana)) or 1.0) + offset_val, 'k--')
        ax1.legend(loc='lower right')

        # --- frecuencias ---
        nu = sp.fft.fftfreq(N, d=1.0/fs)
        nu = nu[1:len(nu)//k_trunc]

        fft_y_signal = FourierT2(y_signal_vent, N)[1:len(nu)+1]
        fft_y_subs   = FourierT2(y_subs_ventana, N)[1:len(nu)+1]

        xmin, xmax = freq_window
        mask = (nu >= xmin) & (nu <= xmax)
        nu_f = nu[mask]
        fft_s = fft_y_signal[mask]
        fft_r = fft_y_subs[mask]

        # --- subplot 2: FFTs normalizadas ---
        ax2.plot(nu_f, np.abs(fft_s)**2 / (np.max(np.abs(fft_s)**2) or 1.0), 'b', label='Sample FFT')
        ax2.plot(nu_f, np.abs(fft_r)**2 / (np.max(np.abs(fft_r)**2) or 1.0), 'r', label='Reference FFT')
        if show_title:
            ax2.set_title("Frequency Domain")
        ax2.set_ylabel(r"$|FFT|^{2}$")
        ax2.set_xlabel(r"$\nu$ (THz)")
        if yscale_fft:
            ax2.set_yscale(yscale_fft)
        ax2.legend()

        plt.tight_layout()
        status.value = f"<b>idx:</b> {idx} | <b>left:</b> {left:.3f} | <b>right_samp:</b> {right_sample:.3f} | <b>right_subs:</b> {right_subs:.3f} | <b>desv:</b> {desv:.3f} | <b>d:</b> {d:.3f}"

    # conectar sliders -> figura
    out = interactive_output(
        _anim3,
        dict(
            left=left_slider,
            right_sample=right_samp_slider,
            right_subs=right_subs_slider,
            index=idx_slider,
            desv=desv_slider,
            d=d_slider
        )
    )

    # -------- lógica de guardado --------
    def _write_values(vals):
        # decide destino
        if write_to == "user_ns":
            try:
                from IPython import get_ipython
                ip = get_ipython()
            except Exception:
                ip = None
            if ip is not None and hasattr(ip, "user_ns"):
                ip.user_ns[var_names[0]] = vals['left']
                ip.user_ns[var_names[1]] = vals['right_sample']
                ip.user_ns[var_names[2]] = vals['right_subs']
                return True
            # si no hay IPython, cae al módulo
        if write_to == "custom" and isinstance(custom_ns, dict):
            custom_ns[var_names[0]] = vals['left']
            custom_ns[var_names[1]] = vals['right_sample']
            custom_ns[var_names[2]] = vals['right_subs']
            return True

        # por defecto escribe en el módulo (functions_master)
        globals()[var_names[0]] = vals['left']
        globals()[var_names[1]] = vals['right_sample']
        globals()[var_names[2]] = vals['right_subs']
        return False  # indicó que fue al módulo

    def _on_save(_):
        vals = dict(
            left=left_slider.value,
            right_sample=right_samp_slider.value,
            right_subs=right_subs_slider.value
        )
        wrote_to_user_ns = False
        if save_globals and isinstance(var_names, (tuple, list)) and len(var_names) == 3:
            wrote_to_user_ns = _write_values(vals)
        if callable(on_save):
            on_save(vals)
        destino = "notebook (user_ns)" if wrote_to_user_ns else ("custom_ns" if write_to=="custom" else "módulo")
        status.value = f"<b>Guardado en {destino}:</b> {vals}"

    btn_save.on_click(_on_save)

    ui = VBox([
        HBox([left_slider, right_samp_slider, right_subs_slider, idx_slider]),
        HBox([btn_save, status]),
        status,
        out
    ])

    display(ui)
    sliders = dict(
        left=left_slider,
        right_sample=right_samp_slider,
        right_subs=right_subs_slider,
        index=idx_slider,
        desv=desv_slider,
        d=d_slider
    )
    return ui, sliders

def plot_all_windowed_samples(
    left,
    right_sample,
    right_ref,
    archivos_dat_samp,
    archivos_dat_ref,
    figsize=(8, 4),
    dpi=200,
    cmap=cm.coolwarm,
    offset_factor=0.3,         # factor de desplazamiento vertical
    field_label="Normalized field + offset",
    xlabel="Time (arb. u.)",
    title="Transmitted field",
    colorbar_label="Temperature (K)",
    invert_order=True,          # invertir el orden de los archivos
    params_window1 = ['nuttall']
):
    """
    Dibuja todas las señales procesadas con desplazamiento y coloreadas por temperatura.

    Parámetros
    ----------
    left : float
        Parámetro 'left' para getFilterdata.
    right_sample : float
        Parámetro 'right_sample' para getFilterdata.
    archivos_dat_samp : list[str]
        Lista de rutas de archivos .dat de las muestras.
    getFilterdata : callable
        Función para obtener (x, y) desde un archivo y parámetros.
    extraer_temperatura : callable
        Función que devuelve la temperatura desde el nombre del archivo.
    figsize : tuple, opcional
        Tamaño de la figura.
    dpi : int, opcional
        Resolución en DPI de la figura.
    cmap : matplotlib colormap, opcional
        Mapa de colores para la temperatura.
    offset_factor : float, opcional
        Multiplicador para el desplazamiento vertical entre trazas.
    field_label : str, opcional
        Etiqueta del eje Y.
    xlabel : str, opcional
        Etiqueta del eje X.
    title : str, opcional
        Título de la gráfica.
    colorbar_label : str, opcional
        Etiqueta de la barra de colores.
    invert_order : bool, opcional
        Si True, invierte el orden de los archivos y temperaturas.
    """

    # Extraer temperaturas válidas
    temps = [extraer_temperatura(p) for p in archivos_dat_samp if extraer_temperatura(p) is not None]
    if not temps:
        raise ValueError("No se encontraron temperaturas válidas en los archivos")

    if invert_order:
        temps = temps[::-1]
        archivos_dat_samp_plot = archivos_dat_samp[::-1]
    else:
        archivos_dat_samp_plot = archivos_dat_samp

    # Configuración de colores según temperatura
    min_temp, max_temp = min(temps), max(temps)
    norm = Normalize(vmin=min_temp, vmax=max_temp)

    # Crear figura
    fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax1.set_ylabel(field_label)
    ax1.set_xlabel(xlabel)
    ax1.set_title(title)

    val = 0  # desplazamiento acumulado
    for index, path_signal in enumerate(archivos_dat_samp_plot):
        temp = temps[index]
        color = cmap(norm(temp)) if temp is not None else 'blue'

        path_signal = path_signal
        path_ref = archivos_dat_ref[0]

        # Obtener datos filtrados
        phase, y_signal_ventaneada, y_substrate_padding, ventana = getSignalWindowed(
            path_signal, path_ref, left, right_sample, right_ref, params_window1
        )


        y_signal_vent = pd.Series(y_signal_ventaneada * ventana)

        # --- subplot 1: dominio temporal ---
        
        ax1.plot(y_signal_vent,color=color)
        # ax1.plot(ventana / (np.max(np.abs(ventana)) or 1.0) + offset_val, 'k--')
        # ax1.legend(loc='lower right')

        # Graficar señal desplazada
        # ax1.plot(x, y + val, color=color)
        # val += offset_factor * max(y)  # incremento desplazamiento
    # ax1.plot(ventana / (np.max(np.abs(ventana)) or 1.0),'k')
    # Barra de color lateral
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cbar_ax, label=colorbar_label)

    plt.show()


def preparar_y_procesar(
    sample: int = 6,
    base_dir: str | None = None,
    project_rel: str = os.path.join("EuZn2P2", "src"),
    out_subdir: str = "carpeta1",
    pattern: str = "*.dat",
    rang: int = 4,                 # parámetro que pasas a convert_dats (p.ej. número de bins)
    limpiar_salidas_previas: bool = True,
):
    """
    Limpia los .dat previos en las carpetas de salida de sample y reference,
    ejecuta convert_dats en ambas carpetas base, y regresa las listas de .dat
    generados y ordenados por temperatura.

    Returns:
        (archivos_dat_ref, archivos_dat_samp) como listas de rutas (strings)
    """
    ruta_actual = base_dir or os.getcwd()

    carpeta_sample = os.path.join(ruta_actual, project_rel, f"sample{sample}_ang")
    carpeta_ref    = os.path.join(ruta_actual, project_rel, f"reference{sample}")

    # Asegurar que existan las carpetas de salida para evitar glob vacíos por inexistencia
    os.makedirs(os.path.join(carpeta_sample, out_subdir), exist_ok=True)
    os.makedirs(os.path.join(carpeta_ref, out_subdir), exist_ok=True)

    # 1) Limpiar archivos previos en las carpetas de salida
    if limpiar_salidas_previas:
        archivos_dat_samp = glob(os.path.join(carpeta_sample, out_subdir, pattern))
        archivos_dat_ref  = glob(os.path.join(carpeta_ref,    out_subdir, pattern))

        for archivo in archivos_dat_samp:
            try:
                os.remove(archivo)
                print(f'Archivo {archivo} eliminado.')
            except Exception as e:
                print(f'No se pudo eliminar {archivo}: {e}')

        for archivo in archivos_dat_ref:
            try:
                os.remove(archivo)
                print(f'Archivo {archivo} eliminado.')
            except Exception as e:
                print(f'No se pudo eliminar {archivo}: {e}')

    # 2) Procesar (esto crea nuevos .dat en out_subdir)
    convert_dats(carpeta_ref, rang)
    convert_dats(carpeta_sample, rang)

    # 3) Re-listar y ordenar por temperatura (usar rutas NUEVAS)
    archivos_dat_ref = glob(os.path.join(carpeta_ref, out_subdir, pattern))
    archivos_dat_samp = glob(os.path.join(carpeta_sample, out_subdir, pattern))

    archivos_dat_ref = sorted(archivos_dat_ref, key=lambda x: extraer_temperatura(x))
    archivos_dat_samp = sorted(archivos_dat_samp, key=lambda x: extraer_temperatura(x))

    return archivos_dat_ref, archivos_dat_samp

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




def getFilterdata(path_signal, right=None, left=None):
    """
    Lee un archivo .dat y devuelve las columnas 'pos' y 'X'.
    Si 'left' y 'right' se proporcionan, recorta la señal en ese rango.
    Si son None, devuelve la señal completa.

    Parameters
    ----------
    path_signal : str
        Ruta del archivo a leer.
    right : float or None, optional
        Límite superior para el recorte (inclusive).
    left : float or None, optional
        Límite inferior para el recorte (inclusive).

    Returns
    -------
    pos : pandas.Series
        Columna 'pos'.
    X : pandas.Series
        Columna 'X'.
    """
    df1 = pd.read_csv(path_signal, delim_whitespace=True)

    # Si ambos límites están definidos, filtrar
    if left is not None and right is not None:
        df1 = df1.loc[(df1.iloc[:, 0] >= left) & (df1.iloc[:, 0] <= right)]

    return df1['pos'], df1['X']


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

    # Asegurar mismos tamaños (por si hay diferencias residuales)
    len_diff = len(y) - len(y_substrate)
    if len_diff > 0:
        y_substrate = np.pad(y_substrate, (len_diff, 0), 'constant')
    elif len_diff < 0:
        y = np.pad(y, (-len_diff, 0), 'constant')

    # # Encontrar máximos
    idx_max_y = np.argmax(y)
    idx_max_subs = np.argmax(y_substrate)

    # # Desplazar y para alinear su máximo al del substrato
    desplazamiento = idx_max_subs - idx_max_y
    
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