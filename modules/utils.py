import numpy as np
from scipy import stats

def calcular_prueba_z(media_muestra, media_h0, sigma, n, tipo_prueba="bilateral"):
    """
    Calcula el estadístico Z y el p-valor.
    """
    # Cálculo del estadístico Z
    z_stat = (media_muestra - media_h0) / (sigma / np.sqrt(n))
    
    # Cálculo del p-valor según el tipo de prueba
    if tipo_prueba == "bilateral":
        p_value = stats.norm.sf(abs(z_stat)) * 2
    elif tipo_prueba == "cola_derecha":
        p_value = stats.norm.sf(z_stat)
    else:  # cola_izquierda
        p_value = stats.norm.cdf(z_stat)
        
    return z_stat, p_value
    