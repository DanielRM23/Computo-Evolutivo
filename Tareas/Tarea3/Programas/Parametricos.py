import numpy as np
import matplotlib.pyplot as plt

# 1. Se importan los algoritmos desde los otros archivos
from Parametrico_binario import AG_Simple as run_binary_ga
from Parametrico_real import ciclo_algoritmo as run_real_ga

# 2. DEFINE las funciones de prueba 
#    Las definimos aquí para que ambos algoritmos usen exactamente las mismas.
def esfera(*x):
    return np.sum(np.array(x)**2)

def eggholder(x, y):
    x, y = float(x), float(y)
    term1 = -(y + 47) * np.sin(np.sqrt(np.abs(x/2 + (y + 47))))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return term1 + term2

# 3. CONFIGURAR los experimentos
INSTANCIAS = {
    "ESFERA_binaria": dict(f=esfera, codificacion='binaria', variables_rango=[[-10,10],[-10,10]], precision=4, pob_size=50, num_gen=100, p_cruce=0.9),
    "ESFERA_real":    dict(f=esfera, codificacion='real', variables_rango=[[-10,10],[-10,10]], precision=4, pob_size=50, num_gen=100, p_cruce=0.9),
    "EGGHOLDER_binaria": dict(f=eggholder, codificacion='binaria', variables_rango=[[-512,512],[-512,512]], precision=4, pob_size=250, num_gen=1000, p_cruce=0.9),
    "EGGHOLDER_real":    dict(f=eggholder, codificacion='real',    variables_rango=[[-512,512],[-512,512]], precision=4, pob_size=250, num_gen=1000, p_cruce=0.9),
}

# 4. EJECUTA los experimentos
NUM_EJECUCIONES = 30 
resultados_finales = {nombre: [] for nombre in INSTANCIAS.keys()}

for nombre, instancia in INSTANCIAS.items():
    print(f"--- Ejecutando: {nombre} ({NUM_EJECUCIONES} veces) ---")
    
    for i in range(NUM_EJECUCIONES):
        print(f"  Corrida {i+1}/{NUM_EJECUCIONES}", end='\r')
        
        if instancia['codificacion'] == 'binaria':
            # El AG_Simple espera los parámetros de forma separada
            _, _, _, historial = run_binary_ga(
                variables_rango=instancia["variables_rango"],
                precision=instancia["precision"],
                M=instancia["pob_size"],
                G=instancia["num_gen"],
                p_cruce=instancia["p_cruce"],
                f=instancia["f"]
            )
        else: # 'real'
            # El ciclo_algoritmo espera el diccionario 'instancia' completo
            _, _, historial = run_real_ga(instancia=instancia, e=4, m=5)
        
        resultados_finales[nombre].append(historial)
    print("\nCompletado.")

# 5. GRAFICA los resultados
print("\n--- Generando Gráficas Comparativas ---")
problemas = set(nombre.split('_')[0] for nombre in INSTANCIAS.keys())

for problema in problemas:
    plt.figure(figsize=(15, 7))
    for nombre, historiales in resultados_finales.items():
        if nombre.startswith(problema):
            mean = np.mean(historiales, axis=0)
            std = np.std(historiales, axis=0)
            color = 'red' if 'binaria' in nombre else 'green'
            label = f"Codificación {nombre.split('_')[1]}"
            plt.plot(mean, label=label, color=color)
            plt.fill_between(range(len(mean)), mean - std, mean + std, color=color, alpha=0.1)

    plt.title(f'Desempeño para la Función {problema}\n(Promedio de {NUM_EJECUCIONES} ejecuciones)')
    plt.xlabel('Generación')
    plt.ylabel('Mejor Fitness')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'Parametricos_{problema}.png') 
    plt.show()
