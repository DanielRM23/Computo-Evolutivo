import random
import math
import numpy as np
import matplotlib.pyplot as plt

# ===================================================================
# 1. Fitness, esta es común a todas las codificaciones
# ===================================================================

def fitness(tablero):
    """
    Calcula el fitness penalizando filas repetidas (columnas en nuestra representación)
    y ataques diagonales.
    """
    n = len(tablero)
    ataques_diagonales = 0
    
    # --- PASO 1: Penalizar las filas repetidas ---
    # Usamos un set para contar cuántos números únicos hay.
    filas_unicas = set(tablero)
    colisiones_filas = n - len(filas_unicas) # Cada colisión es un número repetido
    
    # Una penalización grande por cada fila repetida
    penalizacion = colisiones_filas * n 

    # --- PASO 2: Calcular ataques diagonales ---
    diagonal_principal = {}
    diagonal_secundaria = {}
    
    for columna, fila in enumerate(tablero):
        dp = fila - columna
        ds = fila + columna
        if dp in diagonal_principal:
            ataques_diagonales += diagonal_principal[dp]
            diagonal_principal[dp] += 1
        else:
            diagonal_principal[dp] = 1
        if ds in diagonal_secundaria:
            ataques_diagonales += diagonal_secundaria[ds]
            diagonal_secundaria[ds] += 1
        else:
            diagonal_secundaria[ds] = 1
            
    # El fitness total es la suma de la penalización y los ataques
    return penalizacion + ataques_diagonales

# ===================================================================
# 2. FUNCIONES PARA LA CODIFICACIÓN ENTERA
# ===================================================================

def generar_poblacion_entera(tamano_poblacion, n):
    """Genera una población inicial usando permutaciones de 0 a n-1."""
    return [random.sample(range(n), n) for _ in range(tamano_poblacion)]

def cruza_entera(padre1, padre2):
    n = len(padre1)
    hijo1, hijo2 = [-1] * n, [-1] * n
    punto_corte1, punto_corte2 = sorted(random.sample(range(n), 2))
    
    hijo1[punto_corte1:punto_corte2] = padre1[punto_corte1:punto_corte2]
    hijo2[punto_corte1:punto_corte2] = padre2[punto_corte1:punto_corte2]
    
    def completar_hijo(hijo, otro_padre):
        elementos_faltantes = [item for item in otro_padre if item not in hijo]
        idx = 0
        for i in range(n):
            if hijo[i] == -1:
                hijo[i] = elementos_faltantes[idx]
                idx += 1
    
    completar_hijo(hijo1, padre2)
    completar_hijo(hijo2, padre1)
    return hijo1, hijo2

def mutacion_entera(tablero):
    n = len(tablero)
    pos1, pos2 = random.sample(range(n), 2)
    tablero[pos1], tablero[pos2] = tablero[pos2], tablero[pos1]
    return tablero

def ordenar_poblacion_entera(poblacion):
    """Ordena la población entera. No necesita decodificación."""
    poblacion_fit = {i: (individuo, fitness(individuo)) for i, individuo in enumerate(poblacion)}
    return dict(sorted(poblacion_fit.items(), key=lambda item: item[1][1]))



# ===================================================================
# 3. FUNCIONES PARA LA CODIFICACIÓN BINARIA
# ===================================================================

def decodificar_binario(individuo, n):
    """Traduce un cromosoma binario a un tablero de enteros."""
    bits_por_reina = math.ceil(math.log2(n))
    tablero = []
    for i in range(n):
        segmento = individuo[i*bits_por_reina : (i+1)*bits_por_reina]
        valor_entero = int("".join(map(str, segmento)), 2)
        tablero.append(valor_entero % n) # Usamos módulo para asegurar que esté en rango
    return tablero

def generar_poblacion_binaria(tamano_poblacion, n):
    bits_por_reina = math.ceil(math.log2(n))
    longitud_cromosoma = n * bits_por_reina
    return [[random.randint(0, 1) for _ in range(longitud_cromosoma)] for _ in range(tamano_poblacion)]

def cruza_binaria(padre1, padre2):
    """Cruce de un punto para binario."""
    n = len(padre1)
    punto_corte = random.randint(1, n - 1)
    hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
    hijo2 = padre2[:punto_corte] + padre1[punto_corte:]
    return hijo1, hijo2

def mutacion_binaria(individuo, prob_mut=0.01):
    """Mutación de bit-flip."""
    for i in range(len(individuo)):
        if random.random() < prob_mut:
            individuo[i] = 1 - individuo[i]
    return individuo

def ordenar_poblacion_binaria(poblacion, n):
    """Ordena la población binaria, decodificando antes de evaluar."""
    poblacion_fit = {}
    for i, individuo in enumerate(poblacion):
        tablero_deco = decodificar_binario(individuo, n)
        valor_fit = fitness(tablero_deco)
        poblacion_fit[i] = (individuo, valor_fit) # Guardamos el individuo BINARIO
    return dict(sorted(poblacion_fit.items(), key=lambda item: item[1][1]))


# ===================================================================
# 4. FUNCIONES PARA LA CODIFICACIÓN REAL
# ===================================================================

def discretizar_real(individuo):
    """Traduce un cromosoma real a un tablero de enteros.
        Esto es como hacer la función floor, pero asegurando que
        el valor máximo n-1 se pueda alcanzar.
    """
    return [int(gen) for gen in individuo]

def generar_poblacion_real(tamano_poblacion, n):
    """Genera individuos con genes reales en [0, n)."""
    return [[random.uniform(0, n) for _ in range(n)] for _ in range(tamano_poblacion)]

def cruza_real(padre, madre, n):
    """
    Esta es la función que uso en las funciones de la esfera y eggholder.
    'n' es el tamaño del tablero.
    """
    # Para N-Reinas, los rangos son los mismos para todos los genes: [0, n)
    low = 0
    high = n

    hijo1 = []
    hijo2 = []
    for (gen_padre, gen_madre) in zip(padre, madre):
        alpha = np.random.uniform(-0.25, 1.25)
        
        gen_hijo_1 = gen_padre + alpha*(gen_madre - gen_padre)
        
        # Se puede usar el mismo alpha o uno diferente, yo uso diferente
        beta = np.random.uniform(-0.25, 1.25)
        gen_hijo_2 = gen_madre + beta*(gen_padre - gen_madre)

        hijo1.append(gen_hijo_1)
        hijo2.append(gen_hijo_2)

    # np.clip se asegura de que ningún gen se salga del rango [0, n)
    hijo1 = np.clip(hijo1, low, high - 1e-9)
    hijo2 = np.clip(hijo2, low, high - 1e-9)

    # Devolvemos como listas para mantener la consistencia con las otras funciones
    return hijo1.tolist(), hijo2.tolist()


def mutacion_real(hijo, n, m=20): 
    """
    Esta es la función que uso en las funciones de la esfera y eggholder.
    'n' es el tamaño del tablero.
    'm' es el parámetro de precisión de la mutación.
    """
    numero_genes = len(hijo)
    p = 1.0 / numero_genes # Probabilidad de mutar cada gen
    
    low = 0
    high = n
    rango_mutacion = 0.5 * (high - low)
    
    hijo_mutado = []
    for gen in hijo:
        if np.random.rand() < p:
            delta = 0.0
            for j in range(m):
                if np.random.rand() < (1.0 / m): 
                    delta += 2**(-j)
            
            s = np.random.choice([-1, 1])
            nuevo_gen = gen + s * rango_mutacion * delta

            # Aseguramos que el nuevo gen no se salga del rango
            nuevo_gen = np.clip(nuevo_gen, low, high - 1e-9)
            hijo_mutado.append(nuevo_gen)
        else:
            hijo_mutado.append(gen)
    
    return hijo_mutado


def ordenar_poblacion_real(poblacion, n):
    """Ordena la población real, discretizando antes de evaluar."""
    poblacion_fit = {}
    for i, individuo in enumerate(poblacion):
        tablero_disc = discretizar_real(individuo)
        valor_fit = fitness(tablero_disc)
        poblacion_fit[i] = (individuo, valor_fit) # Guardamos el individuo REAL
    return dict(sorted(poblacion_fit.items(), key=lambda item: item[1][1]))




# ===================================================================
# DICCIONARIO DE CODIFICACIONES Y OPERADORES
# ===================================================================
operadores_por_codificacion = {
    'entera': {
        'generar': generar_poblacion_entera,
        'cruza': lambda p1, p2, n: cruza_entera(p1, p2),
        'mutacion': lambda ind, n: mutacion_entera(ind),
        'ordenar': lambda pop, n: ordenar_poblacion_entera(pop)
    },
    'binaria': {
        'generar': generar_poblacion_binaria,
        'cruza': lambda p1, p2, n: cruza_binaria(p1, p2),
        'mutacion': lambda ind, n: mutacion_binaria(ind),
        'ordenar': ordenar_poblacion_binaria
    },
    'real': {
        'generar': generar_poblacion_real,
        'cruza': cruza_real,
        'mutacion': mutacion_real,
        'ordenar': ordenar_poblacion_real
    }
}

# ===================================================================
# Estas funciones funcionan para CUALQUIER codificación.
# ===================================================================

def seleccionar_padres(poblacion_ordenada, tamano_torneo=3):
    """Selección por torneo genérica."""
    # Extrae solo la lista de (individuo, fit)
    participantes = list(poblacion_ordenada.values())
    
    # Elige 'tamano_torneo' competidores al azar
    torneo = random.sample(participantes, tamano_torneo)
    
    # El ganador del torneo es el que tiene el menor fit (menor número de ataques)
    ganador1 = min(torneo, key=lambda x: x[1])
    
    # Segundo torneo para el segundo padre
    torneo = random.sample(participantes, tamano_torneo)
    ganador2 = min(torneo, key=lambda x: x[1])
    
    return ganador1[0], ganador2[0] # Devuelve solo los individuos


def evolucionar(poblacion_ordenada, n, tamano_poblacion, max_generaciones, operadores):
    """
    Ciclo evolutivo principal que ahora registra el historial de fitness.
    
    Devuelve:
    - La mejor solución encontrada.
    - El fitness de esa solución.
    - Una lista con el mejor fitness de cada generación (historial).
    """
    mejor_solucion_global = None
    mejor_fit_global = float('inf')
    
    # Lista para guardar el historial de fitness de esta corrida
    historial_fitness = []

    for gen in range(max_generaciones):
        # El mejor de la generación es el primero de la población ya ordenada
        mejor_individuo_actual, mejor_fit_actual = next(iter(poblacion_ordenada.values()))

        # Actualizamos el mejor global si encontramos uno nuevo
        if mejor_fit_actual < mejor_fit_global:
            mejor_solucion_global = mejor_individuo_actual
            mejor_fit_global = mejor_fit_actual
        
        # Guardamos el mejor fitness global encontrado HASTA esta generación
        historial_fitness.append(mejor_fit_global)

        # Condición de paro: si encontramos la solución perfecta (0 ataques)
        if mejor_fit_global == 0:
            # Rellenamos el resto del historial con 0 para que todas las listas tengan el mismo tamaño
            generaciones_restantes = max_generaciones - len(historial_fitness)
            historial_fitness.extend([0] * generaciones_restantes)
            
            # Devolvemos los resultados anticipadamente
            return mejor_solucion_global, mejor_fit_global, historial_fitness

        # --- Crear la siguiente generación ---
        nueva_poblacion = []
        while len(nueva_poblacion) < tamano_poblacion:
            padre1, padre2 = seleccionar_padres(poblacion_ordenada)
            
            # Usamos los operadores de la estrategia actual
            hijo1, hijo2 = operadores['cruza'](padre1, padre2, n=n)
            hijo1 = operadores['mutacion'](hijo1, n=n)
            hijo2 = operadores['mutacion'](hijo2, n=n)

            nueva_poblacion.extend([hijo1, hijo2])
        
        # Recortamos por si se generaron individuos de más
        nueva_poblacion = nueva_poblacion[:tamano_poblacion]
        
        # Evaluamos y ordenamos la nueva población para la siguiente iteración
        poblacion_ordenada = operadores['ordenar'](nueva_poblacion, n)

    # Si el bucle termina sin encontrar fit = 0, se devuelve el mejor encontrado
    return mejor_solucion_global, mejor_fit_global, historial_fitness

# ===================================================================
# FUNCIÓN PRINCIPAL Y EJECUCIÓN
# ===================================================================

def N_reinas(n, tamano_poblacion, max_generaciones, codificacion):
    #print(f"--- Resolviendo N-Reinas para N={n} con codificación: {codificacion.upper()} ---")

    # 1. Selecciona el "paquete de codificación" de funciones correcto
    operadores = operadores_por_codificacion[codificacion]

    # 2. Genera la población inicial usando la función correcta
    poblacion_inicial = operadores['generar'](tamano_poblacion, n)
    
    # 3. Evalúa y ordena la población inicial
    poblacion_ordenada = operadores['ordenar'](poblacion_inicial, n)
    
    # 4. Inicia el proceso evolutivo
    solucion, fit, historial = evolucionar(poblacion_ordenada, n, tamano_poblacion, max_generaciones, operadores)
    # 5. Muestra los resultados
    #print("\n--- Resultados ---")
    #print(f"Mejor fit encontrado: {fit}")
    
    # "Traducimos" la solución final para que siempre se muestre como un tablero de enteros
    if codificacion == 'binaria':
        solucion_final = decodificar_binario(solucion, n)
    elif codificacion == 'real':
        solucion_final = discretizar_real(solucion)
    else: # Entera
        solucion_final = solucion
        
    #print(f"Mejor tablero encontrado: {solucion_final}")
    return solucion, fit, historial




if __name__ == "__main__":
    # ===================================================================
    # 1. CONFIGURACIÓN DE EXPERIMENTOS
    # ===================================================================
    NUM_EJECUCIONES = 30 # Número de veces que se repetirá cada experimento

    INSTANCIAS = {
        "N8_entera": dict(tamanio_tablero=8, tamano_poblacion=100, max_generaciones=100, codificacion='entera'),
        "N8_binaria": dict(tamanio_tablero=8, tamano_poblacion=100, max_generaciones=100, codificacion='binaria'),
        "N8_real": dict(tamanio_tablero=8, tamano_poblacion=100, max_generaciones=100, codificacion='real'),
        "N12_entera": dict(tamanio_tablero=12, tamano_poblacion=200, max_generaciones=500, codificacion='entera'),
        "N12_binaria": dict(tamanio_tablero=12, tamano_poblacion=200, max_generaciones=500, codificacion='binaria'),
        "N12_real": dict(tamanio_tablero=12, tamano_poblacion=200, max_generaciones=500, codificacion='real'),
    }

    # Diccionario para almacenar los historiales de fitness de todas las ejecuciones
    resultados_finales = {nombre: [] for nombre in INSTANCIAS.keys()}

    # ===================================================================
    # 2. BUCLE DE EJECUCIÓN DE EXPERIMENTOS
    # ===================================================================
    for nombre, instancia in INSTANCIAS.items():
        #print(f"==============================================================")
        #print(f"-> EJECUTANDO EXPERIMENTO: {nombre} ({NUM_EJECUCIONES} veces)")
        
        for i in range(NUM_EJECUCIONES):
            #print(f"   Corrida {i+1}/{NUM_EJECUCIONES}...", end='\r')
            # Usar una semilla diferente para cada corrida asegura resultados distintos
            random.seed(i)
            np.random.seed(i)

            # Ejecutamos la simulación y guardamos solo el historial
            _, _, historial = N_reinas(
                n=instancia["tamanio_tablero"],
                tamano_poblacion=instancia["tamano_poblacion"],
                max_generaciones=instancia["max_generaciones"],
                codificacion=instancia["codificacion"]
            )
            resultados_finales[nombre].append(historial)
        print(f"\n-> Experimento {nombre} completado.")

    print("\nTODOS LOS EXPERIMENTOS COMPLETADOS. GENERANDO GRÁFICAS...")

    # ===================================================================
    # 3. BUCLE DE GRAFICACIÓN
    # ===================================================================
    # Agrupamos los resultados por tamaño del problema (8, 12, etc.)
    problemas_agrupados = {}
    for nombre, historiales in resultados_finales.items():
        n_problema = INSTANCIAS[nombre]['tamanio_tablero']
        if n_problema not in problemas_agrupados:
            problemas_agrupados[n_problema] = {}
        problemas_agrupados[n_problema][nombre] = historiales

    # Generamos una gráfica por cada tamaño de problema
    for n_problema, resultados_problema in problemas_agrupados.items():
        plt.figure(figsize=(15, 7))
        colores = {'entera': 'blue', 'binaria': 'red', 'real': 'green'}
        
        for nombre, historiales in resultados_problema.items():
            codificacion = INSTANCIAS[nombre]['codificacion']
            
            # Calcular promedios y desviación estándar
            mean = np.mean(historiales, axis=0)
            std = np.std(historiales, axis=0)
            
            # Graficar la línea de promedio
            plt.plot(mean, label=f'Codificación {codificacion.capitalize()}', color=colores[codificacion])
            
            # Graficar la sombra de desviación estándar
            plt.fill_between(range(len(mean)), mean - std, mean + std, color=colores[codificacion], alpha=0.1)

        # Títulos y etiquetas a la gráfica
        plt.title(f'Desempeño de Codificaciones para N-Reinas (N={n_problema})\n(Promedio de {NUM_EJECUCIONES} ejecuciones)')
        plt.xlabel('Generación')
        plt.ylabel('Mejor Fitness (Nº de Ataques)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.savefig(f'N_Reinas_N{n_problema}.png')
        plt.show()
