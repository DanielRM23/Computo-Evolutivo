import numpy as np 
import matplotlib.pyplot as plt
import random




def longitud(variables_rango):
    """
    variables_rango: [[a1,b1], [a2,b2], ...]
    Regresa las longitudes |b - a| de cada variable.
    """
    return [abs(r[1] - r[0]) for r in variables_rango]
        

def valores_a_representar(precision, longitudes):
    diez_elevado_a = 10**precision
    return [longitud*diez_elevado_a for longitud in longitudes]


def bits_necesarios(valores):
    return [np.floor(np.log2(i)+1) for i in valores]


def suma_bits(bits_necesarios):
    return np.sum(bits_necesarios)


def generar_individuo(N):
    N = int(N)
    return [random.randint(0,1) for _ in range(0,N)]


def binario_a_decimal(bits):
    """
    Convierte un arreglo de 0s y 1s (representando un número binario)
    a su valor decimal.
    
    Ejemplo:
    [1, 0, 1, 1] -> 11
    """
    decimal = 0
    n = len(bits)
    
    for i in range(n):
        # multiplicamos el bit por la potencia de 2 correspondiente
        decimal += bits[n - 1 - i] * (2 ** i)
    
    return decimal


def decodificacion(a,b, individuo_binario, m):
    x = a + binario_a_decimal(individuo_binario)*( (b-a)/((2**m) - 1) )
    return x


def individuos_decodificados(M,a1,b1,a2,b2,bits_necesaios):
    n = int(bits_necesaios[0])
    m = int(bits_necesaios[-1])

    N = suma_bits(bits_necesaios)

    parejas_individuos = []

    for _ in range(M):
        individuo = generar_individuo(N)
        
        bits_izquierdo = individuo[:n] 
        bits_derecho = individuo[n:n+m]

        x = decodificacion(a1,b1,bits_izquierdo,n)
        y = decodificacion(a2,b2,bits_derecho,m)

        parejas_individuos.append([x,y])

    return parejas_individuos


# ======================================= Ejecución del Programa =======================================

variables_rango = [[-3.0, 12.1],
                   [ 4.1,  5.8]]

a1, b1 = variables_rango[0]
a2, b2 = variables_rango[1]

longitudes = longitud(variables_rango)
precision = 4

valores = valores_a_representar(precision, longitudes)     # pasos por variable (aprox)
bits = bits_necesarios(valores)                            # bits por variable

M = 10
resultado = individuos_decodificados(M, a1, b1, a2, b2, bits)


print("Rangos:", variables_rango)
print("Longitudes:", longitudes)
print("Pasos (aprox):", valores)
print("Bits por variable:", bits)
print("Individuos decodificados:")
for par in resultado:
    print(par)
    # Puedes verificar los rangos:
    assert a1 <= par[0] <= b1
    assert a2 <= par[1] <= b2