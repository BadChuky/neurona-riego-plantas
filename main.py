"""Neurona artificial para decidir si una planta necesita riego.

Entradas : humedad del suelo (0-100 %) y temperatura ambiental (0-50 °C).
Salida   : 1 = regar, 0 = no regar.
Activación: sigmoide.  Error: error cuadrático medio (MSE).
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Datos de entrenamiento
# ---------------------------------------------------------------------------
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

# ---------------------------------------------------------------------------
# 2. Normalización (misma escala para entrenamiento y datos nuevos)
# ---------------------------------------------------------------------------
# La humedad se encuentra aproximadamente entre 0 y 100.
# La temperatura se encuentra aproximadamente entre 0 y 50.
#
# Dividimos cada columna por su valor máximo esperado
# para dejar los datos en una escala cercana a 0 y 1.
escala = np.array([100, 50])

X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)


# ---------------------------------------------------------------------------
# 3. Neurona
# ---------------------------------------------------------------------------
def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def entrenar(tasa_aprendizaje, epocas, mostrar=False, semilla=7):
    """Entrena la neurona y devuelve (pesos, sesgo, historial_de_errores, error_final)."""
    rng = np.random.default_rng(semilla)
    pesos = rng.normal(size=(2, 1))  # peso humedad y peso temperatura
    sesgo = 0.0
    historial = []

    for epoca in range(1, epocas + 1):
        # Suma ponderada (usa X_normalizado)
        z = X_normalizado @ pesos + sesgo
        probabilidades = sigmoide(z)

        # Error y error cuadrático medio
        error = probabilidades - y
        mse = float(np.mean(error ** 2))
        historial.append(mse)

        # Gradientes (regla de la cadena: MSE -> sigmoide -> z), igual que en clase
        gradiente_z = 2 * error * probabilidades * (1 - probabilidades) / len(X_normalizado)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = float(np.sum(gradiente_z))

        # Actualización
        pesos = pesos - tasa_aprendizaje * gradiente_pesos
        sesgo = sesgo - tasa_aprendizaje * gradiente_sesgo

        if mostrar and (epoca == 1 or epoca % (epocas // 10) == 0):
            print(f"  Época {epoca:>6} | error cuadrático medio = {mse:.6f}")

    # Error final con los pesos ya actualizados
    probabilidades = sigmoide(X_normalizado @ pesos + sesgo)
    mse_final = float(np.mean((probabilidades - y) ** 2))
    return pesos, sesgo, historial, mse_final


def predecir_probabilidad(humedad, temperatura, pesos, sesgo):
    """Normaliza con la MISMA escala del entrenamiento y calcula la probabilidad."""
    entrada = np.array([humedad, temperatura], dtype=float) / escala
    return float(sigmoide(entrada @ pesos + sesgo)[0])


def a_binario(probabilidad, umbral=0.5):
    return (np.asarray(probabilidad) >= umbral).astype(int)


def epoca_en_alcanzar(historial, objetivo=0.05):
    """Primera época en la que el error cuadrático medio baja del objetivo (o None)."""
    h = np.array(historial)
    idx = np.where(h < objetivo)[0]
    return int(idx[0]) + 1 if len(idx) else None


def clasificar_aprendizaje(historial, mse_final, aciertos):
    """Etiqueta cualitativa del comportamiento del entrenamiento.

    - inestable   : el error sube con frecuencia o deja de ser un número válido.
    - insuficiente: no alcanza error < 0.05 o no acierta los 10 casos.
    - rápido      : alcanza error < 0.05 en 2000 épocas o menos.
    - lento       : alcanza error < 0.05, pero tarda más de 2000 épocas.
    """
    h = np.array(historial)
    if not np.all(np.isfinite(h)):
        return "inestable"
    subidas = int(np.sum(np.diff(h) > 1e-9))
    if subidas > 0.05 * len(h):
        return "inestable"
    alcanzo = epoca_en_alcanzar(historial)
    if aciertos < 10 or alcanzo is None:
        return "insuficiente"
    return "rápido" if alcanzo <= 2000 else "lento"


# ---------------------------------------------------------------------------
# 4. Configuración inicial (prueba base)
# ---------------------------------------------------------------------------
tasa_aprendizaje = 0.5
epocas = 10000

print("\n" + "=" * 70)
print(f"ENTRENAMIENTO BASE: tasa_aprendizaje={tasa_aprendizaje}, epocas={epocas}")
print("=" * 70)
pesos, sesgo, historial, error_final = entrenar(tasa_aprendizaje, epocas, mostrar=True)

print("\nResultados del entrenamiento base:")
print(f"  Peso de la humedad     : {pesos[0, 0]:.4f}")
print(f"  Peso de la temperatura : {pesos[1, 0]:.4f}")
print(f"  Sesgo                  : {sesgo:.4f}")
print(f"  Error final (MSE)      : {error_final:.6f}")

probabilidades = sigmoide(X_normalizado @ pesos + sesgo)
respuestas = a_binario(probabilidades)

print("\nProbabilidad calculada para cada caso de entrenamiento:")
print("  Caso | Humedad | Temp. | Esperado | Probabilidad | Respuesta")
for i in range(len(X)):
    print(f"  {i + 1:>4} | {X[i, 0]:>6.0f}% | {X[i, 1]:>4.0f}C | "
          f"{int(y[i, 0]):>8} | {probabilidades[i, 0]:>12.4f} | {respuestas[i, 0]:>9}")

# ---------------------------------------------------------------------------
# 5. Pruebas con condiciones nuevas
# ---------------------------------------------------------------------------
nuevos = [(75, 30), (45, 34), (25, 22), (50, 25), (30, 40)]

print("\n" + "=" * 70)
print("PREDICCIONES CON CONDICIONES NUEVAS (modelo base)")
print("=" * 70)
print("  Humedad | Temperatura | Probabilidad | Decisión")
for humedad, temperatura in nuevos:
    p = predecir_probabilidad(humedad, temperatura, pesos, sesgo)
    decision = int(a_binario(p))
    texto = "Regar" if decision == 1 else "No regar"
    print(f"  {humedad:>6}% | {temperatura:>9}C | {p:>12.4f} | {decision} ({texto})")

# ---------------------------------------------------------------------------
# 6. Experimentos con los parámetros
# ---------------------------------------------------------------------------
experimentos = [
    ("Prueba base",          10000, 0.5),
    ("Pocas épocas",           100, 0.5),
    ("Cantidad intermedia",   1000, 0.5),
    ("Más épocas",           20000, 0.5),
    ("Tasa pequeña",         10000, 0.01),
    ("Tasa moderada",        10000, 0.1),
    ("Tasa alta",            10000, 1.0),
    ("Tasa muy alta",        10000, 2.0),
]

print("\n" + "=" * 70)
print("EXPERIMENTOS CON LOS PARÁMETROS")
print("=" * 70)
print(f"{'Experimento':<20}{'Épocas':>7}{'Tasa':>6}{'Error final':>13}"
      f"{'Aciertos':>10}{'P(45%,34C)':>12}{'Época<0.05':>12}  Aprendizaje")

for nombre, ep, tasa in experimentos:
    w, b, hist, mse = entrenar(tasa, ep)
    p_train = sigmoide(X_normalizado @ w + b)
    aciertos = int(np.sum(a_binario(p_train) == y))
    p_45_34 = predecir_probabilidad(45, 34, w, b)
    etiqueta = clasificar_aprendizaje(hist, mse, aciertos)
    alcanzo = epoca_en_alcanzar(hist)
    alcanzo_txt = str(alcanzo) if alcanzo else "nunca"
    print(f"{nombre:<20}{ep:>7}{tasa:>6}{mse:>13.6f}{aciertos:>7}/10"
          f"{p_45_34:>12.4f}{alcanzo_txt:>12}  {etiqueta}")

# ---------------------------------------------------------------------------
# 7. Prueba adicional con el umbral (sin volver a entrenar)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPARACIÓN DE UMBRALES (con los pesos del modelo base)")
print("=" * 70)

# Se evalúan los 10 casos de entrenamiento y los 5 casos nuevos
casos = [(X[i, 0], X[i, 1], f"Entrenamiento {i + 1}") for i in range(len(X))]
casos += [(h, t, "Nuevo") for h, t in nuevos]

print(f"{'Caso':<18}{'Hum.':>6}{'Temp.':>7}{'Prob.':>9}{'U=0.4':>7}{'U=0.5':>7}{'U=0.6':>7}")
for humedad, temperatura, etiqueta in casos:
    p = predecir_probabilidad(humedad, temperatura, pesos, sesgo)
    r04, r05, r06 = (int(a_binario(p, u)) for u in (0.4, 0.5, 0.6))
    marca = "  <-- cambia" if len({r04, r05, r06}) > 1 else ""
    print(f"{etiqueta:<18}{humedad:>5.0f}%{temperatura:>6.0f}C{p:>9.4f}"
          f"{r04:>7}{r05:>7}{r06:>7}{marca}")

print("\nLos pesos y el sesgo son los mismos en las tres columnas: el umbral solo")
print("cambia cómo se interpreta la probabilidad, no lo aprendido.")
