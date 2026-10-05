# Neurona artificial para el riego de plantas

Hecho por: Jairo Andres Niño Santos

Actividad de Inteligencia Artificial: una neurona artificial (2 entradas, 2 pesos y 1 sesgo) con
función de activación sigmoide que decide si una planta necesita riego.

- **Entrada x1:** humedad del suelo (0 a 100 %)
- **Entrada x2:** temperatura ambiental (0 a 50 °C aprox.)
- **Salida y:** `1` = regar, `0` = no regar

> Los datos son didácticos y no representan una recomendación agronómica para una especie real.

## Cómo ejecutar

Requisito: tener [uv](https://docs.astral.sh/uv/) instalado.

```bash
git clone <URL-DEL-REPOSITORIO>
cd neurona-riego-plantas
uv sync
uv run main.py
```

La salida completa de una ejecución está guardada en [`salida.txt`](salida.txt).

## Cómo funciona

1. **Normalización:** `X_normalizado = X / [100, 50]`. La misma variable `escala` se usa para los datos
   nuevos.
2. **Suma ponderada:** `z = X_normalizado @ pesos + sesgo`.
3. **Activación:** `probabilidad = 1 / (1 + e^(-z))`.
4. **Error:** error cuadrático medio (MSE) entre la probabilidad y la salida esperada.
5. **Gradientes** (regla de la cadena): `gradiente_z = 2 * error * p * (1 - p) / n`,
   `gradiente_pesos = X_normalizado.T @ gradiente_z` y `gradiente_sesgo = sum(gradiente_z)`.
6. **Actualización:** `pesos -= tasa_aprendizaje * gradiente_pesos` (igual para el sesgo).
7. **Decisión:** `respuesta = (probabilidad >= 0.5).astype(int)`.

Los pesos iniciales son aleatorios con semilla fija (7, como en el ejemplo de clase), así que los resultados son reproducibles.

## Resultado del entrenamiento base (10000 épocas, tasa 0.5)

| Parámetro | Valor |
|---|---|
| Peso de la humedad | -19.5032 |
| Peso de la temperatura | 2.3101 |
| Sesgo | 7.4022 |
| Error final (MSE) | 0.019645 |

Probabilidad calculada para cada caso de entrenamiento:

| Caso | Humedad | Temperatura | Esperado | Probabilidad | Respuesta |
|---|---|---|---|---|---|
| 1 | 80 % | 18 °C | 0 | 0.0006 | 0 |
| 2 | 70 % | 22 °C | 0 | 0.0053 | 0 |
| 3 | 65 % | 28 °C | 0 | 0.0183 | 0 |
| 4 | 55 % | 25 °C | 0 | 0.1025 | 0 |
| 5 | 50 % | 32 °C | 0 | 0.2951 | 0 |
| 6 | 40 % | 30 °C | 1 | 0.7285 | 1 |
| 7 | 35 % | 25 °C | 1 | 0.8496 | 1 |
| 8 | 30 % | 32 °C | 1 | 0.9539 | 1 |
| 9 | 20 % | 35 °C | 1 | 0.9941 | 1 |
| 10 | 10 % | 38 °C | 1 | 0.9993 | 1 |

**Significado del signo de cada peso**

- **Peso de la humedad (negativo):** a mayor humedad, `z` baja y la probabilidad de regar disminuye.
  Un suelo más húmedo reduce la necesidad de riego.
- **Peso de la temperatura (positivo):** a mayor temperatura, `z` sube y la probabilidad de regar
  aumenta. Más calor favorece que se necesite riego.

## Predicciones con condiciones nuevas (modelo base)

Cada dato nuevo se divide por la misma `escala = [100, 50]` antes de entrar a la neurona.

| Humedad | Temperatura | Probabilidad | Decisión |
|---|---|---|---|
| 75 % | 30 °C | 0.0029 | 0 (No regar) |
| 45 % | 34 °C | 0.5490 | 1 (Regar) |
| 25 % | 22 °C | 0.9719 | 1 (Regar) |
| 50 % | 25 °C | 0.2325 | 0 (No regar) |
| 30 % | 40 °C | 0.9677 | 1 (Regar) |

## Experimentos con los parámetros

En cada prueba solo cambia el parámetro indicado (misma semilla, mismos datos). "Época < 0.05" es la
primera época en la que el error baja de 0.05.

| Experimento | Épocas | Tasa | Error final | Aciertos | P(45 %, 34 °C) | Época < 0.05 | Aprendizaje |
|---|---|---|---|---|---|---|---|
| Prueba base | 10000 | 0.5 | 0.019645 | 10/10 | 0.5490 | 1978 | Rápido |
| Pocas épocas | 100 | 0.5 | 0.165623 | 10/10 | 0.5217 | nunca | Insuficiente |
| Cantidad intermedia | 1000 | 0.5 | 0.067481 | 10/10 | 0.5793 | nunca | Insuficiente |
| Más épocas | 20000 | 0.5 | 0.011705 | 10/10 | 0.5263 | 1978 | Rápido |
| Tasa pequeña | 10000 | 0.01 | 0.130048 | 10/10 | 0.5388 | nunca | Insuficiente |
| Tasa moderada | 10000 | 0.1 | 0.049728 | 10/10 | 0.5853 | 9885 | Lento |
| Tasa alta | 10000 | 1.0 | 0.011705 | 10/10 | 0.5263 | 989 | Rápido |
| Tasa muy alta | 10000 | 2.0 | 0.006566 | 10/10 | 0.5082 | 495 | Rápido |

Criterio de las etiquetas: *rápido* = error < 0.05 en 2000 épocas o menos; *lento* = lo logra, pero
tarda más; *insuficiente* = nunca baja de 0.05 o no acierta los 10 casos; *inestable* = el error
sube con frecuencia (no ocurrió en ningún experimento).

## Prueba adicional con el umbral (sin volver a entrenar)

Se usan los pesos del modelo base. Solo cambia un caso:

| Humedad | Temperatura | Probabilidad | Umbral 0.4 | Umbral 0.5 | Umbral 0.6 |
|---|---|---|---|---|---|
| 45 % | 34 °C | 0.5490 | 1 | 1 | **0** |

Los otros 14 casos (los 10 de entrenamiento y los otros 4 nuevos) tienen probabilidades fuera del
rango 0.4 a 0.6, así que su respuesta es la misma con los tres umbrales. El caso 45 % / 34 °C está en
la zona de duda de la neurona: con umbral 0.6 deja de regar.

Modificar el umbral **no cambia los pesos aprendidos** porque el umbral se aplica después de calcular
la probabilidad. El entrenamiento ya terminó y los pesos y el sesgo solo se actualizan con los
gradientes durante las épocas. El umbral únicamente decide cómo se traduce la probabilidad a 0 o 1.

## Análisis

**1. ¿Por qué fue necesario normalizar?** La humedad llega hasta 100 y la temperatura hasta 50, y
ambas escalas son grandes. Con valores grandes `z` se vuelve enorme, la sigmoide se satura (derivada
casi cero) y el aprendizaje se frena. Además, la variable de mayor magnitud dominaría los gradientes.
Al dejar ambas entradas cerca de 0 y 1 aprenden en condiciones comparables.

**2. ¿Dónde se usó `X_normalizado` y para qué se conservó `X`?** Se usó en la suma ponderada
(`z = X_normalizado @ pesos + sesgo`) y en el gradiente de los pesos
(`X_normalizado.T @ gradiente_z`). `X` se conservó con los valores originales (% y °C) para mostrar
los resultados de forma interpretable.

**3. ¿Qué ocurrió con solo 100 épocas?** La neurona aprendió muy poco: el error final fue 0.1656 (contra
0.0196 del modelo base) y todas las probabilidades quedaron en un rango estrecho, entre 0.29 y 0.69. Los
casos cercanos a la frontera quedaron casi en 0.5 (el caso 5 dio 0.489 y el caso 6 dio 0.523), así que
acertó 10/10 por muy poco margen. Es una neurona que apenas empezó a separar las clases.

**4. ¿Más épocas siempre mejoraron mucho?** No. De 100 a 1000 épocas el error bajó de 0.1656 a 0.0675 y
de 1000 a 10000 bajó a 0.0196, pero de 10000 a 20000 solo bajó de 0.0196 a 0.0117. Incluso con 100 épocas
ya se acertaban los 10 casos, y la decisión para los datos nuevos casi no cambia entre experimentos.
Más épocas solo hacen las probabilidades más seguras, con rendimientos decrecientes.

**5. ¿Qué efecto tuvo una tasa demasiado pequeña?** Con 0.01 las actualizaciones fueron tan pequeñas
que tras 10000 épocas el error seguía en 0.1300, con probabilidades poco seguras (todas entre 0.20 y
0.79). Acertó 10/10, pero con poco margen: el aprendizaje fue lento e insuficiente. Con 0.1 sí llegó
a error menor de 0.05, pero apenas en la época 9885.

**6. ¿Qué efecto tuvo una tasa alta o muy alta?** En estos datos, acelerar el aprendizaje: con 1.0 el
error bajó de 0.05 en 989 épocas y con 2.0 en 495 (contra 1978 con 0.5), y terminaron con errores de
0.0117 y 0.0066. No hubo inestabilidad (el error nunca oscilaba), porque el gradiente del MSE con
sigmoide es pequeño y el problema es sencillo. Una tasa aún mayor podría causar saltos y oscilaciones,
pero eso no se probó en este trabajo.

**7. ¿Qué representa el signo del peso de la humedad?** Es negativo (-19.50): la humedad y la
necesidad de riego van en sentidos opuestos. Más humedad, menos probabilidad de regar. Su magnitud es
grande porque la humedad es la variable que más separa las dos clases en estos datos.

**8. ¿Qué representa el signo del peso de la temperatura?** Es positivo (2.31): más temperatura aumenta
la probabilidad de regar. Su efecto es menor que el de la humedad (menor magnitud), es decir, en estos
datos la humedad pesa más en la decisión.

**9. ¿Por qué convertir la probabilidad con un umbral?** La sigmoide devuelve un valor continuo entre
0 y 1, pero la acción es discreta: se riega o no se riega. El umbral (0.5 por defecto) convierte la
confianza de la neurona en una decisión. Se puede ajustar según el costo de equivocarse (por ejemplo,
bajar el umbral si es peor dejar secar la planta que regarla de más).

**10. ¿Qué limitaciones tiene para una planta real?** Solo usa 2 variables y 10 datos inventados, sin
validación con datos distintos a los de entrenamiento. No considera tipo de planta, tipo de suelo,
lluvia, hora del día, luz, drenaje ni estación del año. La frontera de decisión es lineal (una sola
neurona no puede representar relaciones más complejas). Y no extrapola con fiabilidad fuera de los
rangos con los que se entrenó (0 a 100 % y 0 a 50 °C).

## Estructura del repositorio

```
main.py          desarrollo de la neurona, experimentos, umbrales y predicciones
pyproject.toml   configuración del proyecto (generado por uv)
uv.lock          versiones exactas de las dependencias (generado por uv)
salida.txt       salida completa de `uv run main.py`
README.md        este documento
```
