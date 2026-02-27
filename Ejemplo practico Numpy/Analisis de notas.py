import numpy as np
import matplotlib.pyplot as plt

# Generar notas aleatorias

np.random.seed()  # Para reproducibilidad

# 40 estudiantes, 5 trabajos
notas = np.random.uniform(0, 5.0, (40, 5))

# Redondear a 2 decimales
notas = np.round(notas, 2)

print("Notas generadas:\n")
print(notas)

# Estadísticas generales

media_general = np.mean(notas)
mediana_general = np.median(notas)
desviacion_general = np.std(notas)
maximos = np.max(notas, axis=0)
minimos = np.min(notas, axis=0)
rangos = maximos - minimos

print("\nRango por trabajo:")
for i in range(5):
    print(f"Trabajo {i+1}: {round(rangos[i],2)}")

print("\nEstadisticas Generales:")
print("Media general:", round(media_general, 2))
print("Mediana general:", round(mediana_general, 2))
print("Desviacion estandar:", round(desviacion_general, 2))


# Moda

valores, conteos = np.unique(notas, return_counts=True)
indice_moda = np.argmax(conteos)
moda = valores[indice_moda]

print("Moda:", moda)

# Promedio por estudiante

promedios_estudiantes = np.mean(notas, axis=1)

# Promedio por trabajo

promedios_trabajos = np.mean(notas, axis=0)

print("\nPromedio por trabajo:")
for i, prom in enumerate(promedios_trabajos):
    print(f"Trabajo {i+1}: {round(prom,2)}")


# Aprobados y Reprobados

aprobados = promedios_estudiantes >= 2.5
num_aprobados = np.sum(aprobados)
num_reprobados = 40 - num_aprobados

porc_aprobados = (num_aprobados / 40) * 100
porc_reprobados = (num_reprobados / 40) * 100

print("\nAprobados:", num_aprobados, f"({round(porc_aprobados,2)}%)")
print("Reprobados:", num_reprobados, f"({round(porc_reprobados,2)}%)")

# Top 5 mejores estudiantes

indices_top5 = np.argsort(promedios_estudiantes)[-5:][::-1]

print("\nTop 5 Estudiantes:")
for i in indices_top5:
    print(f"Estudiante {i+1} - Promedio: {round(promedios_estudiantes[i],2)}")

# Top 5 peores estudiantes

indices_peores5 = np.argsort(promedios_estudiantes)[:5]

print("\nTop 5 peores estudiantes:")
for i in indices_peores5:
    print(f"Estudiante {i+1} - Promedio: {round(promedios_estudiantes[i],2)}")


# Asimetria

media = np.mean(promedios_estudiantes)
desviacion = np.std(promedios_estudiantes)

asimetria = np.mean(((promedios_estudiantes - media) / desviacion) ** 3)

print("\nAsimetria de los promedios:", round(asimetria,4))

# Outliers con metodo IQR

Q1 = np.percentile(promedios_estudiantes, 25)
Q3 = np.percentile(promedios_estudiantes, 75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = promedios_estudiantes[
    (promedios_estudiantes < limite_inferior) |
    (promedios_estudiantes > limite_superior)
]

print("\nOutliers encontrados:", outliers)



# Matriz de correlacion

matriz_correlacion = np.corrcoef(notas, rowvar=False)

print("\nMatriz de correlacion:\n")
print(np.round(matriz_correlacion,2))

# Matriz de covarianza

matriz_covarianza = np.cov(notas, rowvar=False)

print("\nMatriz de covarianza:\n")
print(np.round(matriz_covarianza,2))

# Graficas

# Grafica matriz de correlacion

plt.figure()
plt.imshow(matriz_correlacion)
plt.title("Matriz de Correlacion")
plt.colorbar()
plt.xticks(range(5), [f"T{i}" for i in range(1,6)])
plt.yticks(range(5), [f"T{i}" for i in range(1,6)])
plt.show()

# Grafica matriz de covarianza

plt.figure()
plt.imshow(matriz_covarianza)
plt.title("Matriz de Covarianza")
plt.colorbar()
plt.xticks(range(5), [f"T{i}" for i in range(1,6)])
plt.yticks(range(5), [f"T{i}" for i in range(1,6)])
plt.show()

# Grafica de promedios con outliers

indices = np.arange(1, len(promedios_estudiantes) + 1)

# Condicion para detectar outliers
es_outlier = (promedios_estudiantes < limite_inferior) | \
             (promedios_estudiantes > limite_superior)

plt.figure()

# Todos los estudiantes
plt.scatter(indices[~es_outlier],
            promedios_estudiantes[~es_outlier],
            label="Estudiantes normales")

# Outliers en rojo
plt.scatter(indices[es_outlier],
            promedios_estudiantes[es_outlier],
            label="Outliers")

# Lineas de limite
plt.axhline(limite_superior, linestyle="--", label="Limite superior")
plt.axhline(limite_inferior, linestyle="--", label="Limite inferior")

plt.title("Deteccion de Outliers (Metodo IQR)")
plt.xlabel("Estudiante")
plt.ylabel("Promedio")
plt.legend()
plt.show()
                                                            
# Histograma de promedios
plt.figure()
plt.hist(promedios_estudiantes, bins=10)
plt.title("Distribución de Promedios")
plt.xlabel("Promedio")
plt.ylabel("Cantidad de Estudiantes")
plt.show()


# Promedio por trabajo
plt.figure()
plt.bar(np.arange(1,6), promedios_trabajos)
plt.title("Promedio por Trabajo")
plt.xlabel("Trabajo")
plt.ylabel("Promedio")
plt.show()


# Aprobados vs Reprobados
plt.figure()
plt.pie([num_aprobados, num_reprobados],
        labels=["Aprobados", "Reprobados"],
        autopct='%1.1f%%')
plt.title("Porcentaje de Aprobación")
plt.show()