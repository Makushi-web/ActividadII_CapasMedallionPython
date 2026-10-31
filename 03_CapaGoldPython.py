# 03_gold_analytics.py

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# =========================
# 0. Configuración
# =========================

CATALOG = "airline_mantenimiento"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

spark = SparkSession.builder.getOrCreate()

# =========================
# 1. Asegurar schema Gold
# =========================

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

# =========================
# 2. Leer tablas Silver
# =========================

vuelos_slv = spark.table(
    f"{CATALOG}.{SILVER_SCHEMA}.vuelos"
)

aeronaves_slv = spark.table(
    f"{CATALOG}.{SILVER_SCHEMA}.aeronaves"
)

aeropuertos_slv = spark.table(
    f"{CATALOG}.{SILVER_SCHEMA}.aeropuertos"
)

mantenimientos_slv = spark.table(
    f"{CATALOG}.{SILVER_SCHEMA}.mantenimientos"
)

# =====================================================
# 3. GOLD: Resumen diario de vuelos
# =====================================================
# Objetivo:
# Analizar la cantidad de vuelos, su duracion y sus
# estados agrupados por fecha.

resumen_vuelos_diario = (
    vuelos_slv
    .groupBy("fecha")
    .agg(
        F.countDistinct("vuelo_id").alias("total_vuelos"),

        F.countDistinct("aeronave_id").alias(
            "aeronaves_utilizadas"
        ),

        F.sum("duracion_min").alias(
            "duracion_total_min"
        ),

        F.avg("duracion_min").alias(
            "duracion_promedio_min"
        ),

        F.sum(
            F.when(
                F.col("estado").isNull(),
                1
            ).otherwise(0)
        ).alias("vuelos_sin_estado")
    )
)

# Agregar el conteo de vuelos por estado.
# Cada estado se conserva como una categoría separada.

vuelos_por_estado = (
    vuelos_slv
    .groupBy("fecha", "estado")
    .agg(
        F.countDistinct("vuelo_id").alias("cantidad")
    )
    .groupBy("fecha")
    .pivot("estado")
    .agg(F.first("cantidad"))
)

resumen_vuelos_diario = (
    resumen_vuelos_diario
    .join(vuelos_por_estado, on="fecha", how="left")
)

# =====================================================
# 4. GOLD: Analisis de mantenimiento por aeronave
# =====================================================
# Objetivo:
# Conocer el volumen de mantenimiento, costos y horas
# acumuladas por aeronave.

mantenimiento_por_aeronave = (
    mantenimientos_slv
    .groupBy("aeronave_id")
    .agg(
        F.countDistinct("mantenimiento_id").alias(
            "total_mantenimientos"
        ),

        F.sum("costo_usd").alias(
            "costo_total_usd"
        ),

        F.avg("costo_usd").alias(
            "costo_promedio_usd"
        ),

        F.sum("duracion_hr").alias(
            "horas_mantenimiento"
        ),

        F.avg("duracion_hr").alias(
            "duracion_promedio_hr"
        ),

        F.min("fecha").alias(
            "fecha_primer_mantenimiento"
        ),

        F.max("fecha").alias(
            "fecha_ultimo_mantenimiento"
        )
    )
)

analisis_mantenimiento_aeronave = (
    aeronaves_slv
    .join(
        mantenimiento_por_aeronave,
        on="aeronave_id",
        how="left"
    )
    .fillna({
        "total_mantenimientos": 0
    })
    .withColumn(
        "costo_total_usd",
        F.coalesce(
            F.col("costo_total_usd"),
            F.lit(0.0)
        )
    )
    .withColumn(
        "horas_mantenimiento",
        F.coalesce(
            F.col("horas_mantenimiento"),
            F.lit(0.0)
        )
    )
)

# =====================================================
# 5. GOLD: Analisis de rutas
# =====================================================
# Objetivo:
# Identificar las rutas con mayor numero de vuelos,
# duracion acumulada y duracion promedio.

origen = aeropuertos_slv.select(
    F.col("aeropuerto_id").alias("origen_id"),
    F.col("nombre").alias("nombre_origen"),
    F.col("ciudad").alias("ciudad_origen"),
    F.col("pais").alias("pais_origen")
)

destino = aeropuertos_slv.select(
    F.col("aeropuerto_id").alias("destino_id"),
    F.col("nombre").alias("nombre_destino"),
    F.col("ciudad").alias("ciudad_destino"),
    F.col("pais").alias("pais_destino")
)

vuelos_con_rutas = (
    vuelos_slv
    .join(origen, on="origen_id", how="left")
    .join(destino, on="destino_id", how="left")
)

analisis_rutas = (
    vuelos_con_rutas
    .groupBy(
        "origen_id",
        "nombre_origen",
        "ciudad_origen",
        "destino_id",
        "nombre_destino",
        "ciudad_destino"
    )
    .agg(
        F.countDistinct("vuelo_id").alias(
            "total_vuelos"
        ),

        F.avg("duracion_min").alias(
            "duracion_promedio_min"
        ),

        F.sum("duracion_min").alias(
            "duracion_total_min"
        ),

        F.countDistinct("aeronave_id").alias(
            "aeronaves_utilizadas"
        )
    )
    .orderBy(
        F.desc("total_vuelos")
    )
)

# =====================================================
# 6. GOLD: Resumen mensual de mantenimiento
# =====================================================
# Objetivo:
# Analizar la evolucion mensual de los costos,
# las intervenciones y las horas de mantenimiento.

resumen_mantenimiento_mensual = (
    mantenimientos_slv
    .filter(F.col("fecha").isNotNull())
    .withColumn(
        "anio",
        F.year("fecha")
    )
    .withColumn(
        "mes",
        F.month("fecha")
    )
    .groupBy("anio", "mes")
    .agg(
        F.countDistinct("mantenimiento_id").alias(
            "total_mantenimientos"
        ),

        F.countDistinct("aeronave_id").alias(
            "aeronaves_atendidas"
        ),

        F.sum("costo_usd").alias(
            "costo_total_usd"
        ),

        F.avg("costo_usd").alias(
            "costo_promedio_usd"
        ),

        F.sum("duracion_hr").alias(
            "horas_totales_mantenimiento"
        ),

        F.avg("duracion_hr").alias(
            "duracion_promedio_hr"
        )
    )
    .withColumn(
        "periodo",
        F.format_string("%04d-%02d", "anio", "mes")
    )
    .orderBy("anio", "mes")
)

# =====================================================
# 7. Guardar tablas Gold en Delta
# =====================================================

tablas_gold = {
    "resumen_vuelos_diario": resumen_vuelos_diario,
    "analisis_mantenimiento_aeronave": (
        analisis_mantenimiento_aeronave
    ),
    "analisis_rutas": analisis_rutas,
    "resumen_mantenimiento_mensual": (
        resumen_mantenimiento_mensual
    )
}

for nombre_tabla, dataframe in tablas_gold.items():

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(
            f"{CATALOG}.{GOLD_SCHEMA}.{nombre_tabla}"
        )
    )

    print(
        f"Tabla creada: "
        f"{CATALOG}.{GOLD_SCHEMA}.{nombre_tabla}"
    )

# =====================================================
# 8. Validacion final
# =====================================================

print("\n========== VALIDACION GOLD ==========")

for nombre_tabla in tablas_gold.keys():

    tabla = f"{CATALOG}.{GOLD_SCHEMA}.{nombre_tabla}"

    print(f"\nTabla: {tabla}")

    print(
        "Cantidad de registros:",
        spark.table(tabla).count()
    )

    spark.table(tabla).show(5, truncate=False)

print("\nProceso Gold completado correctamente.")