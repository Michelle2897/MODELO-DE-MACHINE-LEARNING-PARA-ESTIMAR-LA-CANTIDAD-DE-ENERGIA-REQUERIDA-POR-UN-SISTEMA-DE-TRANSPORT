# Programación-para-predicción-de-energía-camarán eléctrico funcional en Galápagos
Este repositorio contiene la programación de modelos de aprendizaje automático empleados para la predicción de energía de un catamarán funcionar en Galápagos, para efectos del cumplimiento de proyecto investigativo de masterado en curso.

*Consideraciones: Para efectos de la programación requerida en las distintas faces del proceso de modelación se consideró el uso de dos bases debido al formato de la columna "Fecha". para los modelos ARIMA y SARIMAX se aplico un formato fecha día, hora, minutos, segundos. No obstante, este formato fue requerido al ejecutar el Análisis de componentes principales y la Red Neuronal.
La nombrada "Base_final_1.7" será usada en la modelación ACP y Red neuronal
La nombrada "Base_final_1.8" será usada en la modelación de los modelos de media movil*


# Análisis de componentes principales (PCA)
A continuación de explica el proceso de la programación por etapas del PCA para la correoboración de variables representativas para el estudio.

# Importar las bibliotecas necesarias
Partiendo del supuesto de haber realizado un análisis exploratorio de las variables que vas a someterse al estudio y familiarizarse con el comportamiento se importan las bibliotecas que se utilizarán. Las más comunes son pandas para la manipulación de datos, numpy para operaciones numéricas y scikit-learn importando PCA para el análisis de componentes principales.
# Carga y procesamiento de datos
El siguiente paso consiste en cargar el conjunto de datos y realizar el preprocesamiento.  Se inicia el proceso se preparación de variables, iniciando por declarar todas las variables que participarán en el análisis en un solo Dataframe, asignar el formato correcto a la variable que contiene fecha y hora en el formato correcto de manera que no detenga los procesos a futuro y se declaran las variables numéricas como tipo float. En esta etapa se incluye la gestión de valores nulos, normalización y escalamiento de datos. 
# Aplicación del PCA
Una vez que los datos están listos, se aplica el PCA desde la librería sklearn. Primero se define creas una instancia del objeto designado al análisis PCA y se especifica el número de componentes que deseas conservar, el presente caso seleccionaremos 2 componentes principales para definir la representatividad de las variables.
# Visualización del PCA y análisis de resultados
Después de aplicar PCA, es importante analizar los resultados para lo cual se estandarizan los datos de manera que se facilite la interpretación. Se imprime la varianza explicada por cada componente. Posterior a ello se implementa un nuevo PCA para componentes adicionales en caso de su existencia. Se generan gráficos, medidas de tendencia e indicadores que permiten la interpretación de representatividad de las variables de estudios, así como la determinación de carga de cada componente. Finalmente se realiza una exploración individual para las variables representativas definidas.

# MODELACIÓN CON SERIES TEMPORALES(ARIMA/SARIMA)

# Modelo ARIMA

# Preparación de datos 
Previo a la programación del modelo, se realiza la exploración de datos a usarse para obtener las estadísticas correspondientes a las variables de las que está compuesta la base. El inicio del proceso consiste en cargar y preparar los datos de la serie temporal. Esto incluye la limpieza de datos, la conversión de la columna de fecha a un tipo de dato datetime especificando el formato, que para efectos de la variable “Fecha” será en datos formato fecha y hora; para luego especificar la frecuencia horaria y finalmente el establecimiento de esta columna como índice del DataFrame. 
Es crucial examinar las variables en busca de valores atípicos o datos faltantes, que deben ser tratados adecuadamente implementando correcciones para limpiar valores numéricos con separadores incorrectos y manejar errores. Adicional se transforman dos datos valores reales tipo float para implementar la normalización de datos. Además, es importante visualizar la serie utilizando gráficos para identificar patrones, tendencias y estacionalidad.
# Estacionariedad y Diferenciación
Una vez que los datos están preparados, se debe comprobar la estacionariedad de la serie. Para comprobar que la serie temporal se encuentra completa se grafica la variable “Energía (Mwh)” con la serie temporal, a manera de una prueba visual de patrón temporal. Posteriormente se aplican pruebas específicas para estacionalidad utilizando la prueba de Dickey-Fuller aumentada (AD), como resultado la serie es estacionaria. Si la serie no es estacionaria, se aplican diferencias a los datos para eliminar tendencias y estabilizar la varianza. Es común utilizar el método de diferenciación una o más veces hasta lograr una serie estacionaria, que se puede verificar nuevamente con la prueba ADF.
# Identificación de Parámetros
En esta etapa, se analizan las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF) para identificar los parámetros del modelo ARIMA. Los gráficos ACF ayudan a determinar el orden del componente de media móvil (q), mientras que los gráficos PACF indican el orden del componente autorregresivo (p). La selección del parámetro de diferenciación (d) se basa en el número de diferencias aplicadas previamente para lograr la estacionariedad.
# Ajuste del Modelo
Con los parámetros identificados, se procede a ajustar el modelo ARIMA utilizando la función ARIMA del paquete statsmodels. Se especifican los valores de p, d y q, y se entrena el modelo con los datos históricos desde el 2022. Se realiza una división de los datos en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo una vez se emule el comportamiento de datos pasados con el conjunto de entrenamiento para realizar el test de prueba y generar los pronósticos.
# Evaluación del Modelo
Una vez ajustado el modelo, se evalúa su desempeño mediante métricas como el error cuadrático medio (MSE) o el error absoluto medio (MAE). Además, se analizan los residuos del modelo para verificar que no exista autocorrelación significativa, utilizando la prueba de Ljung-Box. Gráficos de los residuos, como el gráfico de dispersión, también son útiles para asegurarse de que los residuos se comporten como ruido blanco.
# Pronóstico
Finalmente, el modelo ajustado se utiliza para realizar pronósticos a futuro. Se puede utilizar el método forecast del modelo ARIMA para obtener predicciones que se visualizan junto con los valores reales de entrenamiento y test aplicados en gráficamente. Esta etapa es crucial para entender cómo el modelo se comportará en un horizonte temporal específico. Programar un modelo ARIMA en Python implica una serie de pasos que van desde la preparación de datos hasta la evaluación y pronóstico, asegurando que se sigan las mejores prácticas para obtener resultados precisos y significativos

# Modelo SARIMAX 

# Importación de librerías 
La construcción de un modelo SARIMA con variable exógena en, si bien es cierto considera los mismos pasos del modelo ARIMA, sin embargo, es un proceso que requiere a algunos detalles. Para empezar, es esencial importar las bibliotecas necesarias para la extensión del modelo. Las más comunes son pandas para la manipulación de datos, statsmodels para el modelado, y matplotlib para la visualización. Una vez importadas las bibliotecas, el paso a seguir es cargar el conjunto de datos revisaremos que la serie temporada esté definida como índice, idealmente en un formato de fecha.
# Ajuste del modelo
Después de haber analizado los datos, se procede a preparar los parámetros del modelo SARIMA. Esto implica definir ppp, ddd, qqq, PPP, DDD, QQQ y sss, donde ddd es el grado de diferenciación. Para la construcción de la variable exógena que usaremos se calcula el rendimiento de la variable “Energía MWh” y posteriormente a partir del cálculo del rendimiento se obtiene la variable “Volatilidad” misma que también debe estar preparada, asegurando que tenga la misma longitud que la serie temporal. Con todos estos elementos listos, se ajusta el modelo utilizando la función SARIMAX de la biblioteca statsmodels, que permite incluir variables exógenas.
# Evaluación del modelo
Una vez que el modelo ha sido ajustado, es fundamental evaluar su rendimiento. Esto se realiza dividiendo los datos en conjuntos de entrenamiento y prueba, y luego realizando predicciones sobre el conjunto de prueba. Se pueden utilizar métricas como impresas en el cuadro value SARIMAX.  Además, es recomendable realizar un análisis de residuos para verificar que no existan patrones no capturados por el modelo.
# Pronóstico de datos
Finalmente, se pueden visualizar los resultados. Esto implica graficar las predicciones junto con la serie temporal original y la variable exógena para entender mejor cómo interactúan. También es útil comparar las predicciones del modelo con las observaciones reales, lo que puede ofrecer insights sobre la efectividad del modelo y posibles áreas de mejora. Con este enfoque estructurado, puedes construir un modelo SARIMA robusto que incorpore la cantidad necesaria de variables exógenas, mejorando así la calidad de tus predicciones.

# Red Neuronal con autoencoder aplicado 

# Importación de librerías 
El primer paso es importar las bibliotecas necesarias. Para construir la red neuronal, es común utilizar TensorFlow y Keras, que proporcionan herramientas eficientes para crear y entrenar modelos específicamente de redes. También se incluyen librerías usadas con anterioridad como numpy y pandas para la manipulación de datos, así como matplotlib para visualizar los resultados.
# Preparación de datos
Una vez que los datos están cargados, se aplica un análisis exploratorio a las variables, para determinar el comportamiento y las métricas de cada una. Se procede a agrupar la columna “Fecha”. Esto incluye definir y visualizar la serie temporal para identificar tendencias, patrones estacionales y posibles outliers usando gráficos boxplot y de dispersión. Además, se la matriz de correlación entre las variables para determinar aquellas que serán las variables declaradas en la red como independientes y dependiente. 
# Ajuste de la Red
Se establecen las columnas en conjuntos diferentes de entrada y salida, aquellas que serán las independientes y la columna dependiente o a proyectarse. En el tercer paso, se define la arquitectura del autoencoder. Un autoencoder típicamente consta de un codificador y un decodificador. La parte del codificador reduce la dimensionalidad de la entrada, mientras que el decodificador intenta reconstruir la entrada original a partir de esta representación comprimida, para efectos de la red se deben escalar los datos de entrada y salida, decir el número de épocas de la serie de tiempo que en este caso se hará para 365 días. Posteriormente se definen las capas a generar en la red.
# Entrenamiento 
La fase de entrenamiento de un autoencoder implica la optimización de su estructura y el aprendizaje, para lo cual se dividen los datos en conjunto de entrenamiento y prueba. Con el uso de técnicas de retro propagación y optimizadores, como Adam o SGD, se ajustan los pesos de la red para minimizar esta pérdida a lo largo de múltiples épocas. Así, el modelo se va refinando, aprendiendo a captar las características esenciales de los datos, lo que le permite generalizar y realizar tareas como reducción de dimensionalidad o eliminación de ruido.
# Creación y optimización 
Se general el modelo desde la función créate_autoencoder para entrada y salida definidas, se implementan capas LSTM adicionales que intentan reconstruir la secuencia original a partir del vector comprimido. Es fundamental especificar parámetros como el número de unidades en cada capa, la función de activación, y la configuración de las capas de salida para asegurar que la arquitectura se adapte adecuadamente a la naturaleza de los datos y a los objetivos de la tarea específica. En este caso, se puede optar por el optimizador RMSprop, para adaptar la tasa de aprendizaje de manera dinámica, definiendo el optimizador, la función de pérdida y, opcionalmente, métricas adicionales para monitorizar el rendimiento durante el entrenamiento.
# Evaluación de la Red
En este paso se evalúa el rendimiento del autoencoder y se visualizan los resultados, incluye la comparación entre las entradas originales con las salidas reconstruidas para verificar la efectividad del modelo. Se imprimen los resultados de la red programada y se generan gráficos para visualizar, adicionalmente se general los valores de métricas de valor para interpretación del ajuste del modelo.
# Pronostico y visualización de refultados
En la fase de pronóstico de predicciones con un autoencoder se utiliza el modelo entrenado para hacer predicciones sobre nuevos datos calculados para escenarios de demanda específicos. Con el modelo evaluado, se pueden realizar predicciones aplicando el método predict() sobre los datos de interés. Los resultados obtenidos se comparan visualmente con las entradas originales, lo que facilita la identificación de patrones y errores en la reconstrucción. Esta fase es crucial, ya que permite validar la efectividad del autoencoder en tareas como la detección de anomalías o la predicción de series temporales, se generan gráficos complementarios que permiten visualizar la eficiencia de la red así como las predicciones graficadas conjuntamente con la energía real del periodo de prueba y entrenamiento.



