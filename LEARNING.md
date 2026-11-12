# Lo que aprendimos construyendo exoplanet-cnn (noviembre de 2026)

Notas en español para Christian. Explican las ideas de este repositorio en el orden en que tienen sentido.
Los números de resultados están en el README y salen de nuestras propias ejecuciones.

## 1. Qué es un TCE

Kepler observó unas 200.000 estrellas (Shallue y Vanderburg 2018) durante cuatro años, midiendo su brillo cada 29,4 minutos. Un programa de
la misión (el *Transiting Planet Search*) busca en cada curva de luz caídas de brillo que se repiten con un
periodo fijo. Cada señal que supera el umbral (un "multiple event statistic", MES, de 7,1 sigma) se llama
**TCE**: *Threshold Crossing Event*, "evento que cruzó el umbral".

Un TCE **no** es un planeta. Es solo "algo se repite". Puede ser:

- un **planeta** que transita (PC, *planet candidate*);
- un **falso positivo astrofísico** (AFP): casi siempre una **binaria eclipsante**, dos estrellas que se
  eclipsan entre sí, a veces una binaria de fondo cuya luz se mezcla con la de nuestra estrella;
- un **fenómeno que no es un tránsito** (NTP): ruido del instrumento, saltos, estrellas variables.

En el catálogo DR24 hay 3.600 PC, 9.596 AFP y 2.541 NTP con etiqueta (Shallue y Vanderburg 2018, sec. 3.1).
Esas etiquetas las pusieron personas (el equipo TCERT) y otras pruebas; son nuestra "verdad", con algunos
errores conocidos (el artículo cita dos binarias etiquetadas como PC).

Es exactamente el mismo problema que tenemos en nuestra búsqueda de enanas blancas: el survey encuentra una
señal periódica y hay que decidir si es real. Este repositorio aprende a hacer esa decisión con una red
neuronal, sobre datos donde sí sabemos la respuesta.

## 2. Por qué dos "vistas" de la misma curva

Se dobla la curva de luz sobre el periodo del TCE (todas las órbitas encima de la anterior, con el tránsito en
el centro) y se resume en un vector de longitud fija con medianas por intervalo:

- **vista global** (2001 puntos): la órbita completa. Ahí se ve si hay un **eclipse secundario** a media
  órbita, la firma de una binaria: un planeta casi no emite luz, una estrella sí.
- **vista local** (201 puntos): solo ±4 duraciones alrededor del tránsito, con mucha resolución. Ahí se ve la
  **forma**: un planeta da una caída en forma de **U** (fondo plano mientras el disco del planeta está entero
  sobre la estrella); una binaria rasante da una **V**.

Si solo usáramos la global, un tránsito de 5 horas en una órbita de 300 días ocuparía 1 o 2 de los 2001
puntos. Por eso hacen falta las dos. En el artículo, usar ambas mejora todas las arquitecturas (tabla 2).

Cada vista se normaliza a mediana 0 y mínimo −1, así la red ve la forma, no la profundidad.

## 3. Lo que añadimos: pensar como un experto (ExoMiner)

Un astrónomo que revisa un TCE mira el informe de validación de datos (DV) de Kepler y hace pruebas concretas.
ExoMiner (Valizadegan et al., arXiv:2111.10009) convirtió cada prueba en una rama de la red. Nosotros añadimos tres:

- **impares contra pares**: si el pipeline detecta una binaria a la mitad de su periodo verdadero, el eclipse
  primario y el secundario caen alternadamente como tránsito "impar" y "par", y casi nunca tienen la misma
  profundidad. Un planeta da siempre la misma profundidad. La misma rama convolucional procesa la vista de tránsitos impares y
  la de pares, y la red recibe la **resta** de sus rasgos (idea de ExoMiner).
- **vista del secundario**: una ventana local centrada en la fase del evento secundario más fuerte que
  encontró el pipeline (`tce_maxmesd`).
- **números escalares del DV**: profundidad, duración, periodo, radio del planeta, significación del
  desplazamiento del centroide (si la luz que baja no viene de nuestra estrella sino de una vecina), etc.

Detalle importante: impares, pares y secundario se escalan con la **misma** normalización que la vista local,
para que una diferencia de profundidad no se borre al normalizar.

## 4. Cómo se mide si funciona

La red da una probabilidad entre 0 y 1. Con un umbral (por ejemplo 0,5) se decide planeta o no planeta.

- **Precisión** (fiabilidad): de lo que llamé planeta, ¿qué fracción lo es?
- **Recall** (completitud): de los planetas reales, ¿qué fracción encontré?
- Subir el umbral sube la precisión y baja el recall.
- **AUC** (área bajo la curva ROC): la probabilidad de que un planeta elegido al azar reciba una puntuación más
  alta que un falso positivo elegido al azar. No depende del umbral. 0,5 es tirar una moneda; 1 es perfecto.
- **Recall a precisión 99 %**: si exijo que el 99 % de mis "planetas" sean reales, ¿cuántos encuentro? Es la
  métrica que destaca ExoMiner (0,936 en su conjunto de prueba, con su propio catálogo y etiquetas: no es
  directamente comparable con el nuestro).
- **Calibración**: si la red dice 0,8, ¿son planetas el 80 % de esos casos? Una red puede ordenar bien (AUC
  alta) y aun así estar mal calibrada.

## 5. Dos precauciones que tomamos

- **Separar por estrella, no por TCE.** Una binaria eclipsante suele producir varios TCE (el eclipse primario,
  el secundario, alias del periodo). Si uno cae en entrenamiento y otro en prueba, la red "reconoce" la estrella
  y el resultado parece mejor de lo que es. El artículo original separó por TCE; nosotros por estrella.
- **Un subconjunto honesto.** Descargar las 9.865 estrellas con TCE etiquetados serían unos 70 GB (medimos unos 7 MB por estrella). Usamos 1.900 estrellas elegidas al azar
  (3.043 TCE, 13,2 GB), con la mezcla natural de clases. La red ve cinco veces menos ejemplos que en el
  artículo, así que esperábamos números algo peores, y lo decimos.

## 6. Qué salió (prueba: 318 TCE de 190 estrellas que la red nunca vio, 65 de ellos planetas)

- **AstroNet reproducido:** AUC 0,973 (intervalo del 95 %: 0,950–0,989) y exactitud 0,947. El artículo, con
  cinco veces más datos, da 0,988 y 0,960. Su AUC cae dentro de nuestro intervalo: la reproducción funciona.
- **Con las vistas de experto y los números del DV:** AUC 0,995. Exigiendo que el 99 % de lo que llamo
  "planeta" sea real, encuentra 51 de los 65 planetas; AstroNet solo 10.
- **La sorpresa honesta:** casi toda la mejora viene de los números del catálogo, no de las vistas nuevas. Solo
  con esos números (sin mirar la curva de luz) un modelo de árboles ya da AUC 0,988, pero su TCE más "seguro"
  es una binaria, así que no puede llegar al 99 % de fiabilidad. La curva de luz arregla la cabeza de la lista;
  los números arreglan el resto.
- **Un error que nos enseñó algo:** al principio borrábamos de la curva los tránsitos de los otros TCE de la
  misma estrella, como hace el artículo con los planetas confirmados. En una binaria, ese "otro TCE" suele ser
  su propio eclipse secundario, así que borrábamos la prueba del delito. Lo vimos en la figura de ejemplo y lo
  corregimos: ahora no se borran los TCE cuyo periodo es múltiplo o mitad del periodo del que estudiamos.

## 7. Qué tiene que ver con las enanas blancas

Nuestro survey de enanas blancas produce candidatos que hoy revisamos a mano con la lista de la regla 1 del
proyecto (impares y pares, secundario, vecinos en el píxel de TESS...). Es la misma lista que ExoMiner convierte
en ramas de la red. El siguiente paso natural es entrenar algo así con tránsitos inyectados en curvas de TESS
de enanas blancas, donde los tránsitos duran minutos en vez de horas.

## Ejercicio

Abre `figures/example_views.png`. En la fila de la binaria, ¿qué vista la delata? Ahora piensa: ¿qué vería la
red si la binaria tuviera dos estrellas idénticas y el pipeline hubiera encontrado la mitad del periodo real?
(Pista: la vista de impares y pares.)
