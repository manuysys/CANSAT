# Guion — Misión secundaria CanSat LB135

### Duración objetivo: 3 minutos 30 segundos

> Explicativo y ameno, con lo técnico justo (datasets, tipos de modelos y un
> número por modelo). El detalle completo está en el Informe.

## Antes de empezar (2 min)

1. `cd EstacionTerrena_MuestreoDeDatos && python web_server.py` → abrir
   **http://localhost:8000** (pantalla completa).
2. Tener a mano el **esquema** (`EstacionTerrena_MuestreoDeDatos/docs/
   esquema_mision_secundaria.png`) y, por si falla algo, el **video de respaldo**
   y el **póster**.

---

**0:00 — Qué hace nuestra misión**
*(Mostrar el esquema de la misión)*

"En nuestra misión secundaria hay dos partes que trabajan juntas: **una cámara
que analiza lo que ve durante el vuelo y una estación terrena que reconstruye
toda la misión**.

Mientras el CanSat desciende, la cámara observa el terreno y busca identificar
**vegetación, edificios, agua y suelo**. También puede detectar **personas** y
analizar situaciones de **daño, inundación o fuego**.

Además, cada imagen queda asociada a información del vuelo, como **posición y
altura**. Y después, cuando termina el descenso, toda esa información se lleva a
tierra para reconstruir lo que pasó durante la misión."

---

**0:30 — La estación terrena**
*(Mostrar la vista "Vuelo": escena 3D + recorrido)*

"Esta es nuestra estación terrena.

Lo primero que aparece es una **escena en 3D del descenso**: el CanSat, el
paracaídas, la altura y la velocidad en cada momento — una forma muy visual de
seguir el vuelo.

Alrededor está **el recorrido sobre el mapa, las imágenes que fue capturando y
un resumen de lo que encontró**. La idea es que no sea solamente una herramienta
técnica, sino que permita **entender rápidamente qué pasó**, incluso sin conocer
los detalles del sistema. Se puede **recorrer la misión como una película**
—vuelo, post-vuelo, informe y una vista pensada para el jurado—, se **actualiza
sola** y también puede mostrar el vuelo **en vivo, mientras la Pi captura**.

Todo funciona **sin internet**, y permite hacer consultas sobre los datos: por
ejemplo, **cuántas personas detectó cerca de un camino**, con la respuesta
construida a partir de las mediciones reales."

---

**1:00 — Los modelos de inteligencia artificial**
*(Mostrar el detalle de un frame y después Grad-CAM)*

"Para conseguir esto desarrollamos **una familia de modelos**, porque cada
problema necesita datos y herramientas distintas.

Para el **terreno** usamos imágenes satelitales: distingue **cinco clases
—vegetación, edificios, agua, suelo y otros—** con un desempeño de alrededor del
**40 % por clase**. Elegimos una arquitectura liviana porque tiene que funcionar
en una computadora limitada.

Para **daño** combinamos datasets como **xBD y RescueNet**. Y esto fue
importante: para nuestro caso las imágenes de dron son mucho más parecidas a lo
que realmente va a ver el CanSat que una imagen satelital. En ese dominio
llegamos a **0,78 de desempeño**.

También sumamos modelos de **inundación y fuego** (FloodNet y FLAME) y uno de
**severidad**, que clasifica el nivel de destrucción en cinco categorías. Y para
las personas, un detector entrenado con **imágenes aéreas de VisDrone**."

---

**1:50 — El hardware real**
*(Mostrar la Raspberry Pi / cámara / demo en vivo)*

"Lo más importante es que esto no quedó en una computadora de escritorio:
**lo llevamos al hardware que realmente vamos a utilizar**.

Usamos una **Raspberry Pi Zero junto con una AI Camera basada en el IMX500**.
La cámara tiene procesamiento propio, así que **la detección de personas corre
en su NPU sin cargar el procesador de la Raspberry**.

Y hay un avance que para nosotros es especialmente importante: **nuestro propio
modelo, el que entrenamos nosotros, también conseguimos ejecutarlo dentro de la
cámara**.

Además, conectamos el sistema con los sensores reales —**presión, temperatura,
movimiento y GPS**— y con la radio LoRa. Hicimos una prueba continua de **diez
minutos**, y durante esa prueba **no perdimos ningún dato**."

---

**2:30 — Qué estamos haciendo ahora**
*(Mostrar Post-vuelo / Informe o volver al esquema)*

"Ahora estamos en la etapa de **optimización y preparación del vuelo real**:
seguimos mejorando los modelos, terminando el armado del CanSat y preparando las
pruebas en el predio.

Y hay algo que para nosotros es fundamental: **cada decisión la tomamos a partir
de pruebas**. Incorporamos un modelo de daño que mostró mejores resultados, una
corrección de imagen que mejora la precisión y un sistema que puede **combinar
dos modelos según lo que encuentra en cada imagen**.

Pero también documentamos las cosas que no funcionaron: probamos distintas
ideas, las medimos y, cuando no alcanzaron, **las descartamos**. Eso también
forma parte del desarrollo."

---

**3:10 — Cierre**
*(Mostrar póster o portada)*

"En resumen, construimos un sistema que **ve, analiza y cuenta lo que ocurre
debajo del CanSat**, integrado con **hardware real, sensores, GPS y
comunicaciones**, y una estación terrena capaz de reconstruir y mostrar toda la
misión.

Y algo que queremos destacar: **cada resultado que presentamos fue probado,
medido y documentado por nosotros**. Por eso no es solamente una idea: es un
sistema que ya estamos llevando a una implementación real."

---

## Qué mostrar en cada momento

| Momento | Pantalla |
|---|---|
| Arranque | Esquema de la misión (`esquema_mision_secundaria.png`) |
| La estación | Vista **Vuelo**: escena 3D + recorrido + fotos + resumen |
| Modelos | **Detalle** de un frame + botón **Grad-CAM** |
| Hardware | Foto real de la Pi, o el modelo en vivo en la cámara |
| Ahora | **Post-vuelo**/Informe, o volver al esquema |
| Cierre | Póster o portada |

## Los datos técnicos que sí conviene decir (y no más)

| Tema | Qué decir |
|---|---|
| Terreno | 5 clases; ~40 % por clase; modelo **liviano** a propósito (la placa es humilde) |
| Daño | **xBD** (satélite) + **RescueNet** (dron): **0.78** en dominio dron |
| Inundación / Fuego | **FloodNet** (~0.49) y **FLAME** (~0.79) |
| Severidad | 5 niveles de destrucción; alimenta la estimación de pérdidas |
| Personas | Detector con imágenes aéreas (**VisDrone**); corre en el **NPU** |
| Estación | 3D del descenso · mapa + corredor · consultas · informe imprimible · sin internet |
| Honestidad | Todo medido; lo que no funcionó se descartó y se documentó |

## Checklist rápido

- [ ] Servidor de la estación corriendo, navegador en pantalla completa.
- [ ] Esquema abierto en el visor de imágenes.
- [ ] Un frame con Grad-CAM ya abierto una vez (queda en caché).
- [ ] Pi encendida (si se muestra en vivo).
- [ ] Video de respaldo y póster a mano (plan B).
