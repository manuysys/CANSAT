# Guión — Misión secundaria (3:30) · CanSat LB135

> Explicativo, con lo técnico justo: los datasets, los tipos de modelos y los
> números que más importan. No es un listado de métricas: es lo que hace falta
> para que se entienda. El detalle completo está en el Informe.

## Antes de empezar (2 min)

1. `cd EstacionTerrena_MuestreoDeDatos && python web_server.py` → abrir
   **http://localhost:8000** (pantalla completa).
2. Tener a mano la imagen del **esquema**
   (`EstacionTerrena_MuestreoDeDatos/docs/esquema_mision_secundaria.png`) y,
   por si falla algo, el **video de respaldo** y el **póster**.

## Guión (hablado)

### 0:00 — Qué es *(mostrar el esquema)*

"Nuestra misión secundaria tiene dos mitades: **una cámara que piensa** a
bordo, y una **estación terrena** que reconstruye la misión. Mientras el CanSat
baja, la cámara mira el suelo y va entendiendo lo que ve: vegetación, edificios,
agua; cuenta personas; y guarda cada foto con la posición y la altura. Después,
en tierra, la estación arma **la película completa de la misión**."

### 0:30 — La estación *(mostrar la vista Vuelo)*

"Esto es la estación: el recorrido del CanSat, las fotos que sacó y un resumen
pensado para que **cualquiera entienda qué pasó**, sin ser experto. Funciona
**sin internet** y además permite **preguntarle cosas** — por ejemplo, cuántas
personas hay cerca de un camino — y responde con lo que midió, **sin inventar**."

### 1:00 — Qué construimos *(mostrar el detalle de un frame y el Grad-CAM)*

"Para lograrlo entrenamos **una familia de modelos**, cada uno con un dataset
público distinto. El de **terreno** aprendió con imágenes satelitales y
distingue cinco clases —vegetación, edificios, agua, suelo y otros— con una
precisión por clase de alrededor del **40 %** (entre 36 y 52 % según la clase). El de **daño** aprendió con
catástrofes reales, y sobre todo con **tomas de dron**, porque el CanSat mira
como un dron, no como un satélite: ahí llega al **78 %**. Sumamos
**inundaciones**, **fuego y humo**, y un modelo de **severidad** que estima
cuánto se destruyó. Y para las **personas**, un detector entrenado con imágenes
aéreas, que corre dentro de la cámara."

### 1:50 — El hardware real *(foto de la Pi o demo del modelo en vivo)*

"Y no quedó en la computadora: lo hicimos andar en el **hardware real**. La
placa es una Raspberry Pi diminuta con una cámara que tiene **cerebro propio**:
descubrimos que puede detectar personas **sola, sin cargar el procesador** — y
eso importa muchísimo, porque la computadora es humilde. Lo más nuevo es que
**nuestro propio modelo, el que entrenamos nosotros, ahora corre adentro de esa
cámara**. La conectamos con los sensores reales —presión, temperatura,
movimiento, GPS— y con la radio; la dejamos corriendo **diez minutos seguidos**
para ver si perdía datos, y **no perdió ninguno**."

### 2:30 — En qué estamos *(mostrar el post-vuelo o volver al esquema)*

"Ahora estamos mejorando los modelos y preparando el **vuelo de verdad**: el
armado del CanSat, la prueba en el predio y los últimos detalles. Y algo que
nos importa: cada idea nueva la **medimos**. En este tiempo adoptamos un modelo
de daño mejor y una corrección de imagen que sube la precisión; integramos un
sistema que **combina dos modelos** según lo que ve cada foto. Y lo que no
funcionó —varias ideas nuestras— **también lo contamos**: probamos, medimos y
descartamos. Eso también es parte del trabajo."

### 3:10 — Cierre

"En resumen: construimos un sistema que **ve, entiende y cuenta** lo que pasa
abajo, que funciona en **hardware real** y que se puede mostrar **en vivo**.
Y todo lo que les contamos acá lo probamos nosotros."

## Los datos técnicos que sí conviene decir (y no más)

| Tema | Qué decir |
|---|---|
| Terreno | 5 clases (vegetación, edificios, agua, suelo, otros); ~40–50 % por clase; el modelo que vuela es **liviano** a propósito (la placa es humilde) |
| Daño | Entrenado con **xBD** (satélite) y **RescueNet** (dron): **0.78** en dominio dron |
| Inundación / Fuego | **FloodNet** (~0.49) y **FLAME** (~0.79) |
| Severidad | 5 niveles de destrucción; alimenta la estimación de pérdidas |
| Personas | Detector con imágenes aéreas (**VisDrone**); corre en el **NPU** de la cámara |
| Hardware | Pi Zero + AI Camera IMX500; sensores reales; radio LoRa; GPS |
| Honestidad | Todo medido; lo que no funcionó se descartó y se documentó |

## Qué mostrar en cada momento (resumen)

| Momento | Pantalla |
|---|---|
| Arranque | Esquema de la misión |
| La estación | Vista **Vuelo**: recorrido + fotos + resumen |
| Qué construimos | **Detalle** de un frame + botón **Grad-CAM** |
| El hardware | Foto real de la Pi, o el modelo en vivo en la cámara |
| En qué estamos | **Post-vuelo**/Informe, o volver al esquema |
| Cierre | Póster o portada |

## Checklist rápido

- [ ] Servidor de la estación corriendo, navegador en pantalla completa.
- [ ] Esquema abierto en el visor de imágenes.
- [ ] Un frame con Grad-CAM ya abierto una vez (queda en caché).
- [ ] Pi encendida (si se muestra en vivo).
- [ ] Video de respaldo y póster a mano (plan B).
