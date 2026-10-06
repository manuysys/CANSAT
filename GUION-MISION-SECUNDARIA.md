# Guión — Misión secundaria (2–3 min) · CanSat LB135

> Hablado, sin tecnicismos: **qué es, qué hicimos y en qué estamos**. La idea es
> que lo entienda cualquiera, aunque no sepa nada de inteligencia artificial.
> El detalle técnico está en el Informe, para quien lo pida.

## Antes de empezar (2 min)

1. `cd EstacionTerrena_MuestreoDeDatos && python web_server.py` → abrir
   **http://localhost:8000** (pantalla completa).
2. Tener a mano la imagen del **esquema**
   (`EstacionTerrena_MuestreoDeDatos/docs/esquema_mision_secundaria.png`) y,
   por si falla algo, el **video de respaldo** y el **póster**.

## Guión (hablado)

### 0:00 — Qué es *(mostrar el esquema)*

"Nuestra misión secundaria, en una frase, es **una cámara que piensa**.
Mientras el CanSat baja, una cámara especial mira el suelo y va entendiendo lo
que ve: qué parte es vegetación, qué parte son edificios, dónde hay agua.
Y una computadora muy chiquita guarda cada foto junto con la posición y la
altura del CanSat en ese momento. Después, en tierra, la estación terrena arma
**la película completa de la misión**."

### 0:25 — La estación *(mostrar la vista Vuelo)*

"Esto es la estación terrena. Acá se ve el recorrido del CanSat, las fotos que
fue sacando y un resumen de lo que vio. La pensamos para que **cualquiera mire
la pantalla y entienda qué pasó**, sin ser experto. Y todo funciona **sin
internet**, porque en el aula no vamos a tener."

### 0:50 — Qué hicimos *(mostrar el detalle de un frame y el Grad-CAM)*

"¿Y qué hicimos todo este tiempo? Primero, la inteligencia: la cámara no solo
saca fotos, las **clasifica**; **cuenta personas**; y en tierra estimamos
**daños, inundaciones e incendios**. Después vino la parte más difícil y la más
linda: hacerlo funcionar en el **hardware real**. Nos llegó la placa con la
cámara y las pusimos a prueba de verdad. Ahí descubrimos algo clave: el cerebro
de la cámara puede **detectar personas por su cuenta**, sin cargar el
procesador — y eso importa muchísimo, porque la computadora es diminuta.
Y lo más nuevo: **nuestro propio modelo**, el que entrenamos nosotros, ahora
**corre adentro de la cámara**."

### 1:30 — Cómo lo probamos *(foto de la Pi o demo del modelo en vivo)*

"No nos quedamos en la simulación. Conectamos la placa con los **sensores
reales**, la hicimos **hablar con la computadora de vuelo**, medimos cuánto
tarda cada foto, dejamos el sistema corriendo **diez minutos seguidos** para ver
si perdía datos, y probamos la **radio y el GPS**. Todo quedó documentado, con
las cosas que salieron bien y también con las que no."

### 2:00 — En qué estamos *(mostrar el post-vuelo o volver al esquema)*

"Ahora estamos **mejorando los modelos** y preparando el **vuelo de verdad**:
el armado del CanSat, la prueba en el predio y los últimos detalles. Y algo
que nos parece importante: cada idea nueva la **medimos**. Si mejora, la
adoptamos; y si no mejora, **también lo contamos**. Varias ideas nuestras
quedaron en el camino, y eso también es parte del trabajo."

### 2:35 — Cierre

"En resumen: construimos un sistema que **ve, entiende y cuenta** lo que pasa
abajo, que funciona en **hardware real** y que se puede mostrar **en vivo**.
Y todo lo que les contamos acá lo probamos nosotros."

## Qué mostrar en cada momento (resumen)

| Momento | Pantalla |
|---|---|
| Arranque | Esquema de la misión (`esquema_mision_secundaria.png`) |
| La estación | Vista **Vuelo**: recorrido + fotos + resumen |
| Qué hicimos | **Detalle** de un frame + botón **Grad-CAM** |
| Cómo lo probamos | Foto real de la Pi, o el modelo en vivo en la cámara (si hay tiempo) |
| En qué estamos | **Post-vuelo**/Informe, o volver al esquema |
| Cierre | Póster o portada |

## Checklist rápido

- [ ] Servidor de la estación corriendo, navegador en pantalla completa.
- [ ] Esquema abierto en el visor de imágenes.
- [ ] Un frame con Grad-CAM ya abierto una vez (queda en caché).
- [ ] Pi encendida (si se muestra en vivo).
- [ ] Video de respaldo y póster a mano (plan B).
