# Argiope × SegmentR — tutorial de R, de cero a galería

Cómo, partiendo de **una carpeta de imágenes**, obtener máscaras de opistosoma, paletas
CIELAB, una galería paginada y una ficha por ejemplar, conduciendo todo desde R.

El segmentador es una U-Net entrenada que vive en PyTorch. R **no** lo reimplementa: lo
invoca a través de un proceso y luego lee los artefactos del run. Por eso el tutorial tiene
dos mitades — preparar el lado Python una vez, y trabajar desde R siempre.

---

## 0. Qué necesitas antes de empezar

| Pieza | Para qué | Cómo comprobarla |
|---|---|---|
| Entorno conda `argiope` | ejecuta la U-Net (torch + CUDA) | `conda env list` |
| `checkpoints/opistho_unet.pt` | los pesos entrenados | debe existir en la raíz del proyecto |
| R ≥ 4.1 | la capa de este tutorial | `R --version` |
| Este repositorio | `adapt_unet.py` y `R/argiope_segmentR.R` | `git clone` |

### Estructura de directorios

R necesita encontrar dos cosas: el **script del adaptador** y el **checkpoint**. Ésta es la
disposición que el código asume por defecto:

```
Argiope/                              <- raíz del proyecto (parent)
├── checkpoints/
│   └── opistho_unet.pt               <- pesos (no versionados; ~130 MB)
├── data/
│   └── raw/gbif/argiope_bruennichi/  <- una carpeta de imágenes cualquiera
└── repro/segmentr/                   <- ESTE repositorio
    ├── adapt_unet.py                 <- lo que R invoca
    └── R/
        ├── argiope_segmentR.R        <- todo el flujo de R
        └── argiope.R                 <- (opcional) wrapper de `argiope describe`
```

`argiope_segmentR.R` localiza `adapt_unet.py` **relativo a sí mismo**, así que mientras
mantengas `R/` dentro de `repro/segmentr/` no hay rutas que configurar. Si mueves el script,
pásale `adapter = "ruta/a/adapt_unet.py"`.

Las imágenes pueden estar **en cualquier carpeta**, dentro o fuera del proyecto: la ruta se
pasa como argumento. Se buscan recursivamente `.jpg`, `.jpeg` y `.png`.

---

## 1. Preparar el lado Python (una sola vez)

```bash
conda activate argiope
python -c "import torch, segmentation_models_pytorch; print(torch.__version__, torch.cuda.is_available())"
```

Debe imprimir la versión de torch y `True` si tienes GPU (con `False` funciona igual, solo
más lento). Si falta el checkpoint, se regenera con:

```bash
argiope train-segmenter
```

Anota la ruta del intérprete de ese entorno — R la necesita:

```bash
python -c "import sys; print(sys.executable)"
# p.ej. C:/Users/tu_usuario/miniforge3/envs/argiope/python.exe
```

## 2. Instalar los paquetes de R

Tres, todos binarios de CRAN, sin compilación:

```r
install.packages(c("jsonlite", "jpeg", "png"))
```

| Paquete | Para qué |
|---|---|
| `jsonlite` | leer `run_config.json` y los JSON por imagen |
| `jpeg` | leer las fotografías `.jpg` |
| `png` | leer las máscaras `.png` |

El dibujado usa `graphics` y `grDevices`, que vienen con R. La conversión sRGB↔CIELAB usa
`grDevices::convertColor` — la misma llamada que usaba el SegmentR original.

Comprueba:

```r
for (p in c("jsonlite", "jpeg", "png")) cat(p, requireNamespace(p, quietly = TRUE), "\n")
```

## 3. Cargar el flujo y decirle dónde está Python

```r
setwd("C:/ruta/a/Argiope")                       # la raíz del proyecto
source("repro/segmentr/R/argiope_segmentR.R")

options(argiope.python = "C:/Users/tu_usuario/miniforge3/envs/argiope/python.exe")
```

Si lanzaste R desde el entorno conda activado, el `options()` sobra: se encuentra solo.
Puedes comprobarlo con `argiope_python()`, que devuelve la ruta que va a usar o falla con
un mensaje que dice exactamente qué configurar.

## 4. Procesar una carpeta de imágenes

```r
g <- argiope_gallery(
  dir = "data/raw/gbif/argiope_bruennichi",   # la carpeta de entrada
  out = "salidas",                            # dónde escribir el run (por defecto: temporal)
  run_id = "bruennichi",
  n = 40,                                     # opcional: muestra aleatoria con semilla
  seed = 42
)
g
```

```
<argiope gallery> salidas/bruennichi
  40 images · 31 with a mask · 9 empty
  median score 0.968 · pages of 6: 6
```

Esto **carga el modelo una sola vez** para toda la carpeta. Es la diferencia con llamar a
`argiope describe` imagen por imagen, que recargaría los pesos en cada una.

Las imágenes en las que la U-Net no encuentra nada **no se descartan en silencio**: quedan
registradas con su motivo. Argumentos útiles:

| Argumento | Efecto |
|---|---|
| `n = NULL` | procesar la carpeta entera (por defecto) |
| `n_colors = 5` | número de clústeres de color por máscara |
| `reuse = TRUE` | si el run ya existe, lo relee en vez de recalcular |
| `quiet = TRUE` | silencia la salida de Python |

Para releer un run terminado sin volver a ejecutar nada:

```r
g <- argiope_load_gallery("salidas/bruennichi")
```

## 5. Mirar la tabla antes que las imágenes

```r
it <- argiope_items(g)
head(it[, c("image", "group", "has_mask", "score", "px", "reason")])

table(it$has_mask)                    # cuántas con y sin máscara
subset(it, !has_mask)[, c("image", "reason")]   # y por qué fallaron
```

La paleta de una imagen concreta:

```r
argiope_palette_of(g, it$image[it$has_mask][1])
```

```
      hex coverage lab_l lab_a lab_b size
1 #403B36    0.226  25.4   1.0   3.6 2859
2 #655F5A    0.217  40.7   1.2   3.5 2749
...
```

`coverage` es la fracción de píxeles de la máscara que cae en ese clúster; `lab_*` son las
coordenadas CIELAB del centroide. Eso es el «pantone»: HEX + Lab + cobertura.

## 6. La galería en grid paginado

```r
argiope_pages(g)               # cuántas páginas de 6
argiope_plot(g, page = 1)      # dibuja la primera
argiope_plot(g, page = 2)
```

Cada celda lleva la fotografía con el contorno de la máscara en amarillo y el fondo
atenuado, la barra de paleta proporcional a la cobertura, los HEX principales, el score y
el área de la máscara.

Ajustes:

```r
argiope_plot(g, page = 1, per_page = 12, ncol = 4)   # rejilla más densa
argiope_plot(g, include_empty = TRUE)                # incluye las vacías, etiquetadas
argiope_plot(g, maxdim = 900)                        # más resolución por celda (más lento)
```

## 7. Seleccionar ejemplares de una lista

```r
sel <- argiope_pick(g)          # abre una lista de selección múltiple
argiope_plot(g, select = sel)
```

En sesión interactiva `argiope_pick()` abre una lista real (`utils::select.list`) donde
marcas los que quieras. En un script no interactivo devuelve todos, para que los lotes no se
rompan.

También puedes seleccionar sin diálogo, por nombre o por índice:

```r
argiope_plot(g, select = c("0cb8b75b7c53.jpg", "3712a49aa7fa.jpg"))
argiope_plot(g, select = 1:6)
```

## 8. La ficha de un ejemplar

```r
argiope_dashboard(g, "0cb8b75b7c53.jpg")
```

Cuatro paneles: la fotografía con la caja de detección y el contorno de la máscara, los
colores dominantes con HEX / cobertura / Lab, el histograma RGB de los píxeles enmascarados,
y la máscara recoloreada asignando cada píxel a su centroide.

Sin argumento, te deja elegir de una lista:

```r
argiope_dashboard(g)                                   # elige uno
argiope_dashboard(g, file = "ficha.png")               # a PNG en vez de a pantalla
res <- argiope_dashboard(g, "0cb8b75b7c53.jpg")        # devuelve los números
res$mask_px; res$mean_color; res$cluster_sizes
```

Devuelve invisiblemente la paleta, el recuento de píxeles y los colores medio y mediano, de
modo que las cifras del panel están disponibles para el código, no solo dibujadas.

## 9. Exportar todo a PDF

```r
argiope_pdf(g, "galeria_bruennichi.pdf")                        # todas las páginas
argiope_pdf(g, "seleccion.pdf", select = sel, per_page = 4)     # solo lo elegido
```

Los rásteres van sin comprimir dentro del PDF, así que para una galería grande baja
`maxdim` (`argiope_pdf(g, ..., maxdim = 360)`) o exporta páginas sueltas a PNG:

```r
png("pagina_%02d.png", width = 1500, height = 1000, res = 130)
for (p in seq_len(argiope_pages(g))) argiope_plot(g, page = p)
dev.off()
```

## 10. Todo el flujo, de un tirón

```r
source("repro/segmentr/R/argiope_segmentR.R")
options(argiope.python = "C:/Users/tu_usuario/miniforge3/envs/argiope/python.exe")

g   <- argiope_gallery("data/raw/gbif/argiope_bruennichi", out = "salidas",
                       run_id = "bruennichi", n = 40)
it  <- argiope_items(g)
cat(sum(it$has_mask), "de", nrow(it), "con máscara\n")

argiope_plot(g, page = 1)
argiope_dashboard(g, it$image[it$has_mask][1])
argiope_pdf(g, "galeria_bruennichi.pdf")
```

Y desde la terminal, sin abrir R:

```bash
Rscript repro/segmentr/R/argiope_segmentR.R data/raw/gbif/argiope_bruennichi galeria.pdf 40
```

---

## Qué deja escrito en disco

```
salidas/bruennichi/
├── run_config.json      todos los parámetros, semilla, versiones y la lista de imágenes
├── colors.csv           una fila por imagen × clúster: HEX, Lab, cobertura, medio y mediano
├── skipped.csv          las imágenes sin máscara, con su motivo
├── summary.json         procesadas / saltadas
├── json/
│   ├── *.json           un artefacto por imagen (etiqueta, score, caja, máscara)
│   └── masks/*.png      las máscaras binarias
├── qa/*.png             la ficha de cuatro paneles (versión Python)
└── cutouts/*.png        el opistosoma recortado con fondo transparente
```

Un run se reproduce **solo con `run_config.json`**. Y como el color se puede recalcular a
partir de los artefactos sin tocar el modelo:

```bash
python repro/segmentr/adapt_unet.py --from-json salidas/bruennichi
```

Eso reanaliza el color sin cargar la U-Net — útil para probar otro `n_colors` o una paleta
de referencia sin repetir la inferencia.

---

## Problemas frecuentes

**`Could not find the argiope environment's Python`**
R no sabe dónde está el intérprete. `options(argiope.python = ".../envs/argiope/python.exe")`,
o lanza R con el entorno conda ya activado.

**`adapt_unet.py not found`**
Estás llamando al script desde otra ubicación. Pásale la ruta:
`argiope_gallery(dir, adapter = "C:/.../repro/segmentr/adapt_unet.py")`.

**`missing checkpoint: .../opistho_unet.pt`**
No están los pesos. Regénéralos con `argiope train-segmenter`, o pasa
`argiope_gallery(dir, ...)` sobre un proyecto donde sí estén.

**Todas las imágenes salen «sin máscara»**
Comprueba primero una a mano en Python; si ahí también sale vacía, es el modelo y no la capa
de R. Que la U-Net no devuelva nada en una parte de las fotos de campo es un
comportamiento conocido, no un fallo de este código: se registra en `skipped.csv`.

**`package "jpeg" is required`**
`install.packages("jpeg")`. Lo mismo con `png` y `jsonlite`.

**El PDF pesa muchísimo**
Los rásteres van sin comprimir. Baja `maxdim` o exporta a PNG (paso 9).

---

## Qué hace cada pieza

| Capa | Archivo | Responsabilidad |
|---|---|---|
| Modelo | `checkpoints/opistho_unet.pt` | segmenta el opistosoma (PyTorch) |
| Adaptador | `adapt_unet.py` | máscara → color CIELAB, JSON, QA, recortes |
| R | `R/argiope_segmentR.R` | ejecuta el lote, lee artefactos, dibuja grid y fichas |
| R | `R/argiope.R` | alternativa por imagen sobre `argiope describe` |

Las etapas de color, artefacto y QA son un port de **SegmentR** (Boyko 2025, MIT); el
segmentador es propio. Los colores se reportan como HEX, coordenadas Lab y cobertura: no se
afirma compatibilidad con PANTONE®.
