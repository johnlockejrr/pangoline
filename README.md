# PangoLine

PangoLine is a basic tool to render raw (horizontal) text into PDF documents
and create parallel ALTO files for each page containing baseline and bounding
box information. 

It is intended to support the rendering of most of the world's writing systems
in order to create synthetic page-level training data for automatic text
recognition systems. Functionality is fairly basic for now. PDF output is
single column, justified text without word breaking. Paragraphs are split
automatically once a page is full.

## Installation

You'll need PyGObject and the Pango/Cairo libraries on your system. As
PyGObject is only shipped in source form this also requires a C compiler and
the usual build environment dependencies installed. An easier way is to use conda:

    ~> conda create --name pangoline-py3.11 -c conda-forge python=3.11
    ~> conda activate pangoline-py3.11
    ~> conda install -c conda-forge pygobject pango Cairo click jinja2 rich pypdfium2 lxml pillow
    ~> pip install --no-deps .

## Usage

### Rendering

PangoLine renders text first into vector PDFs and ALTO facsimiles using some
configurable "physical" dimensions.

    ~> pangoline render doc.txt
    Rendering ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

Various options to direct rendering such as page size, margins, language, and
base direction can be manually set, for example:

    ~> pangoline render -p 216 279 -l en-us -f "Noto Sans 24" doc.txt

### Rasterization

In a second step those vector files can be rasterized into PNGs and the
coordinates in the ALTO files scaled to the selected resolution (per default
300dpi):

    ~> pangoline rasterize doc.0.xml doc.1.xml ...
    Rasterizing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

Rasterized files and their ALTOs can be used as is as ATR training data.

To obtain slightly more realistic input images it is possible to overlay the
rasterized text into images of writing surfaces.

    ~> pangoline rasterize -w ~/background_1.jpg doc.0.xml doc.1.xml ...

Rasterization can be invoked with multiple background images in which case they
will be sampled randomly for each output page.

For larger collections of texts it is advisable to parallelize processing,
especially for rasterization with overlays:

    ~> pangoline --workers 8 render *.txt
    ~> pangoline --workers 8 rasterize *.xml
