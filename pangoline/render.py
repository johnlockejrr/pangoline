#
# Copyright 2025 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
pangoline.render
~~~~~~~~~~~~~~~~
"""
import gi
import math
import uuid
import cairo

gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo

from pathlib import Path
from itertools import count
from typing import Union, Tuple, Literal, Optional, TYPE_CHECKING, List, Dict, Any

from jinja2 import Environment, PackageLoader

if TYPE_CHECKING:
    from os import PathLike


def get_cluster_polygons(line_it: Pango.LayoutIter, line: Pango.LayoutLine, baseline: int, 
                        print_space_offset: int, width: float, right_margin: float, 
                        left_margin: float, top_margin: float, _mm_point: float) -> Dict[str, Any]:
    """
    Get polygon points for a line by aggregating cluster extents.
    Returns dictionary with polygon points and other line metrics.
    """
    line_dir = line.get_resolved_direction()
    cluster_points = []
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    
    # Start cluster iteration
    cluster_it = line_it.copy()
    while True:
        ink_rect, _ = cluster_it.get_cluster_extents()
        Pango.extents_to_pixels(ink_rect)
        
        if ink_rect.width > 0 and ink_rect.height > 0:  # Skip empty clusters
            x, y = ink_rect.x, ink_rect.y
            w, h = ink_rect.width, ink_rect.height
            
            # Adjust coordinates based on text direction
            if line_dir == Pango.Direction.RTL:
                x = (width - right_margin) - x - w
            else:
                x = x + left_margin
                
            # Transform to page coordinates
            y = Pango.units_to_double(baseline - print_space_offset) + top_margin + y
            
            # Update bounds
            min_x = min(min_x, x)
            max_x = max(max_x, x + w)
            min_y = min(min_y, y)
            max_y = max(max_y, y + h)
            
            # Add points for this cluster's rectangle
            cluster_points.extend([
                (x, y),           # top-left
                (x + w, y),       # top-right
                (x + w, y + h),   # bottom-right
                (x, y + h)        # bottom-left
            ])
        
        if not cluster_it.next_cluster():
            break
    
    # Convert to mm
    bl = Pango.units_to_double(baseline - print_space_offset) + top_margin
    
    return {
        'points': [(int(round(x / _mm_point)), int(round(y / _mm_point))) 
                  for x, y in cluster_points],
        'baseline': int(round(bl / _mm_point)),
        'top': int(math.floor(min_y / _mm_point)),
        'bottom': int(math.ceil(max_y / _mm_point)),
        'left': int(math.floor(min_x / _mm_point)),
        'right': int(math.ceil(max_x / _mm_point))
    }


def render_text(text: str,
                output_base_path: Union[str, 'PathLike'],
                paper_size: Tuple[int, int] = (210, 297),
                margins: Tuple[int, int, int, int] = (25, 30, 25, 25),
                font: str = 'Serif Normal 10',
                language: Optional[str] = None,
                base_dir: Optional[Literal['R', 'L']] = None,
                enable_markup: bool = True,
                line_spacing: Optional[float] = None):
    """
    Renders (horizontal) text into a sequence of PDF files and creates parallel
    ALTO files for each page.

    PDF output will be single column, justified text without word breaking.
    Paragraphs will automatically be split once a page is full.

    ALTO file output contains baselines and bounding boxes for each line in the
    text. The unit of measurement in these files is mm.

    Args:
        output_base_path: Base path of the output files. PDF files will be
                          created at `Path.with_suffix(f'.{idx}.pdf')`, ALTO
                          files at `Path.with_suffix(f'.{idx}.xml')`.
        paper_size: `(width, height)` of the PDF output in mm.
        margins: `(top, bottom, left, right)` margins in mm.
        font: Font specification to render the text in.
        language: Set language to enable language-specific rendering. If none
                  is set, the system default will be used.
        base_dir: Sets the base direction of the BiDi algorithm.
        enable_markup: Enable Pango markup processing.
        line_spacing: Additional space between lines in points. None for default.
    """
    output_base_path = Path(output_base_path)

    loader = PackageLoader('pangoline', 'templates')
    tmpl = Environment(loader=loader).get_template('alto.tmpl')

    _mm_point = 72 / 25.4
    width, height = paper_size[0] * _mm_point, paper_size[1] * _mm_point
    top_margin = margins[0] * _mm_point
    bottom_margin = margins[1] * _mm_point
    left_margin = margins[2] * _mm_point
    right_margin = margins[3] * _mm_point

    font_desc = Pango.font_description_from_string(font)
    font_desc.set_features('liga=1, clig=1, dlig=1, hlig=1')
    pango_text_width = Pango.units_from_double(width-(left_margin+right_margin))
    if language:
        pango_lang = Pango.language_from_string(language)
    else:
        pango_lang = Pango.language_get_default()
    pango_dir = {'R': Pango.Direction.RTL,
                 'L': Pango.Direction.LTR,
                 None: None}[base_dir]

    dummy_surface = cairo.PDFSurface(None, 1, 1)
    dummy_context = cairo.Context(dummy_surface)

    # as it is difficult to truncate a text containing RTL runs to split it
    # into pages we render the whole text into a single PangoLayout and then
    # manually place each line on the correct position of a cairo context for
    # each page, translating the vertical coordinates by a print space offset.

    layout = PangoCairo.create_layout(dummy_context)
    layout.set_justify(True)
    layout.set_width(pango_text_width)
    layout.set_wrap(Pango.WrapMode.WORD_CHAR)

    # Set line spacing if specified
    if line_spacing is not None:
        layout.set_spacing(int(line_spacing * Pango.SCALE))

    p_context = layout.get_context()
    p_context.set_language(pango_lang)
    if pango_dir:
        p_context.set_base_dir(pango_dir)
    layout.context_changed()

    layout.set_font_description(font_desc)

    if enable_markup:
        _, attr, text, _ = Pango.parse_markup(text, -1, u'\x00')
        layout.set_text(text)
        layout.set_attributes(attr)
    else:
        layout.set_text(text)

    utf8_text = text.encode('utf-8')

    line_it = layout.get_iter()

    page_print_space = Pango.units_from_double(height-(bottom_margin+top_margin))

    for page_idx in count():
        print_space_offset = page_idx * page_print_space

        pdf_output_path = output_base_path.with_suffix(f'.{page_idx}.pdf')
        alto_output_path = output_base_path.with_suffix(f'.{page_idx}.xml')

        line_splits = []

        pdf_surface = cairo.PDFSurface(pdf_output_path, width, height)
        context = cairo.Context(pdf_surface)
        context.translate(left_margin, top_margin)

        while not line_it.at_last_line():
            line = line_it.get_line_readonly()
            baseline = line_it.get_baseline()
            if baseline > print_space_offset + page_print_space:
                break
            s_idx, e_idx = line.start_index, line.length
            line_text = utf8_text[s_idx:s_idx+e_idx].decode('utf-8')
            if line_text := line_text.strip():
                # Get polygon points and metrics using clusters
                line_data = get_cluster_polygons(
                    line_it, line, baseline, print_space_offset,
                    width, right_margin, left_margin, top_margin, _mm_point
                )
                
                # Add line data to splits
                line_splits.append({
                    'id': str(uuid.uuid4()),
                    'text': line_text,
                    'polygon_points': line_data['points'],
                    'baseline': line_data['baseline'],
                    'top': line_data['top'],
                    'bottom': line_data['bottom'],
                    'left': line_data['left'],
                    'right': line_data['right']
                })
                
                # Render the line
                _, log_extents = line.get_extents()
                if line.get_resolved_direction() == Pango.Direction.RTL:
                    lleft = (width - right_margin) - Pango.units_to_double(log_extents.x + log_extents.width)
                else:
                    lleft = Pango.units_to_double(log_extents.x) + left_margin
                
                context.move_to(lleft - left_margin, Pango.units_to_double(baseline - print_space_offset))
                PangoCairo.show_layout_line(context, line)
                
            line_it.next_line()

        # write ALTO XML file
        with open(alto_output_path, 'w') as fo:
            fo.write(tmpl.render(pdf_path=pdf_output_path.name,
                                 language=pango_lang.to_string(),
                                 base_dir={'L': 'ltr', 'R': 'rtl', None: None}[base_dir],
                                 text_block_id=str(uuid.uuid4()),
                                 page_width=paper_size[0],
                                 page_height=paper_size[1],
                                 lines=line_splits))

        pdf_surface.finish()
        if line_it.at_last_line():
            break
