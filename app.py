"""
Steel Section Properties Calculator
A Dash app for computing cross-section properties using the sectionproperties library.
Supports parametric (custom) and UK standard catalogue sections (Tata Steel / Corus).
"""

import traceback

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html, no_update

from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

import sectionproperties.pre.library as sp_lib
from sectionproperties.pre.geometry import Geometry
from uk_catalogue import (
    ALL_CATALOGUES,
    ANGLE_SERIES,
    CHANNEL_SERIES,
    I_SECTION_SERIES,
)
from sectionproperties.analysis import Section

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Steel Section Calculator",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

SECTION_TYPES_CUSTOM = [
    "I-Section (symmetric)",
    "Mono I-Section (asymmetric)",
    "Channel (UPN style)",
    "Tee Section",
    "Angle Section",
    "Rectangular Hollow Section (RHS)",
    "Circular Hollow Section (CHS)",
    "Built-Up Girder (Historical)",
]

IN_TO_MM = 25.4

CATALOGUE_SERIES = list(ALL_CATALOGUES.keys())


def make_param_input(id_prefix, label, value, step=1.0, min_val=0.1):
    """Return a labeled number input row."""
    return dbc.Row(
        [
            dbc.Col(dbc.Label(label, className="fw-semibold"), width=6),
            dbc.Col(
                dbc.Input(
                    id=id_prefix,
                    type="number",
                    value=value,
                    step=step,
                    min=min_val,
                    debounce=True,
                    className="form-control-sm",
                ),
                width=6,
            ),
        ],
        className="mb-2 align-items-center",
    )


def section_outline_trace(geometry):
    """Return a Plotly Scatter trace for the section outline."""
    if geometry.geom.geom_type == "Polygon":
        coords = list(geometry.geom.exterior.coords)
        x, y = zip(*coords)
        traces = [
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                fill="toself",
                fillcolor="rgba(70, 130, 180, 0.25)",
                line={"color": "steelblue", "width": 2},
                name="Cross-section",
            )
        ]
        # Add any holes
        for interior in geometry.geom.interiors:
            ix, iy = zip(*list(interior.coords))
            traces.append(
                go.Scatter(
                    x=list(ix),
                    y=list(iy),
                    mode="lines",
                    fill="toself",
                    fillcolor="white",
                    line={"color": "steelblue", "width": 1.5},
                    showlegend=False,
                )
            )
        return traces
    return []


def centroid_trace(cx, cy):
    return go.Scatter(
        x=[cx],
        y=[cy],
        mode="markers",
        marker={"symbol": "cross", "size": 12, "color": "red", "line": {"width": 2}},
        name="Centroid",
    )


def shear_centre_trace(xsc, ysc):
    return go.Scatter(
        x=[xsc],
        y=[ysc],
        mode="markers",
        marker={"symbol": "diamond", "size": 10, "color": "green", "line": {"width": 2}},
        name="Shear centre",
    )


def mesh_trace(geometry):
    """Return a Plotly Scatter trace showing finite-element triangles."""
    mesh = geometry.mesh
    vertices = mesh["vertices"]
    triangles = mesh["triangles"]

    x_coords = []
    y_coords = []
    for tri in triangles:
        # tri6 elements: first 3 indices are corner nodes
        idx = [tri[0], tri[1], tri[2], tri[0]]
        for i in idx:
            x_coords.append(vertices[i, 0])
            y_coords.append(vertices[i, 1])
        x_coords.append(None)
        y_coords.append(None)

    return go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="lines",
        line={"color": "rgba(100, 100, 100, 0.3)", "width": 0.5},
        name="Finite elements",
        hoverinfo="skip",
    )


def build_section_figure(geometry, cx, cy, xsc, ysc):
    traces = section_outline_trace(geometry)
    traces.append(mesh_trace(geometry))
    traces.append(centroid_trace(cx, cy))
    traces.append(shear_centre_trace(xsc, ysc))

    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis={"scaleanchor": "y", "title": "y (mm)", "showgrid": True, "gridcolor": "#e0e0e0"},
        yaxis={"title": "z (mm)", "showgrid": True, "gridcolor": "#e0e0e0"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "center", "x": 0.5},
        height=420,
    )
    return fig


def fmt(value, decimals=2):
    """Format a numpy/float value nicely."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def props_table(props: dict):
    """Build a Bootstrap table from a dict of {label: value}."""
    rows = [
        html.Tr([html.Td(k, className="fw-semibold text-nowrap pe-3"), html.Td(v)])
        for k, v in props.items()
    ]
    return dbc.Table(
        [html.Tbody(rows)],
        bordered=False,
        striped=True,
        hover=True,
        size="sm",
        className="mb-0",
    )


def build_builtup_girder(dw, tw, bf_top, tf_top, bf_bot, tf_bot,
                         ang_v, ang_h, ang_t, r_root=0, r_toe=0, n_r=8):
    """Build a historical riveted plate girder cross-section.

    All inputs in mm.  Returns a sectionproperties Geometry.

    The web plate runs the full depth (dw) from bottom flange to top flange.
    Angles sit within the web depth at each corner, with one leg against the
    web face and the other leg against the flange inner face.

    r_root: root radius at the inside corner of each angle.
    r_toe:  toe radius at each leg tip (inner-face side).
    n_r:    number of points per quarter-arc.

    Layout (bottom to top):
      bottom flange plate | web plate (with angles at corners) | top flange plate
    Total depth = tf_bot + dw + tf_top.
    """
    hw = tw / 2  # half web thickness
    pi = np.pi

    def _arc(cx, cy, r, a0, a1, n=n_r):
        """Return n points along a circular arc from angle a0 to a1."""
        angles = np.linspace(a0, a1, max(n, 2))
        return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

    # --- Web plate (full depth between flanges) ---
    web = box(-hw, 0, hw, dw)

    # --- Flange plates (directly against web ends) ---
    bot_fl = box(-bf_bot / 2, -tf_bot, bf_bot / 2, 0)
    top_fl = box(-bf_top / 2, dw, bf_top / 2, dw + tf_top)

    # --- Build bottom-left angle with root & toe radii ---
    #
    #        E ─── F          F is against the web face
    #        │     │
    #   r_t  ╮     │  vert leg (against web)
    #        │     │
    #   C────╯     │  r_root at inside corner
    #   │    r_r   │
    #   ╰─C        │  r_t at horiz leg tip
    #   │          │
    #   B ──────── A          A is at flange/web junction
    #
    # A = (-hw, 0)                    horiz leg against flange face
    # B = (-(hw+ang_h), 0)            outer corner of horiz leg tip
    # C = (-(hw+ang_h), ang_t)        toe (inner face of horiz leg tip)
    # D = (-(hw+ang_t), ang_t)        root (inside corner)
    # E = (-(hw+ang_t), ang_t+ang_v)  toe (inner face of vert leg tip)
    # F = (-hw, ang_t+ang_v)          against web face

    pts = [(-hw, 0), (-(hw + ang_h), 0)]

    if r_toe > 0:
        # Toe arc at C: from vertical edge to horizontal edge
        pts += _arc(-(hw + ang_h) + r_toe, ang_t - r_toe,
                    r_toe, pi, pi / 2)
    else:
        pts.append((-(hw + ang_h), ang_t))

    if r_root > 0:
        # Root arc at D: from horizontal edge to vertical edge
        pts += _arc(-(hw + ang_t) + r_root, ang_t + r_root,
                    r_root, 3 * pi / 2, pi)
    else:
        pts.append((-(hw + ang_t), ang_t))

    if r_toe > 0:
        # Toe arc at E: from vertical edge to horizontal edge
        pts += _arc(-(hw + ang_t) + r_toe, ang_t + ang_v - r_toe,
                    r_toe, pi, pi / 2)
    else:
        pts.append((-(hw + ang_t), ang_t + ang_v))

    pts.append((-hw, ang_t + ang_v))

    ang_bl = Polygon(pts)

    # --- Derive other three angles by mirroring ---
    ang_br = affinity.scale(ang_bl, xfact=-1, origin=(0, 0))
    ang_tl = affinity.scale(ang_bl, yfact=-1, origin=(0, dw / 2))
    ang_tr = affinity.scale(ang_bl, xfact=-1, yfact=-1, origin=(0, dw / 2))

    combined = unary_union([web, ang_bl, ang_br, ang_tl, ang_tr, bot_fl, top_fl])
    return Geometry(combined)


def compute_section(geometry):
    """Run full section analysis; return Section object."""
    geometry.create_mesh(mesh_sizes=[min(10, geometry.geom.length / 40)])
    sec = Section(geometry=geometry)
    sec.calculate_geometric_properties()
    sec.calculate_plastic_properties()
    sec.calculate_warping_properties()
    return sec


def extract_props(sec: Section) -> dict:
    """Pull key section properties into a labelled dict."""
    area = sec.get_area()
    cx, cy = sec.get_c()
    ixx_c, iyy_c, _ = sec.get_ic()
    zxx_plus, zxx_minus, zyy_plus, zyy_minus = sec.get_z()
    zpxx_plus, zpxx_minus, zpyy_plus, zpyy_minus = sec.get_zp()
    sfxx, _, sfyy, _ = sec.get_sf()
    rx_c, ry_c = sec.get_rc()
    j = sec.get_j()
    gamma = sec.get_gamma()
    asxx, asyy = sec.get_as()
    xsc, ysc = sec.get_sc()
    phi = sec.get_phi()

    return {
        "Area  A  [mm²]": fmt(area, 1),
        "Centroid  cx  [mm]": fmt(cx, 3),
        "Centroid  cy  [mm]": fmt(cy, 3),
        "Ixx (centroid)  [mm⁴]": fmt(ixx_c, 1),
        "Iyy (centroid)  [mm⁴]": fmt(iyy_c, 1),
        "Zxx,top  [mm³]": fmt(zxx_plus, 1),
        "Zxx,bot  [mm³]": fmt(zxx_minus, 1),
        "Zyy,right  [mm³]": fmt(zyy_plus, 1),
        "Zyy,left  [mm³]": fmt(zyy_minus, 1),
        "Zp,xx,top  [mm³]  (plastic)": fmt(zpxx_plus, 1),
        "Zp,xx,bot  [mm³]  (plastic)": fmt(zpxx_minus, 1),
        "Zp,yy,right  [mm³]  (plastic)": fmt(zpyy_plus, 1),
        "Shape factor  SF,xx": fmt(sfxx, 4),
        "Shape factor  SF,yy": fmt(sfyy, 4),
        "Radius of gyration  rx  [mm]": fmt(rx_c, 3),
        "Radius of gyration  ry  [mm]": fmt(ry_c, 3),
        "Torsion constant  J  [mm⁴]": fmt(j, 1),
        "Warping constant  Iw  [mm⁶]": fmt(gamma, 1),
        "Shear area  Ax  [mm²]": fmt(asxx, 1),
        "Shear area  Ay  [mm²]": fmt(asyy, 1),
        "Shear centre  xsc  [mm]": fmt(xsc, 3),
        "Shear centre  ysc  [mm]": fmt(ysc, 3),
        "Principal axis angle  φ  [°]": fmt(phi, 4),
    }


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def custom_params_layout():
    return html.Div(
        id="custom-params-container",
        children=[
            html.H6("Section Parameters (mm)", className="mt-3 mb-2 text-secondary"),
            # I-section params (shown by default)
            html.Div(
                id="params-i-section",
                children=[
                    make_param_input("inp-d",   "d — Total depth",         200),
                    make_param_input("inp-b",   "b — Flange width",        100),
                    make_param_input("inp-tf",  "tf — Flange thickness",   10),
                    make_param_input("inp-tw",  "tw — Web thickness",       6),
                    make_param_input("inp-r",   "r — Root radius",         12),
                    make_param_input("inp-nr",  "n_r — Fillet points",     16, step=1, min_val=4),
                ],
            ),
            # Mono I-section extras
            html.Div(
                id="params-mono-i",
                style={"display": "none"},
                children=[
                    make_param_input("inp-d-mono",    "d — Total depth",           300),
                    make_param_input("inp-bt",        "b_t — Top flange width",    100),
                    make_param_input("inp-bb",        "b_b — Bottom flange width", 200),
                    make_param_input("inp-tft",       "tf_t — Top flange thick.",    8),
                    make_param_input("inp-tfb",       "tf_b — Bot. flange thick.",  16),
                    make_param_input("inp-tw-mono",   "tw — Web thickness",          8),
                    make_param_input("inp-r-mono",    "r — Root radius",            12),
                    make_param_input("inp-nr-mono",   "n_r — Fillet points",        16, step=1, min_val=4),
                ],
            ),
            # Channel section
            html.Div(
                id="params-channel",
                style={"display": "none"},
                children=[
                    make_param_input("inp-d-ch",   "d — Depth",              200),
                    make_param_input("inp-b-ch",   "b — Flange width",        75),
                    make_param_input("inp-tf-ch",  "tf — Flange thickness",   11),
                    make_param_input("inp-tw-ch",  "tw — Web thickness",       8),
                    make_param_input("inp-r-ch",   "r — Root radius",         12),
                    make_param_input("inp-nr-ch",  "n_r — Fillet points",     16, step=1, min_val=4),
                ],
            ),
            # Tee section
            html.Div(
                id="params-tee",
                style={"display": "none"},
                children=[
                    make_param_input("inp-d-tee",   "d — Depth",             150),
                    make_param_input("inp-b-tee",   "b — Flange width",      150),
                    make_param_input("inp-tf-tee",  "tf — Flange thickness",  10),
                    make_param_input("inp-tw-tee",  "tw — Web thickness",      6),
                    make_param_input("inp-r-tee",   "r — Root radius",         8),
                    make_param_input("inp-nr-tee",  "n_r — Fillet points",    16, step=1, min_val=4),
                ],
            ),
            # Angle section
            html.Div(
                id="params-angle",
                style={"display": "none"},
                children=[
                    make_param_input("inp-d-ang",   "d — Vertical leg",      100),
                    make_param_input("inp-b-ang",   "b — Horizontal leg",    100),
                    make_param_input("inp-tf-ang",  "tf — Flange thickness",  10),
                    make_param_input("inp-tw-ang",  "tw — Web thickness",     10),
                    make_param_input("inp-r-ang",   "r — Root radius",        12),
                    make_param_input("inp-nr-ang",  "n_r — Fillet points",    16, step=1, min_val=4),
                ],
            ),
            # RHS
            html.Div(
                id="params-rhs",
                style={"display": "none"},
                children=[
                    make_param_input("inp-d-rhs",    "d — Depth",             200),
                    make_param_input("inp-b-rhs",    "b — Width",             100),
                    make_param_input("inp-t-rhs",    "t — Wall thickness",      8),
                    make_param_input("inp-rout-rhs", "r_out — Outer radius",   12),
                    make_param_input("inp-nr-rhs",   "n_r — Corner points",    16, step=1, min_val=4),
                ],
            ),
            # CHS
            html.Div(
                id="params-chs",
                style={"display": "none"},
                children=[
                    make_param_input("inp-d-chs",  "d — Outer diameter",     219.1),
                    make_param_input("inp-t-chs",  "t — Wall thickness",       8.0),
                    make_param_input("inp-n-chs",  "n — Points on circle",     64, step=1, min_val=16),
                ],
            ),
            # Built-Up Girder (Historical)
            html.Div(
                id="params-builtup",
                style={"display": "none"},
                children=[
                    html.H6("Built-Up Girder (inches)", className="mt-3 mb-2 text-secondary"),
                    html.P("Web plate", className="fw-semibold mb-1 small"),
                    make_param_input("inp-dw-bu",  "Depth (in.)",           36.0,  step=0.25),
                    make_param_input("inp-tw-bu",  "Thickness (in.)",        0.375, step=0.0625),
                    html.P("Top flange plate", className="fw-semibold mb-1 mt-2 small"),
                    make_param_input("inp-bft-bu", "Width (in.)",           14.0,  step=0.25),
                    make_param_input("inp-tft-bu", "Thickness (in.)",        0.5,   step=0.0625),
                    html.P("Bottom flange plate", className="fw-semibold mb-1 mt-2 small"),
                    make_param_input("inp-bfb-bu", "Width (in.)",           14.0,  step=0.25),
                    make_param_input("inp-tfb-bu", "Thickness (in.)",        0.75,  step=0.0625),
                    html.P("Connecting angles (x4)", className="fw-semibold mb-1 mt-2 small"),
                    make_param_input("inp-av-bu",  "Vert. leg (in.)",        4.0,   step=0.125),
                    make_param_input("inp-ah-bu",  "Horiz. leg (in.)",       3.5,   step=0.125),
                    make_param_input("inp-at-bu",  "Thickness (in.)",        0.375, step=0.0625),
                    make_param_input("inp-rr-bu",  "Root radius (in.)",      0.375, step=0.0625, min_val=0),
                    make_param_input("inp-rt-bu",  "Toe radius (in.)",       0.1875, step=0.0625, min_val=0),
                ],
            ),
        ],
    )


def catalogue_params_layout():
    series_options = [{"label": s, "value": s} for s in CATALOGUE_SERIES]
    first_series = CATALOGUE_SERIES[0]
    first_sections = list(ALL_CATALOGUES[first_series].keys())

    return html.Div(
        id="catalogue-params-container",
        style={"display": "none"},
        children=[
            html.H6("Catalogue Selection", className="mt-3 mb-2 text-secondary"),
            dbc.Row(
                [
                    dbc.Col(dbc.Label("Series", className="fw-semibold"), width=4),
                    dbc.Col(
                        dbc.Select(
                            id="cat-series",
                            options=series_options,
                            value=first_series,
                        ),
                        width=8,
                    ),
                ],
                className="mb-2 align-items-center",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Label("Section", className="fw-semibold"), width=4),
                    dbc.Col(
                        dbc.Select(
                            id="cat-section",
                            options=[{"label": s, "value": s} for s in first_sections],
                            value=first_sections[0],
                        ),
                        width=8,
                    ),
                ],
                className="mb-2 align-items-center",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

sidebar = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Section Input", className="card-title mb-3"),
            # Mode selector
            dbc.Row(
                [
                    dbc.Col(dbc.Label("Mode", className="fw-semibold"), width=4),
                    dbc.Col(
                        dbc.RadioItems(
                            id="mode-radio",
                            options=[
                                {"label": "Custom", "value": "custom"},
                                {"label": "UK Catalogue", "value": "catalogue"},
                            ],
                            value="custom",
                            inline=True,
                        ),
                        width=8,
                    ),
                ],
                className="mb-3 align-items-center",
            ),
            # Custom section type dropdown
            html.Div(
                id="custom-type-container",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(dbc.Label("Type", className="fw-semibold"), width=4),
                            dbc.Col(
                                dbc.Select(
                                    id="section-type",
                                    options=[{"label": t, "value": t} for t in SECTION_TYPES_CUSTOM],
                                    value=SECTION_TYPES_CUSTOM[0],
                                ),
                                width=8,
                            ),
                        ],
                        className="mb-2 align-items-center",
                    ),
                    custom_params_layout(),
                ],
            ),
            # Catalogue selector
            catalogue_params_layout(),
            # Calculate button
            dbc.Button(
                "Calculate",
                id="calc-btn",
                color="primary",
                className="w-100 mt-3",
                n_clicks=0,
            ),
            html.Div(id="error-msg", className="text-danger small mt-2"),
        ]
    ),
    className="h-100",
)

results_panel = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Results", className="card-title mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6("Cross-section", className="text-secondary"),
                            dcc.Graph(
                                id="section-plot",
                                config={"displayModeBar": False},
                                figure=go.Figure(
                                    layout={
                                        "height": 380,
                                        "plot_bgcolor": "white",
                                        "paper_bgcolor": "white",
                                        "xaxis": {"showgrid": True},
                                        "yaxis": {"showgrid": True},
                                        "annotations": [
                                            {
                                                "text": "Press <b>Calculate</b> to display the section",
                                                "xref": "paper", "yref": "paper",
                                                "x": 0.5, "y": 0.5,
                                                "showarrow": False,
                                                "font": {"size": 14, "color": "grey"},
                                            }
                                        ],
                                    }
                                ),
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(
                        [
                            html.H6("Section Properties", className="text-secondary"),
                            html.Div(
                                id="props-table",
                                children=html.P(
                                    "No results yet.",
                                    className="text-muted small",
                                ),
                            ),
                        ],
                        md=7,
                    ),
                ]
            ),
        ]
    )
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H3(
                    "Steel Section Properties Calculator",
                    className="py-3 mb-0",
                ),
            )
        ),
        html.Hr(className="mt-0 mb-3"),
        dbc.Row(
            [
                dbc.Col(sidebar, md=4, className="mb-3"),
                dbc.Col(results_panel, md=8, className="mb-3"),
            ],
            align="start",
        ),
        html.Footer(
            "Powered by sectionproperties · Built with Plotly Dash",
            className="text-center text-muted small pb-3",
        ),
    ],
    fluid=True,
    className="px-4",
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("custom-type-container", "style"),
    Output("catalogue-params-container", "style"),
    Input("mode-radio", "value"),
)
def toggle_mode(mode):
    if mode == "custom":
        return {}, {"display": "none"}
    return {"display": "none"}, {}


@callback(
    Output("params-i-section",  "style"),
    Output("params-mono-i",     "style"),
    Output("params-channel",    "style"),
    Output("params-tee",        "style"),
    Output("params-angle",      "style"),
    Output("params-rhs",        "style"),
    Output("params-chs",        "style"),
    Output("params-builtup",    "style"),
    Input("section-type", "value"),
)
def toggle_param_panels(section_type):
    show = {}
    hide = {"display": "none"}
    mapping = {
        "I-Section (symmetric)":          (show, hide, hide, hide, hide, hide, hide, hide),
        "Mono I-Section (asymmetric)":     (hide, show, hide, hide, hide, hide, hide, hide),
        "Channel (UPN style)":             (hide, hide, show, hide, hide, hide, hide, hide),
        "Tee Section":                     (hide, hide, hide, show, hide, hide, hide, hide),
        "Angle Section":                   (hide, hide, hide, hide, show, hide, hide, hide),
        "Rectangular Hollow Section (RHS)":(hide, hide, hide, hide, hide, show, hide, hide),
        "Circular Hollow Section (CHS)":   (hide, hide, hide, hide, hide, hide, show, hide),
        "Built-Up Girder (Historical)":    (hide, hide, hide, hide, hide, hide, hide, show),
    }
    return mapping.get(section_type, (show, hide, hide, hide, hide, hide, hide, hide))


@callback(
    Output("cat-section", "options"),
    Output("cat-section", "value"),
    Input("cat-series", "value"),
)
def update_catalogue_sections(series):
    sections = list(ALL_CATALOGUES[series].keys())
    options = [{"label": s, "value": s} for s in sections]
    return options, sections[0]


@callback(
    Output("section-plot", "figure"),
    Output("props-table",  "children"),
    Output("error-msg",    "children"),
    Input("calc-btn", "n_clicks"),
    State("mode-radio",    "value"),
    State("section-type",  "value"),
    # Custom I-section
    State("inp-d",  "value"),
    State("inp-b",  "value"),
    State("inp-tf", "value"),
    State("inp-tw", "value"),
    State("inp-r",  "value"),
    State("inp-nr", "value"),
    # Custom Mono-I
    State("inp-d-mono",  "value"),
    State("inp-bt",      "value"),
    State("inp-bb",      "value"),
    State("inp-tft",     "value"),
    State("inp-tfb",     "value"),
    State("inp-tw-mono", "value"),
    State("inp-r-mono",  "value"),
    State("inp-nr-mono", "value"),
    # Channel
    State("inp-d-ch",  "value"),
    State("inp-b-ch",  "value"),
    State("inp-tf-ch", "value"),
    State("inp-tw-ch", "value"),
    State("inp-r-ch",  "value"),
    State("inp-nr-ch", "value"),
    # Tee
    State("inp-d-tee",  "value"),
    State("inp-b-tee",  "value"),
    State("inp-tf-tee", "value"),
    State("inp-tw-tee", "value"),
    State("inp-r-tee",  "value"),
    State("inp-nr-tee", "value"),
    # Angle
    State("inp-d-ang",  "value"),
    State("inp-b-ang",  "value"),
    State("inp-tf-ang", "value"),
    State("inp-tw-ang", "value"),
    State("inp-r-ang",  "value"),
    State("inp-nr-ang", "value"),
    # RHS
    State("inp-d-rhs",    "value"),
    State("inp-b-rhs",    "value"),
    State("inp-t-rhs",    "value"),
    State("inp-rout-rhs", "value"),
    State("inp-nr-rhs",   "value"),
    # CHS
    State("inp-d-chs",  "value"),
    State("inp-t-chs",  "value"),
    State("inp-n-chs",  "value"),
    # Built-Up Girder
    State("inp-dw-bu",  "value"),
    State("inp-tw-bu",  "value"),
    State("inp-bft-bu", "value"),
    State("inp-tft-bu", "value"),
    State("inp-bfb-bu", "value"),
    State("inp-tfb-bu", "value"),
    State("inp-av-bu",  "value"),
    State("inp-ah-bu",  "value"),
    State("inp-at-bu",  "value"),
    State("inp-rr-bu",  "value"),
    State("inp-rt-bu",  "value"),
    # Catalogue
    State("cat-series",  "value"),
    State("cat-section", "value"),
    prevent_initial_call=True,
)
def calculate(
    n_clicks,
    mode,
    section_type,
    # I-section
    d, b, tf, tw, r, nr,
    # Mono-I
    d_mono, bt, bb, tft, tfb, tw_mono, r_mono, nr_mono,
    # Channel
    d_ch, b_ch, tf_ch, tw_ch, r_ch, nr_ch,
    # Tee
    d_tee, b_tee, tf_tee, tw_tee, r_tee, nr_tee,
    # Angle
    d_ang, b_ang, tf_ang, tw_ang, r_ang, nr_ang,
    # RHS
    d_rhs, b_rhs, t_rhs, rout_rhs, nr_rhs,
    # CHS
    d_chs, t_chs, n_chs,
    # Built-Up Girder
    dw_bu, tw_bu, bft_bu, tft_bu, bfb_bu, tfb_bu, av_bu, ah_bu, at_bu, rr_bu, rt_bu,
    # Catalogue
    cat_series, cat_section_name,
):
    empty_fig = no_update
    try:
        geometry = None

        if mode == "custom":
            st = section_type
            if st == "I-Section (symmetric)":
                geometry = sp_lib.i_section(d=d, b=b, t_f=tf, t_w=tw, r=r, n_r=int(nr))

            elif st == "Mono I-Section (asymmetric)":
                geometry = sp_lib.mono_i_section(
                    d=d_mono, b_t=bt, b_b=bb,
                    t_ft=tft, t_fb=tfb,
                    t_w=tw_mono, r=r_mono, n_r=int(nr_mono),
                )

            elif st == "Channel (UPN style)":
                geometry = sp_lib.channel_section(
                    d=d_ch, b=b_ch, t_f=tf_ch, t_w=tw_ch, r=r_ch, n_r=int(nr_ch)
                )

            elif st == "Tee Section":
                geometry = sp_lib.tee_section(
                    d=d_tee, b=b_tee, t_f=tf_tee, t_w=tw_tee, r=r_tee, n_r=int(nr_tee)
                )

            elif st == "Angle Section":
                geometry = sp_lib.angle_section(
                    d=d_ang, b=b_ang, t=tf_ang, r_r=r_ang, r_t=r_ang / 2, n_r=int(nr_ang)
                )

            elif st == "Rectangular Hollow Section (RHS)":
                geometry = sp_lib.rectangular_hollow_section(
                    d=d_rhs, b=b_rhs, t=t_rhs, r_out=rout_rhs, n_r=int(nr_rhs)
                )

            elif st == "Circular Hollow Section (CHS)":
                geometry = sp_lib.circular_hollow_section(
                    d=d_chs, t=t_chs, n=int(n_chs)
                )

            elif st == "Built-Up Girder (Historical)":
                def _in(v, default=0):
                    return float(v) * IN_TO_MM if v is not None else default * IN_TO_MM

                geometry = build_builtup_girder(
                    dw=_in(dw_bu, 36),
                    tw=_in(tw_bu, 0.375),
                    bf_top=_in(bft_bu, 14),
                    tf_top=_in(tft_bu, 0.5),
                    bf_bot=_in(bfb_bu, 14),
                    tf_bot=_in(tfb_bu, 0.75),
                    ang_v=_in(av_bu, 4),
                    ang_h=_in(ah_bu, 3.5),
                    ang_t=_in(at_bu, 0.375),
                    r_root=_in(rr_bu, 0.375),
                    r_toe=_in(rt_bu, 0.1875),
                )

        else:  # catalogue
            dims = ALL_CATALOGUES[cat_series][cat_section_name]
            h  = dims["h"]
            bw = dims["b"]
            tf_c = dims["tf"]
            tw_c = dims["tw"]
            rc   = dims["r"]

            if cat_series in I_SECTION_SERIES:
                geometry = sp_lib.i_section(
                    d=h, b=bw, t_f=tf_c, t_w=tw_c, r=rc, n_r=16
                )
            elif cat_series in CHANNEL_SERIES:
                geometry = sp_lib.channel_section(
                    d=h, b=bw, t_f=tf_c, t_w=tw_c, r=rc, n_r=16
                )
            elif cat_series in ANGLE_SERIES:
                t = dims["t"]
                geometry = sp_lib.angle_section(
                    d=h, b=bw, t=t, r_r=rc, r_t=rc / 2, n_r=16
                )

        if geometry is None:
            return empty_fig, no_update, "Unknown section type."

        sec = compute_section(geometry)
        props = extract_props(sec)

        cx, cy = sec.get_c()
        xsc, ysc = sec.get_sc()

        fig = build_section_figure(geometry, cx, cy, xsc, ysc)
        table = props_table(props)

        return fig, table, ""

    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        return empty_fig, no_update, f"Error: {exc}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
