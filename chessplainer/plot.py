import chess
import chess.svg
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

# board style shared by every SVG chessplainer draws (plain boards included), so
# that boards with and without SHAP values look the same
DEFAULT_COLORS = {
    "square light": "#ffffff",
    "square dark": "#aaaaaa",
    "margin": "#000000",
}


class BoardSvg(chess.svg.SvgWrapper):
    """SVG string shown as an image in notebooks, with a short text representation
    (otherwise notebooks also store the whole SVG as the output's plain text)."""

    def __repr__(self):
        return f"<BoardSvg: {len(self) / 1024:.0f} KB>"


def board_svg(board, size=350, colors=None, lines=(), **kwargs):
    """Draws a board in chessplainer's style.

    Wraps chess.svg.board: colors are merged over DEFAULT_COLORS and coordinates are
    shown unless coordinates=False is passed. lines: (square, square, color, dashed)
    tuples, drawn as lines between the two square centers, on top of the pieces
    (unlike arrows, lines have no direction). Returns an SVG string that notebooks
    display as an image.
    """
    kwargs.setdefault("coordinates", True)
    svg = chess.svg.board(
        board=board, size=size, colors={**DEFAULT_COLORS, **(colors or {})}, **kwargs
    )
    if lines:
        margin = 15 if kwargs["coordinates"] else 0
        flipped = kwargs.get("orientation", chess.WHITE) == chess.BLACK

        def center(square):
            file, rank = chess.square_file(square), chess.square_rank(square)
            col, row = (7 - file, rank) if flipped else (file, 7 - rank)
            return margin + 45 * col + 22.5, margin + 45 * row + 22.5

        elements = []
        for a, b, color, dashed in lines:
            (x1, y1), (x2, y2) = center(a), center(b)
            color, opacity = color[:7], int(color[7:9] or "ff", 16) / 255
            dash = ' stroke-dasharray="12,9"' if dashed else ""
            elements.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                f'stroke-opacity="{opacity:.2f}" stroke-width="8" '
                f'stroke-linecap="round"{dash} class="interaction" />'
            )
        svg = svg.replace("</svg>", "".join(elements) + "</svg>")
    # keep chess.svg's wrapper: a plain str would be shown as text in notebooks
    return BoardSvg(svg)


def rgba_to_hex(rgba):
    r, g, b = [int(255 * x) for x in rgba[:3]]
    return f"{r:02X}{g:02X}{b:02X}"


def board_to_latex_xskak(fen, pieces_idxs, scores, cmap="RdBu", absolute=True):
    # Inputs from your system
    scores_ = {k: v for k, v in zip(pieces_idxs, scores.ravel())}

    # Normalize with symmetric scale
    if absolute:
        absmax = 1
    else:
        absmax = max(abs(min(scores_.values())), abs(max(scores_.values()))) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    cmap = plt.get_cmap(cmap)

    # Build LaTeX color definitions and usage lines
    color_defs = []
    highlight_lines = []

    for square, score in scores_.items():
        square_name = chess.square_name(square)
        rgba = cmap(norm(score))
        hex_code = rgba_to_hex(rgba)
        color_name = f"shap{square_name}"

        color_defs.append(f"\\definecolor{{{color_name}}}{{HTML}}{{{hex_code}}}")
        highlight_lines.append(
            f"  color={color_name},\n  colorbackfield={{{square_name}}},"
        )

    highlight_block = "\n".join(highlight_lines).rstrip(",")
    color_def_block = "\n".join(color_defs)

    # Final LaTeX output
    latex_code = f"""
    {color_def_block}

    \\newgame
    \\chessboard[
      setfen={fen},
      boardfontencoding=LSBC1,
      pgfstyle=color,
      opacity=0.7,
    {highlight_block}
    ]
    """

    return latex_code
