import chess
from matplotlib import colors as mcolors, pyplot as plt

from chessplainer.constants import MATE_VALUE


def rgba_to_hex(rgba):
    r, g, b = [int(255 * x) for x in rgba[:3]]
    return f"{r:02X}{g:02X}{b:02X}"


def board_to_latex_xskak(fen, pieces_idxs, scores, cmap="RdBu", absolute=True):
    # Inputs from your system
    scores_ = {k: v for k, v in zip(pieces_idxs, scores.ravel())}

    # Normalize with symmetric scale
    if absolute:
        absmax = MATE_VALUE
    else:
        absmax = max(abs(min(scores_.values())), abs(max(scores_.values())))
    norm = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    cmap = plt.cm.get_cmap(cmap)

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

    highlight_block = "\n".join(highlight_lines).rstrip(',')
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
