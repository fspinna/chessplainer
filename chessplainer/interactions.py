"""Shapley interactions between pieces, computed with shapiq.

shapiq is an optional dependency (Python >= 3.12): pip install "chessplainer[shapiq]"
"""

import html
import uuid
import warnings

import chess
import chess.svg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shapiq
    from shapiq.explainer.configuration import setup_approximator
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Piece interactions require shapiq (Python >= 3.12): "
        'pip install "chessplainer[shapiq]"'
    ) from e

from chessplainer.plot import board_svg
from chessplainer.wrapper import mask_chessboard

# with at most this many pieces (2**14 = 16384 positions), all coalitions are
# evaluated and interactions are computed exactly by default
EXACT_MAX_PLAYERS = 14
# default number of evaluations when approximating. Single-piece values are stable
# with few samples; pairwise values are not: on a 27-piece middlegame, two seeds
# agreed with correlation ~0 at 2**11 evaluations, 0.5 at 2**13, 0.8 at 2**15
DEFAULT_BUDGET = 2**11
DEFAULT_BUDGET_INTERACTIONS = 2**15


def expected_score(probas):
    """White's expected score (win + draw / 2) from the outputs of predict_proba."""
    probas = np.asarray(probas)
    if probas.shape[1] == 2:  # lichess model: [wins, losses], no draws
        return probas[:, 0]
    return probas[:, 0] + 0.5 * probas[:, 1]


class ChessGame(shapiq.Game):
    """Cooperative game whose players are the pieces of a board (kings excluded).

    The value of a coalition is the engine evaluation of the board keeping only
    those pieces (and the kings). Evaluations are cached, since approximators
    can sample the same coalition more than once.

    output: "score" for White's expected score (win + draw / 2), "pawns" for White's
    evaluation in pawns clipped to [-clip, clip], or an integer to use a column of
    the engine's predict_proba. Win probabilities saturate near 0 and 1, which makes
    pieces look redundant whenever one of them already decides the game; the pawn
    scale saturates only beyond +-clip. Neither is linear in material on the
    lopsided reduced positions explanations are made of, so interactions between
    matching pieces (e.g. the two queens) remain on both scales.
    """

    def __init__(self, board, engine, output="score", clip=10, verbose=False):
        self.board = board
        self.engine = engine
        self.output = output
        self.clip = clip
        self.squares = sorted(
            sq for sq, pc in board.piece_map().items() if pc.piece_type != chess.KING
        )
        self.player_names = [
            f"{board.piece_at(sq).unicode_symbol()} {chess.square_name(sq)}"
            for sq in self.squares
        ]
        self.player_colors = [board.piece_at(sq).color for sq in self.squares]
        self._cache = {}
        n_players = len(self.squares)
        empty_value = self._evaluate(np.zeros((1, n_players), dtype=bool))[0]
        super().__init__(
            n_players=n_players,
            normalization_value=empty_value,
            verbose=verbose,
            player_names=self.player_names,
        )

    def _evaluate(self, coalitions):
        coalitions = np.asarray(coalitions, dtype=bool)
        keys = [row.tobytes() for row in coalitions]
        missing = [i for i, key in enumerate(keys) if key not in self._cache]
        if missing:
            boards = mask_chessboard(coalitions[missing].astype(int), self.board)
            if self.output == "pawns":
                values = self.engine.predict_pawns(boards, clip=self.clip)
            elif self.output == "score":
                values = expected_score(self.engine.predict_proba(boards))
            else:
                values = self.engine.predict_proba(boards)[:, self.output]
            for i, value in zip(missing, values):
                self._cache[keys[i]] = float(value)
        return np.array([self._cache[key] for key in keys])

    def value_function(self, coalitions):
        return self._evaluate(coalitions)


def explain_interactions(
    game,
    index="k-SII",
    max_order=2,
    budget=None,
    approximator="auto",
    random_state=None,
):
    """Computes interaction values of a ChessGame.

    With budget=None, values are exact for up to EXACT_MAX_PLAYERS pieces, otherwise
    approximated with DEFAULT_BUDGET evaluations (DEFAULT_BUDGET_INTERACTIONS when
    max_order > 1). Any budget of at least 2**n_players gives exact values.

    Returns the interaction values and the ExactComputer used (None if approximated),
    which computes other indices on the same game without new engine evaluations.
    """
    n = game.n_players
    if budget is None:
        budget = DEFAULT_BUDGET if max_order == 1 else DEFAULT_BUDGET_INTERACTIONS
        if n <= EXACT_MAX_PLAYERS:
            budget = max(budget, 2**n)
    if budget < 2**n and max_order > 1 and budget < DEFAULT_BUDGET_INTERACTIONS:
        warnings.warn(
            f"Approximating interactions between {n} pieces with {budget} evaluations: "
            "pairwise values are likely dominated by sampling noise. "
            f"Use a budget of at least {DEFAULT_BUDGET_INTERACTIONS}.",
            stacklevel=2,
        )
    if budget >= 2**n:
        exact_computer = shapiq.ExactComputer(game)
        values = exact_computer(index, max_order)
    else:
        exact_computer = None
        approximator_ = setup_approximator(
            approximator, index, max_order, n, random_state=random_state
        )
        values = approximator_.approximate(budget, game)
    return with_baseline(values, game), exact_computer


def with_baseline(interaction_values, game):
    """shapiq computes values on the game centered on the kings-only position:
    restores that position's value as baseline, so that baseline + sum = f(x)."""
    interaction_values = interaction_values.__copy__()
    interaction_values.baseline_value = game.normalization_value
    if () in interaction_values.interaction_lookup:
        interaction_values[()] = game.normalization_value
    return interaction_values


def pair_colors(players, player_colors):
    """The colors of a pair of players ("same" or "opposite"), "" otherwise."""
    if len(players) != 2:
        return ""
    i, j = players
    return "same" if player_colors[i] == player_colors[j] else "opposite"


def piece_values(interaction_values, n_players):
    """Total value of each piece: every interaction it belongs to, split equally
    among its pieces (a pair gives half its value to each piece, a triple a third).
    For k-SII, STII and FSII, of any order, this is exactly the piece's Shapley value.

    The single-piece values alone are not Shapley values: they are what remains once
    the interactions are taken out, and can even have the opposite sign (a lone
    knight cannot win, so its value lives in its interactions with other pieces).
    """
    values = np.zeros(n_players)
    for players, value in interaction_values.dict_values.items():
        for i in players:
            values[i] += value / len(players)
    return values


def interactions_frame(interaction_values, game):
    """One row per non-empty interaction, sorted by absolute value.

    The colors column tells pairs of pieces of the same color (cooperation or
    redundancy) from pairs of opposite colors (attack and defense).
    """
    rows = [
        {
            "pieces": " + ".join(game.player_names[i] for i in players),
            "order": len(players),
            "colors": pair_colors(players, game.player_colors),
            "value": value,
        }
        for players, value in interaction_values.dict_values.items()
        if len(players) > 0
    ]
    df = pd.DataFrame(rows, columns=["pieces", "order", "colors", "value"])
    return df.reindex(df["value"].abs().sort_values(ascending=False).index)


def resolve_piece(game, piece):
    """Player index of a piece given as a square ("c3" or chess.C3) or a player
    name ("♘ c3")."""
    square = piece
    if isinstance(piece, str):
        if piece in game.player_names:
            return game.player_names.index(piece)
        try:
            square = chess.parse_square(piece)
        except ValueError:
            square = None
    if square in game.squares:
        return game.squares.index(square)
    raise ValueError(
        f"No explained piece {piece!r}: choose one of {', '.join(game.player_names)}"
    )


def piece_frame(interaction_values, game, piece):
    """The pairs of one piece: partner, colors and value, sorted by absolute value."""
    i = resolve_piece(game, piece)
    rows = [
        {
            "partner": game.player_names[j],
            "colors": pair_colors(players, game.player_colors),
            "value": value,
        }
        for players, value in interaction_values.dict_values.items()
        if len(players) == 2 and i in players
        for j in players
        if j != i
    ]
    df = pd.DataFrame(rows, columns=["partner", "colors", "value"])
    return df.reindex(df["value"].abs().sort_values(ascending=False).index)


def _check_pairs(pairs):
    if pairs not in ("all", "same", "opposite"):
        raise ValueError(f"pairs must be 'all', 'same' or 'opposite', got {pairs!r}")


def _pairs(interaction_values, game, pairs="all", piece=None):
    """(players, value) of the pairwise interactions, strongest first, filtered by
    colors ("all", "same", "opposite") and, if given, by the piece they contain."""
    _check_pairs(pairs)
    return sorted(
        (
            (players, value)
            for players, value in interaction_values.dict_values.items()
            if len(players) == 2
            and pairs in ("all", pair_colors(players, game.player_colors))
            and (piece is None or piece in players)
        ),
        key=lambda kv: -abs(kv[1]),
    )


def _fill(values, cmap):
    """Square -> color, on a scale symmetric around 0."""
    absmax = max((abs(v) for v in values.values()), default=0) or 1.0
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    return {
        square: matplotlib.colors.to_hex(cmap(norm(value)), keep_alpha=True)
        for square, value in values.items()
    }


def _lines(game, pairs, cmap):
    """Lines for (players, value) pairs on their own color scale, dashed for pieces
    of opposite colors: (square, square, color, dashed) tuples for board_svg."""
    absmax = max((abs(value) for _, value in pairs), default=0) or 1.0
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    lines = []
    for (i, j), value in pairs:
        # weakest lines at least half-way along the colormap, so they stay visible
        strength = abs(value) / absmax
        rgba = list(cmap(norm(np.sign(value) * absmax * (0.5 + 0.5 * strength))))
        rgba[3] = 0.5 + 0.4 * strength  # stronger pairs are more opaque
        color = matplotlib.colors.to_hex(rgba, keep_alpha=True)
        dashed = pair_colors((i, j), game.player_colors) == "opposite"
        lines.append((game.squares[i], game.squares[j], color, dashed))
    return lines


SELECTED_COLOR = "#f5c84ccc"


def interactions_to_svg(
    board,
    game,
    interaction_values,
    top_k=5,
    pairs="all",
    piece=None,
    size=350,
    cmap="RdBu_r",
    **kwargs,
):
    """Board with the strongest pairwise interactions as lines between the two
    pieces (red: the pair together is worth more for White than the two pieces
    separately, blue: less; solid: same color, dashed: opposite colors).

    Without piece, squares are colored by each piece's total value (see
    piece_values: its Shapley value for k-SII, STII and FSII) and the top_k
    strongest pairs are drawn. With piece (a square such as "c3"), only that
    piece's pairs: the piece is highlighted, every partner is colored by the value
    of its pair, and lines go to the top_k strongest partners (all if top_k=None).
    Squares and lines use separate color scales.

    pairs: "all", "same" (only pairs of pieces of the same color) or "opposite".
    """
    cmap = plt.get_cmap(cmap)
    if piece is None:
        fill = _fill(
            dict(zip(game.squares, piece_values(interaction_values, game.n_players))),
            cmap,
        )
        selected = _pairs(interaction_values, game, pairs)[:top_k]
        lines = _lines(game, selected, cmap)
    else:
        i = resolve_piece(game, piece)
        selected = _pairs(interaction_values, game, pairs, piece=i)
        fill = _fill(
            {
                game.squares[j]: value
                for players, value in selected
                for j in players
                if j != i
            },
            cmap,
        )
        fill[game.squares[i]] = SELECTED_COLOR
        lines = _lines(game, selected[:top_k], cmap)
    return board_svg(board, size=size, fill=fill, lines=lines, **kwargs)


class InteractionsViewer(str):
    """Interactive board (HTML): click a piece to see all its interactions, click an
    empty square to go back to the overview. Displayed by Jupyter as HTML; save()
    writes a standalone page."""

    def _repr_html_(self):
        return str(self)

    def __repr__(self):
        # short plain-text fallback, so notebooks do not store the HTML twice
        return f"<InteractionsViewer: {len(self) / 1024:.0f} KB>"

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                '<!doctype html><html><head><meta charset="utf-8">'
                "<title>chessplainer interactions</title></head>"
                f'<body style="font-family: sans-serif">{self}</body></html>'
            )


def _table(rows, header):
    cells = "".join(
        f'<tr{f" data-goto={html.escape(goto)!r}" if goto else ""}>'
        f'<td>{html.escape(name)}</td><td class="cpx-num">{value:+.3f}</td></tr>'
        for name, value, goto in rows
    )
    return (
        f"<table><thead><tr><th>{header}</th><th>value</th></tr></thead>"
        f"<tbody>{cells}</tbody></table>"
    )


def interactions_html(
    board,
    game,
    interaction_values,
    top_k=5,
    pairs="all",
    size=350,
    cmap="RdBu_r",
    **kwargs,
):
    """InteractionsViewer for a board: the overview of interactions_to_svg, and the
    view of each piece (piece=...) shown when the piece is clicked. Self-contained:
    no kernel is needed once displayed, and the page can be exported or saved."""
    _check_pairs(pairs)
    uid = f"cpx-{uuid.uuid4().hex[:12]}"
    margin = 15 if kwargs.get("coordinates", True) else 0
    flipped = kwargs.get("orientation", chess.WHITE) == chess.BLACK
    totals = piece_values(interaction_values, game.n_players)

    views = [
        (
            "overview",
            interactions_to_svg(
                board,
                game,
                interaction_values,
                top_k,
                pairs,
                None,
                size,
                cmap,
                **kwargs,
            ),
            "<b>All pieces</b>: squares by each piece's total value, lines for the "
            f"{top_k} strongest pairs. Click a piece to see its interactions."
            + _table(
                [
                    (" + ".join(game.player_names[k] for k in players), value, None)
                    for players, value in _pairs(interaction_values, game, pairs)[:10]
                ],
                "strongest pairs",
            ),
        )
    ]
    for i, square in enumerate(game.squares):
        partners = _pairs(interaction_values, game, pairs, piece=i)
        views.append(
            (
                chess.square_name(square),
                interactions_to_svg(
                    board,
                    game,
                    interaction_values,
                    top_k,
                    pairs,
                    square,
                    size,
                    cmap,
                    **kwargs,
                ),
                f"<b>{html.escape(game.player_names[i])}</b>: total value "
                f"{totals[i]:+.3f}. Squares and lines show its pairs with the other "
                "pieces. Click another piece, or an empty square to go back."
                + _table(
                    [
                        (
                            game.player_names[j],
                            value,
                            chess.square_name(game.squares[j]),
                        )
                        for players, value in partners
                        for j in players
                        if j != i
                    ],
                    "partner",
                ),
            )
        )

    boards = "".join(
        f'<div data-view="{name}"{"" if name == "overview" else " hidden"}>{svg}</div>'
        for name, svg, _ in views
    )
    panels = "".join(
        f'<div data-view="{name}"{"" if name == "overview" else " hidden"}>{text}</div>'
        for name, _, text in views
    )
    return InteractionsViewer(
        f"""<div id="{uid}" class="cpx">
<style>
#{uid} {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }}
#{uid} .cpx-board {{ cursor: pointer; line-height: 0; }}
#{uid} .cpx-side {{ max-width: 340px; font-size: 13px; line-height: 1.4; }}
#{uid} table {{ border-collapse: collapse; margin-top: 8px; width: 100%; }}
#{uid} th, #{uid} td {{ padding: 2px 8px; text-align: left; }}
#{uid} th {{ border-bottom: 1px solid currentColor; }}
#{uid} .cpx-num {{ text-align: right; font-variant-numeric: tabular-nums; }}
#{uid} tr[data-goto] {{ cursor: pointer; }}
#{uid} tr[data-goto]:hover {{ text-decoration: underline; }}
</style>
<div class="cpx-board">{boards}</div>
<div class="cpx-side">{panels}</div>
<script>
(function () {{
  const root = document.getElementById("{uid}");
  const margin = {margin}, flipped = {str(flipped).lower()};
  let current = "overview";
  function show(name) {{
    if (!root.querySelector('.cpx-board [data-view="' + name + '"]')) name = "overview";
    current = name;
    root.querySelectorAll("[data-view]").forEach(function (el) {{
      el.hidden = el.dataset.view !== name;
    }});
  }}
  root.querySelector(".cpx-board").addEventListener("click", function (event) {{
    const svg = root.querySelector('.cpx-board [data-view="' + current + '"] svg');
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const p = point.matrixTransform(svg.getScreenCTM().inverse());
    let col = Math.floor((p.x - margin) / 45), row = Math.floor((p.y - margin) / 45);
    if (col < 0 || col > 7 || row < 0 || row > 7) return;
    const file = flipped ? 7 - col : col, rank = flipped ? row : 7 - row;
    const square = "abcdefgh"[file] + (rank + 1);
    show(square === current ? "overview" : square);
  }});
  root.querySelector(".cpx-side").addEventListener("click", function (event) {{
    const row = event.target.closest("tr[data-goto]");
    if (row) show(row.dataset.goto);
  }});
}})();
</script>
</div>"""
    )
