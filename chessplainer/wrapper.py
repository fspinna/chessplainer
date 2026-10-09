from typing import Literal

import chess
import chess.engine
import chess.svg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shap import Explanation, KernelExplainer
from sklearn.base import BaseEstimator, ClassifierMixin

from chessplainer.plot import board_svg, board_to_latex_xskak


class EngineWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        path=None,
        fit_limit_kwargs=None,
        predict_limit_kwargs=None,
        output_improvement_delta=False,
        wdl_model: Literal[
            "sf", "sf16.1", "sf16", "sf15.1", "sf15", "sf14", "sf12", "lichess"
        ] = "lichess",
    ):
        # depth limits (unlike time limits) make evaluations reproducible
        self.fit_limit_kwargs = fit_limit_kwargs
        if fit_limit_kwargs is None:
            self.fit_limit_kwargs = dict(depth=20)

        self.predict_limit_kwargs = predict_limit_kwargs
        if predict_limit_kwargs is None:
            self.predict_limit_kwargs = dict(depth=12)

        self.path = path
        self.output_improvement_delta = output_improvement_delta
        if path is None:
            self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        else:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.wdl_model = wdl_model

        self.base_score_ = None
        self.base_score_wdl_ = None
        self.base_eval_ = None
        self.base_predict_eval_ = None
        self.base_predict_wdl_ = None
        self.base_board_ = None
        self.predict_boards_ = None
        self.predict_evals_ = None

    def close(self):
        """Terminates the engine process."""
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _evaluate(self, board, limit_kwargs):
        # a new game object makes python-chess send ucinewgame, which clears the
        # engine hash: each evaluation is independent of the previous ones
        return self.engine.analyse(
            board, chess.engine.Limit(**limit_kwargs), game=object()
        )["score"]

    def _to_wdl(self, score):
        wdl = score.wdl(model=self.wdl_model).white()
        return np.array([wdl.wins, wdl.draws, wdl.losses]) / 1000

    def fit(self, X, y=None):
        # Saves the evaluation of the "base" board
        if len(X) != 1:
            raise ValueError("Only one board is expected for fitting.")
        self.base_board_ = X[0]
        if not self.base_board_.is_valid():
            raise ValueError(f"Invalid board: {self.base_board_.status()!r}")

        # accurate evaluation of the base board (fit limit), for reporting
        self.base_eval_ = self._evaluate(self.base_board_, self.fit_limit_kwargs)
        self.base_score_ = self._to_wdl(self.base_eval_)[None, :]
        self.base_score_wdl_ = self.base_score_.copy()
        # evaluation used inside explanations: same limit as every masked board,
        # so a deeper search is not credited to the pieces
        self.base_predict_eval_ = self._evaluate(
            self.base_board_, self.predict_limit_kwargs
        )
        self.base_predict_wdl_ = self._to_wdl(self.base_predict_eval_)
        if self.wdl_model == "lichess":
            self.base_score_ = self.base_score_[
                0:1, [0, 2]
            ]  # Only wins and losses, no draws

        return self

    def predict_proba(self, boards):
        scores = self._analyze(boards)
        numerical_evals = list()
        for i, score in enumerate(scores):
            # print(i, score)
            if isinstance(score, float) and np.isnan(score):
                # invalid position: treated as a draw
                if self.wdl_model == "lichess":
                    eval_ = np.array([0.5, 0, 0.5])
                else:
                    eval_ = np.array([0, 1, 0])
            else:
                eval_ = self._to_wdl(score)
            if self.output_improvement_delta:
                eval_ = eval_ - self.base_predict_wdl_
            numerical_evals.append(eval_)

        numerical_evals = np.array(numerical_evals)

        if self.wdl_model == "lichess":
            numerical_evals = numerical_evals[
                :, [0, 2]
            ]  # Only wins and losses, no draws

        self.predict_evals_ = numerical_evals
        return self.predict_evals_

    def predict_pawns(self, boards, clip=10):
        """White's evaluation in pawns (centipawns / 100), clipped to [-clip, clip].

        Mates count as +-clip, invalid positions as 0 (a draw). This scale
        saturates less than win probabilities (only beyond +-clip), but it is not
        linear in material either: Stockfish's centipawns are calibrated on win
        chances, and lopsided reduced positions often reach the clip.
        """

        def to_pawns(score):
            if isinstance(score, float) and np.isnan(score):
                return 0.0
            cp = score.white().score(mate_score=100 * clip)
            return float(np.clip(cp / 100, -clip, clip))

        pawns = np.array([to_pawns(score) for score in self._analyze(boards)])
        if self.output_improvement_delta:
            pawns = pawns - to_pawns(self.base_predict_eval_)
        return pawns

    def predict(self, boards):
        return np.argmax(self.predict_proba(boards), axis=1)

    def _analyze(self, boards):
        infos = list()
        self.predict_boards_ = boards
        for board in boards:
            if board.fen() != self.base_board_.fen():
                if board.is_valid():
                    infos.append(self._evaluate(board, self.predict_limit_kwargs))
                else:
                    flipped = board.copy()
                    flipped.turn = not board.turn
                    flipped.ep_square = None
                    if flipped.is_valid():
                        infos.append(self._evaluate(flipped, self.predict_limit_kwargs))
                    else:
                        infos.append(np.nan)
            else:
                infos.append(self.base_predict_eval_)
        return infos


class ChessExplainer(object):
    def __init__(
        self,
        board,
        wrapped_engine=None,
        explainer=KernelExplainer,
        explainer_fit_kwargs=None,
        explainer_predict_kwargs=None,
        **kwargs,
    ):
        if wrapped_engine is None:
            # backward compatibility with the old misspelled argument
            wrapped_engine = kwargs.pop("wrappend_engine", None)
        if wrapped_engine is None:
            raise TypeError("wrapped_engine is required")
        if kwargs:
            raise TypeError(f"Unexpected arguments: {', '.join(kwargs)}")
        self.board = board
        self.engine = wrapped_engine
        self.explainer = explainer
        self.explainer_fit_kwargs = (
            explainer_fit_kwargs if explainer_fit_kwargs is not None else {}
        )
        self.explainer_predict_kwargs = (
            explainer_predict_kwargs if explainer_predict_kwargs is not None else {}
        )

        self.pieces = dict(
            sorted(
                (
                    (sq, pc)
                    for sq, pc in board.piece_map().items()
                    if pc.symbol().lower() != "k"
                ),
                key=lambda kv: kv[0],
            )
        )

        df = pd.DataFrame([self.pieces], index=["piece"]).T
        df["piece"] = df["piece"].apply(lambda x: x.unicode_symbol())
        df = df.reset_index()
        df["square"] = df["index"].apply(lambda x: chess.square_name(x))
        df = df[["square", "piece"]]
        df["feature_name"] = df["piece"] + " " + df["square"]
        self.df_ = df
        self.feature_names = df["feature_name"].tolist()

        self.explanation_ = None
        self.explainer_ = None
        self.values_ = None

        self.game_ = None
        self.exact_computer_ = None
        self.interactions_ = None
        self.interactions_df_ = None
        self.piece_values_ = None

    def explain(self):
        self.explanation_, self.explainer_ = get_explanation(
            board=self.board,
            engine=self.engine,
            explainer=self.explainer,
            explainer_fit_kwargs=self.explainer_fit_kwargs,
            explainer_predict_kwargs=self.explainer_predict_kwargs,
        )
        self.explanation_.feature_names = self.feature_names
        self.values_ = self.explanation_.values
        self.df_ = pd.concat(
            [
                self.df_[["square", "piece", "feature_name"]],
                pd.DataFrame(self.values_[0]),
            ],
            axis=1,
        )
        return self.explanation_

    def to_svg(self, size=350, cmap="RdBu_r", index=0, **kwargs):
        shap_values = self.values_[0, :, index]

        # fall back to a unit range when all values are 0 (TwoSlopeNorm needs vmin < 0)
        absmax_val = np.abs(shap_values).max() or 1.0
        min_val = -absmax_val
        max_val = absmax_val

        norm = matplotlib.colors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
        cmap = plt.get_cmap(cmap)

        board_piece_map = self.board.piece_map()
        board_piece_map_wo_kings = {
            sq: piece
            for sq, piece in board_piece_map.items()
            if piece.symbol() not in ["k", "K"]
        }

        # Map each square to a fill color
        fill_colors = {}
        for i, shap_value in enumerate(shap_values):
            rgba = cmap(norm(shap_value))
            color = matplotlib.colors.to_hex(rgba, keep_alpha=True)
            mapped_square = sorted(board_piece_map_wo_kings.keys())[i]
            fill_colors[mapped_square] = color

        return board_svg(self.board, size=size, fill=fill_colors, **kwargs)

    def to_latex(self, cmap="RdBu_r", index=0, local_range=False):
        return board_to_latex_xskak(
            fen=self.board.fen(),
            pieces_idxs=self.pieces.keys(),
            scores=self.values_[0, :, index],
            cmap=cmap,
            absolute=not local_range,
        )

    def explain_interactions(
        self,
        index="k-SII",
        max_order=2,
        budget=None,
        approximator="auto",
        random_state=None,
        output="score",
        clip=10,
    ):
        """Shapley interactions between pieces, computed with shapiq.

        output: "score" (default) explains White's expected score (win + draw / 2),
        "pawns" White's evaluation in pawns clipped to [-clip, clip], an integer a
        column of the engine's predict_proba. With few pieces (see
        interactions.EXACT_MAX_PLAYERS) values are exact, and other indices of the
        same position and output reuse the engine evaluations.
        """
        from chessplainer import interactions

        if self.game_ is None or self.game_.output != output or self.game_.clip != clip:
            self.game_ = interactions.ChessGame(
                self.board, self.engine, output=output, clip=clip
            )
            self.exact_computer_ = None
        if self.exact_computer_ is not None and budget is None:
            self.interactions_ = interactions.with_baseline(
                self.exact_computer_(index, max_order), self.game_
            )
        else:
            self.interactions_, exact_computer = interactions.explain_interactions(
                self.game_,
                index=index,
                max_order=max_order,
                budget=budget,
                approximator=approximator,
                random_state=random_state,
            )
            self.exact_computer_ = exact_computer or self.exact_computer_
        self.interactions_df_ = interactions.interactions_frame(
            self.interactions_, self.game_
        )
        # each interaction split equally among its pieces: the Shapley value for
        # k-SII, STII and FSII of any order
        self.piece_values_ = pd.Series(
            interactions.piece_values(self.interactions_, self.game_.n_players),
            index=self.game_.player_names,
            name="value",
        ).sort_values(key=abs, ascending=False)
        return self.interactions_

    def interactions_to_svg(
        self, top_k=5, pairs="all", piece=None, size=350, cmap="RdBu_r", **kwargs
    ):
        """Board with the top_k strongest pairwise interactions as lines.

        pairs: "all", "same" (pieces of the same color, solid lines) or
        "opposite" (pieces of opposite colors, dashed lines). piece: a square
        (e.g. "c3") to show only that piece's interactions.
        """
        from chessplainer import interactions

        return interactions.interactions_to_svg(
            self.board,
            self.game_,
            self.interactions_,
            top_k=top_k,
            pairs=pairs,
            piece=piece,
            size=size,
            cmap=cmap,
            **kwargs,
        )

    def piece_interactions(self, piece):
        """DataFrame of one piece's pairs (e.g. piece="c3"), strongest first."""
        from chessplainer import interactions

        return interactions.piece_frame(self.interactions_, self.game_, piece)

    def interactions_viewer(
        self, top_k=5, pairs="all", size=350, cmap="RdBu_r", **kwargs
    ):
        """Interactive board for notebooks: click a piece to see all its
        interactions, an empty square to go back. .save(path) writes an HTML page."""
        from chessplainer import interactions

        return interactions.interactions_html(
            self.board,
            self.game_,
            self.interactions_,
            top_k=top_k,
            pairs=pairs,
            size=size,
            cmap=cmap,
            **kwargs,
        )


def mask_chessboard(zs, original_board):
    # for shap
    # def mask_chessboard that takes a chessboard and a set of arrays that tell the function which piece to remove
    # ex. if there are 4 pieces in the board
    # zs [[0, 1, 1, 0], [1, 1, 0, 1]...]
    # where 0 means remove and 1 keep
    # 1. convert z to position
    # 2. check if valid
    # 3. evaluate position (maybe relative to full board eval to see if pos is improved or not)
    # this function is passed to shap
    # explainer = shap.KernelExplainer(mask_chessboard, data=np.zeros((1, n. of pieces on the full board))))
    board_piece_map = original_board.piece_map()
    board_piece_map_wo_kings = dict()
    board_piece_map_only_kings = dict()
    for key in board_piece_map:
        if board_piece_map[key].symbol() not in ["k", "K"]:
            board_piece_map_wo_kings[key] = board_piece_map[key]
        else:
            board_piece_map_only_kings[key] = board_piece_map[key]
    sorted_squares = sorted(board_piece_map_wo_kings.keys())
    boards = list()
    for z in zs:  # for each combination of presence/absence of pieces
        z = z.ravel()
        # kings are always kept, also when every other piece is removed
        masked_piece_map = dict(board_piece_map_only_kings)
        for square_idx, square in enumerate(z):  # for each occupied square
            if square:  # if the square is kept
                mapped_square = sorted_squares[square_idx]
                masked_piece_map[mapped_square] = board_piece_map_wo_kings[
                    mapped_square
                ]  # add the corresponding piece
        board = original_board.copy()
        board.set_piece_map(masked_piece_map)
        # removing a rook or the pawn that just double-pushed would otherwise
        # leave stale castling rights or en passant square (invalid board)
        board.castling_rights = board.clean_castling_rights()
        if board.status() & chess.STATUS_INVALID_EP_SQUARE:
            board.ep_square = None
        boards.append(board)
    return boards


def get_explanation(
    board,
    engine,
    explainer=KernelExplainer,
    explainer_fit_kwargs=None,
    explainer_predict_kwargs=None,
):
    if explainer_fit_kwargs is None:
        explainer_fit_kwargs = dict()
    if explainer_predict_kwargs is None:
        explainer_predict_kwargs = dict()

    n_pieces = len(board.piece_map()) - 2  # -2 because we don't consider kings

    def f(zs):
        boards = mask_chessboard(zs, board)
        return engine.predict_proba(boards)

    explainer_ = explainer(
        f, np.zeros((1, n_pieces)), **explainer_fit_kwargs
    )  # function, data/masker
    X = np.ones((1, n_pieces))
    if isinstance(explainer_, KernelExplainer):
        # KernelExplainer.__call__ forces l1_reg="num_features(10)", which keeps
        # at most 10 non-zero attributions, and does not accept nsamples:
        # shap_values allows both (e.g. explainer_predict_kwargs=dict(nsamples=...))
        kwargs = {"l1_reg": False, **explainer_predict_kwargs}
        values = explainer_.shap_values(X, **kwargs)
        if isinstance(values, list):
            values = np.stack(values, axis=-1)  # put outputs at the end
        explanation = Explanation(
            values,
            base_values=np.tile(explainer_.expected_value, (X.shape[0], 1)),
            data=X,
        )
    else:
        explanation = explainer_(X, **explainer_predict_kwargs)
    return explanation, explainer_
