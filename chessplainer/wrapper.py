from typing import Literal

import chess
import chess.engine
import chess.svg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shap import KernelExplainer
from sklearn.base import BaseEstimator, ClassifierMixin

from chessplainer.plot import board_to_latex_xskak


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
        self.fit_limit_kwargs = fit_limit_kwargs
        if fit_limit_kwargs is None:
            self.fit_limit_kwargs = dict(time=1)

        self.predict_limit_kwargs = predict_limit_kwargs
        if predict_limit_kwargs is None:
            self.predict_limit_kwargs = dict(time=0.1)

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
        self.base_board_ = None
        self.predict_boards_ = None
        self.predict_evals_ = None

    def fit(self, X, y=None):
        # Saves the evaluation of the "base" board
        assert len(X) == 1  # Only one board is expected for fitting.
        self.base_board_ = X[0]
        assert self.base_board_.is_valid()

        self.base_eval_ = self.engine.analyse(
            self.base_board_, chess.engine.Limit(**self.fit_limit_kwargs)
        )["score"]
        self.base_score_ = (
            np.array(
                [
                    [
                        self.base_eval_.wdl(model=self.wdl_model).white().wins,
                        self.base_eval_.wdl(model=self.wdl_model).white().draws,
                        self.base_eval_.wdl(model=self.wdl_model).white().losses,
                    ]
                ]
            )
            / 1000
        )
        self.base_score_wdl_ = self.base_score_.copy()
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
                if self.output_improvement_delta:
                    numerical_evals.append([0, 0, 0])
                else:
                    if self.wdl_model == "lichess":
                        numerical_evals.append([0.5, 0, 0.5])
                    else:
                        numerical_evals.append([0, 1, 0])
            else:
                eval_ = (
                    np.array(
                        [
                            score.wdl(model=self.wdl_model).white().wins,
                            score.wdl(model=self.wdl_model).white().draws,
                            score.wdl(model=self.wdl_model).white().losses,
                        ]
                    )
                    / 1000
                )
                if self.output_improvement_delta:
                    eval_ = eval_ - self.base_score_wdl_[0]
                numerical_evals.append(eval_)

        numerical_evals = np.array(numerical_evals)

        if self.wdl_model == "lichess":
            numerical_evals = numerical_evals[
                :, [0, 2]
            ]  # Only wins and losses, no draws

        self.predict_evals_ = numerical_evals
        return self.predict_evals_

    def predict(self, boards):
        return np.argmax(self.predict_proba(boards), axis=1)

    def _analyze(self, boards):
        infos = list()
        self.predict_boards_ = boards
        for board in boards:
            if board.fen() != self.base_board_.fen():
                if (
                    board.is_valid() or board.status().value == 256
                ):  # bad castling rights
                    info = self.engine.analyse(
                        board, chess.engine.Limit(**self.predict_limit_kwargs)
                    )
                    infos.append(info["score"])
                else:
                    flipped = board.copy()
                    flipped.turn = not board.turn
                    if flipped.is_valid():
                        info = self.engine.analyse(
                            flipped, chess.engine.Limit(**self.predict_limit_kwargs)
                        )
                        infos.append(info["score"])
                    else:
                        infos.append(np.nan)
            else:
                infos.append(self.base_eval_)
        return infos


class ChessExplainer(object):
    def __init__(
        self,
        board,
        wrappend_engine,
        explainer=KernelExplainer,
        explainer_fit_kwargs=None,
        explainer_predict_kwargs=None,
        **kwargs
    ):
        self.board = board
        self.engine = wrappend_engine
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

    def explain(self):
        self.explanation_, self.explainer_ = get_explanation(
            board=self.board,
            engine=self.engine,
            explainer=self.explainer,
            explainer_fit_kwargs=self.explainer_fit_kwargs,
            explainer_predict_kwargs=self.explainer_predict_kwargs,
        )
        self.values_ = self.explanation_.values
        self.df_ = pd.concat([self.df_, pd.DataFrame(self.values_[0])], axis=1)
        return self.explanation_

    def to_svg(self, size=350, cmap="RdBu_r", index=0, **kwargs):
        shap_values = self.values_[0, :, index]

        absmax_val = np.abs(shap_values).max()
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

        board = chess.svg.board(
            board=self.board,
            fill=fill_colors,
            size=size,
            coordinates=True,
            **kwargs,
        )
        return board

    def to_latex(self, cmap="RdBu_r", index=0, local_range=False):
        return board_to_latex_xskak(
            fen=self.board.fen(),
            pieces_idxs=self.pieces.keys(),
            scores=self.values_[0, :, index],
            cmap=cmap,
            absolute=not local_range,
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
    boards = list()
    for z in zs:  # for each combination of presence/absence of pieces
        z = z.ravel()
        masked_piece_map = dict()
        # board = board.copy()
        for square_idx, square in enumerate(z):  # for each occupied square
            if square:  # if the square is kept
                mapped_square = sorted(list(board_piece_map_wo_kings.keys()))[
                    square_idx
                ]
                masked_piece_map[mapped_square] = board_piece_map_wo_kings[
                    mapped_square
                ]  # add the corresponding piece
                masked_piece_map = {**masked_piece_map, **board_piece_map_only_kings}
        board = original_board.copy()
        board.set_piece_map(masked_piece_map)
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
    explanation = explainer_(np.ones((1, n_pieces)), **explainer_predict_kwargs)
    return explanation, explainer_
