import chess
import chess.engine
import chess.svg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shap import KernelExplainer
from sklearn.base import BaseEstimator, RegressorMixin

from chessplainer._deprecated.constants import MATE_VALUE
from chessplainer._deprecated.plot import board_to_latex_xskak


def min_max_norm(x, vmin=-MATE_VALUE, vmax=MATE_VALUE):
    return (x - vmin) / (vmax - vmin)


class StockfishWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        path=None,
        fit_analyze_time=1,
        predict_analyze_time=0.1,
        output_improvement_delta=False,
        mate_value=MATE_VALUE,
        **kwargs
    ):
        self.path = path
        self.fit_analyze_time = fit_analyze_time
        self.predict_analyze_time = predict_analyze_time
        self.output_improvement_delta = output_improvement_delta
        if path is None:
            self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        else:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.mate_value = mate_value

        self.base_score_ = None
        self.base_eval_ = None
        self.base_board_ = None
        self.predict_boards_ = None
        self.predict_evals_ = None

    def fit(self, X, y=None):
        # Saves the evaluation of the "base" board
        assert len(X) == 1  # Only one board is expected for fitting.
        self.base_board_ = X[0]
        self.base_eval_ = self.engine.analyse(
            X[0], chess.engine.Limit(time=self.fit_analyze_time)
        )["score"]
        # if self.base_eval_.is_mate():
        #     self.base_score_ = (self.base_eval_.white().mate() > 0) * self.mate_value
        if self.base_eval_.is_mate():
            # If it's a mate score: positive means White is mating, negative means Black is mating
            if self.base_eval_.white().mate() > 0:
                self.base_score_ = self.mate_value
            elif self.base_eval_.white().mate() == 0:
                if self.base_eval_.white() == chess.engine.MateGivenType():
                    self.base_score_ = self.mate_value
                else:
                    self.base_score_ = -self.mate_value
            else:
                self.base_score_ = -self.mate_value
        else:
            self.base_score_ = self.base_eval_.white().cp / 100
        return self

    def predict(self, boards):
        scores = self._analyze(boards, self.predict_analyze_time)
        numerical_evals = list()
        for i, score in enumerate(scores):
            if score is np.nan:
                numerical_evals.append(0)
                # numerical_evals.append(np.nan)
            else:
                if score.is_mate():
                    # If it's a mate score: positive means White is mating, negative means Black is mating
                    if score.white().mate() > 0:
                        numerical_evals.append(self.mate_value)
                    elif score.white().mate() == 0:
                        if score.white() == chess.engine.MateGivenType():
                            numerical_evals.append(self.mate_value)
                        else:
                            numerical_evals.append(-self.mate_value)
                    else:
                        numerical_evals.append(-self.mate_value)
                else:
                    current_score = score.white().cp / 100
                    if not self.output_improvement_delta:
                        numerical_evals.append(current_score)
                    else:
                        numerical_evals.append(current_score - self.base_score_)

        self.predict_evals_ = np.array(numerical_evals)
        return np.clip(np.array(numerical_evals), -self.mate_value, self.mate_value)

    def _analyze(self, boards, time=0.1):
        infos = list()
        self.predict_boards_ = boards
        for board in boards:
            if board.is_valid():
                info = self.engine.analyse(board, chess.engine.Limit(time=time))
                infos.append(info["score"])
            # elif board.is_checkmate():
            #     infos.append(chess.engine.PovScore(self.mate_value, not board.turn))
            else:
                flipped = board.copy()
                flipped.turn = not board.turn
                if flipped.is_valid():
                    info = self.engine.analyse(flipped, chess.engine.Limit(time=time))
                    infos.append(info["score"])
                # elif board.is_checkmate():
                #     infos.append(chess.engine.PovScore(self.mate_value, board.turn))
                else:
                    infos.append(np.nan)
            # else:
            #     infos.append(np.nan)
        return infos


class ChessExplainer(object):
    def __init__(
        self,
        board,
        wrappend_engine,
        explainer=KernelExplainer,
        mate_value=MATE_VALUE,
        explainer_fit_kwargs=None,
        explainer_predict_kwargs=None,
        **kwargs
    ):
        self.board = board
        self.engine = wrappend_engine
        self.explainer = explainer
        self.mate_value = mate_value
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
        self.df_["value"] = self.values_.ravel()
        return self.explanation_

    def to_svg(self, size=350, local_range=False, cmap="RdBu_r", **kwargs):
        shap_values = self.values_[0]

        if not local_range:
            min_val = -self.mate_value
            max_val = self.mate_value
        else:
            absmax_val = np.abs(shap_values).max()
            min_val = -absmax_val
            max_val = absmax_val

        shap_values = np.clip(shap_values, -self.mate_value, self.mate_value)

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

    def to_latex(self, cmap="RdBu_r", local_range=False):
        return board_to_latex_xskak(
            fen=self.board.fen(),
            pieces_idxs=self.pieces.keys(),
            scores=self.values_[0],
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
        return engine.predict(boards)

    explainer_ = explainer(
        f, np.zeros((1, n_pieces)), **explainer_fit_kwargs
    )  # function, data/masker
    explanation = explainer_(np.ones((1, n_pieces)), **explainer_predict_kwargs)
    return explanation, explainer_


if __name__ == "__main__":
    pass
