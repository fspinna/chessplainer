import chess
import chess.engine
import chess.svg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from shap import KernelExplainer
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd

from chessplainer.constants import MATE_VALUE
from chessplainer.plot import board_to_latex_xskak


def min_max_norm(x, vmin=-MATE_VALUE, vmax=MATE_VALUE):
    return (x - vmin) / (vmax - vmin)


class StockfishWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, path=None, fit_analyze_time=1, predict_analyze_time=0.1, output_improvement_delta=False,
                 **kwargs):
        self.path = path
        self.fit_analyze_time = fit_analyze_time
        self.predict_analyze_time = predict_analyze_time
        self.output_improvement_delta = output_improvement_delta
        if path is None:
            self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        else:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)

        self.base_score_ = None
        self.base_eval_ = None
        self.base_board_ = None
        self.predict_boards_ = None
        self.predict_evals_ = None

    def fit(self, X, y=None):
        # Saves the evaluation of the "base" board
        assert len(X) == 1  # Only one board is expected for fitting.
        self.base_board_ = X[0]
        self.base_eval_ = self.engine.analyse(X[0], chess.engine.Limit(time=self.fit_analyze_time))["score"]
        # if self.base_eval_.is_mate():
        #     self.base_score_ = (self.base_eval_.white().mate() > 0) * MATE_VALUE
        if self.base_eval_.is_mate():
            # If it's a mate score: positive means White is mating, negative means Black is mating
            if self.base_eval_.white().mate() > 0:
                self.base_score_ = MATE_VALUE
            elif self.base_eval_.white().mate() == 0:
                if self.base_eval_.white() == chess.engine.MateGivenType():
                    self.base_score_ = MATE_VALUE
                else:
                    self.base_score_ = -MATE_VALUE
            else:
                self.base_score_ = -MATE_VALUE
        else:
            self.base_score_ = self.base_eval_.white().cp / 100
        return self

    def predict(self, boards):
        scores = self._analyze(boards, self.fit_analyze_time)
        numerical_evals = list()
        for i, score in enumerate(scores):
            if score is np.nan:
                numerical_evals.append(0)
                # numerical_evals.append(np.nan)
            else:
                if score.is_mate():
                    # If it's a mate score: positive means White is mating, negative means Black is mating
                    if score.white().mate() > 0:
                        numerical_evals.append(MATE_VALUE)
                    elif score.white().mate() == 0:
                        if score.white() == chess.engine.MateGivenType():
                            numerical_evals.append(MATE_VALUE)
                        else:
                            numerical_evals.append(-MATE_VALUE)
                    else:
                        numerical_evals.append(-MATE_VALUE)
                else:
                    current_score = score.white().cp / 100
                    if not self.output_improvement_delta:
                        numerical_evals.append(current_score)
                    else:
                        numerical_evals.append(current_score - self.base_score_)

        self.predict_evals_ = np.array(numerical_evals)
        return np.clip(np.array(numerical_evals), -MATE_VALUE, MATE_VALUE)

    def _analyze(self, boards, time=0.1):
        infos = list()
        self.predict_boards_ = boards
        for board in boards:
            if board.is_valid():
                info = self.engine.analyse(board, chess.engine.Limit(time=time))
                infos.append(info["score"])
            # elif board.is_checkmate():
            #     infos.append(chess.engine.PovScore(MATE_VALUE, not board.turn))
            else:
                flipped = board.copy()
                flipped.turn = not board.turn
                if flipped.is_valid():
                    info = self.engine.analyse(flipped, chess.engine.Limit(time=time))
                    infos.append(info["score"])
                # elif board.is_checkmate():
                #     infos.append(chess.engine.PovScore(MATE_VALUE, board.turn))
                else:
                    infos.append(np.nan)
            # else:
            #     infos.append(np.nan)
        return infos


class ChessShap(object):
    def __init__(self, board, wrappend_engine, exp=KernelExplainer, **kwargs):
        self.board = board
        self.engine = wrappend_engine
        self.exp = exp

        self.pieces = dict(
            sorted(
                ((sq, pc) for sq, pc in board.piece_map().items()
                 if pc.symbol().lower() != 'k'),
                key=lambda kv: kv[0]
            )
        )

        self.shap_values_ = None
        self.mapped_shap_values = None
        self.explainer_ = None

    def predict(self, boards):
        y = self.engine.predict(boards)
        return y

    def shap_values(self, nsamples):
        self.shap_values_, self.explainer_ = chess_shap_values(self.board, self, nsamples, self.exp)
        df = pd.DataFrame([self.pieces], index=["piece"]).T
        df["value"] = self.shap_values_.ravel()
        df["piece"] = df["piece"].apply(lambda x: x.unicode_symbol())
        df = df.reset_index()
        df["square"] = df["index"].apply(lambda x: chess.square_name(x))
        df = df[["square", "piece", "value"]]
        df["feature_name"] = df["piece"] + " " + df["square"]
        self.df_ = df
        return self.shap_values_

    def to_svg(self, size=350, local_range=False, **kwargs):
        shap_values = self.shap_values_

        if not local_range:
            min_val = -MATE_VALUE
            max_val = MATE_VALUE
        else:
            absmax_val = np.abs(shap_values).max()
            min_val = -absmax_val
            max_val = absmax_val

        shap_values = np.clip(self.shap_values_[0], -MATE_VALUE, MATE_VALUE)

        norm = matplotlib.colors.TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
        cmap = plt.get_cmap('coolwarm_r')

        board_piece_map = self.board.piece_map()
        board_piece_map_wo_kings = {
            sq: piece for sq, piece in board_piece_map.items()
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

    def to_latex(self, cmap="RdBu", local_range=False):
        return board_to_latex_xskak(
            fen=self.board.fen(),
            pieces_idxs=self.pieces.keys(),
            scores=self.shap_values_[0],
            cmap=cmap,
            absolute=not local_range
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
                mapped_square = sorted(list(board_piece_map_wo_kings.keys()))[square_idx]
                masked_piece_map[mapped_square] = board_piece_map_wo_kings[mapped_square]  # add the corresponding piece
                masked_piece_map = {**masked_piece_map, **board_piece_map_only_kings}
        board = original_board.copy()
        board.set_piece_map(masked_piece_map)
        boards.append(board)
        # if board.is_valid():
        #     boards.append(board)
        # else:
        #     boards.append(chess.Board("R7/5k2/8/8/2r5/8/5K2/8 w - - 0 1"))  # append a draw
    return boards


def chess_shap_values(board, engine, nsamples, exp=KernelExplainer, verbose=True):
    n_pieces = len(board.piece_map()) - 2  # -2 because we don't consider kings

    def f(zs):
        boards = mask_chessboard(zs, board)
        return engine.predict(boards)

    explainer = exp(f, data=np.zeros((1, n_pieces)))
    shap_values = explainer.shap_values(np.ones((1, n_pieces)), nsamples=nsamples, silent=not verbose)
    return shap_values, explainer



if __name__ == "__main__":
    # board = chess.Board("5k2/4ppp1/8/8/2K5/5R2/8/8 w - - 0 1")
    # engine = StockfishWrapper()
    # chesshap = ChessShap(board, engine)
    # print(chesshap.board_eval)
    # shap_values = chesshap.shap_values(100)
    pass

    # board = chess.Board("8/8/8/7k/8/3K4/1Q6/8 w - - 0 1")  # white to mate
    # board = chess.Board("8/6q1/8/8/8/5k2/8/4K3 b - - 0 1")  # black to mate
    # # board = chess.Board()
    # board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")  # black to move
    # # board = chess.Board("7k/8/6Q1/8/8/6K1/8/8 b - - 0 1")  # stalemate
    # board = chess.Board("Q7/4pkp1/4p1p1/8/2K2P2/5R2/8/8 w - - 0 1")
    # engine = StockfishWrapper()
    # # shap_engine = ChessShap(board, engine)
    # # info = shap_engine.analyze([board])
    # # print("Score:", info)
    # # print(info.relative.cp)
    # # print(shap_engine.predict([board]))
    #
    # chesshap = ChessShap(board, engine)
    # shap_values = chesshap.shap_values(100)
    # # chesshap.plot()
    #
    # # mask_chessboard(np.array([[1,1,1]]), board)
    #
    # # chess.square Gets a square number by file and rank index.
