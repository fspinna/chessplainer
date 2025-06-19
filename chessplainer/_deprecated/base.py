import chess
import chess.engine
import chess.svg
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib
import matplotlib.colors as colors
from chessplainer._deprecated.my_shap import MyKernelExplainer
from chessplainer.utils import get_project_root

NAN_VALUE = 9999
MATE_VALUE = 100
COLUMNS = ["A", "B", "C", "D", "E", "F", "G", "H"]
ROWS = [str(i) for i in range(1, 9)]
SQUARES = [column + row for column in COLUMNS for row in ROWS]
MAPPING_DICT = dict(zip(list(range(64)), SQUARES))
RED = (1, 0, 0)
GREEN = (0, 1, 0)
ENGINE_LOCATION = get_project_root() / "engines" / "stockfish_14.1_win_32bit"


def min_max_norm(x, vmin=-MATE_VALUE, vmax=MATE_VALUE):
    return (x - vmin) / (vmax - vmin)


class EngineWrapper:
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, board):
        return


class StockfishWrapper(EngineWrapper):
    def __init__(self, path=None, time=0.1, **kwargs):
        self.path = path
        self.time = time
        if path is None:
            self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        else:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)

    def predict(self, boards):
        scores = self.analyze(boards, self.time)
        numerical_evals = list()
        for score in scores:
            if score is np.nan:
                numerical_evals.append(np.nan)
            else:
                if score.is_mate():
                    if score.turn:  # if white to move
                        numerical_evals.append(MATE_VALUE)
                    else:  # if black to move
                        numerical_evals.append(-MATE_VALUE)
                else:
                    if score.turn:  # if white to move
                        numerical_evals.append(score.relative.cp / 100)
                    else:
                        numerical_evals.append(-score.relative.cp / 100)
        return np.array(numerical_evals)

    def analyze(self, boards, time=0.1):
        infos = list()
        for board in boards:
            if board.is_valid():
                info = self.engine.analyse(board, chess.engine.Limit(time=time))
                infos.append(info["score"])
            else:
                infos.append(np.nan)
        return infos


class ChessShap(object):
    def __init__(self, board, wrappend_engine, **kwargs):
        self.board = board
        self.engine = wrappend_engine
        self.board_eval = self.engine.predict([self.board])[0]

        self.shap_values_ = None
        self.mapped_shap_values = None
        self.explainer_ = None

    # def predict(self, boards):
    #     y = self.engine.predict(boards)
    #     y = np.nan_to_num(y, nan=self.board_eval)  # replace np.nan with the original_prediction
    #     if self.board.turn:
    #         y -= self.board_eval  # positive means position improvement ex. 2 --> 4
    #     else:
    #         y = -1 * (y - self.board_eval)  # positive means position improvement ex -2 --> -4
    #     return y  # variation in evaluation + means good - means bad
    #     # y = np.nan_to_num(y, nan=0)  # replace np.nan with the original_prediction
    #     # return y

    def predict(self, boards):
        y = self.engine.predict(boards)
        # y = y[~np.isnan(y)]
        # y = np.nan_to_num(y, nan=0)  # replace np.nan with the original_prediction
        return y

    # def predict(self, boards):
    #     y = self.engine.predict(boards)
    #     # y = np.nan_to_num(y, nan=self.board_eval)  # replace np.nan with the original_prediction
    #     if self.board.turn:
    #         y -= self.board_eval  # positive means position improvement ex. 2 --> 4
    #     else:
    #         y = -1 * (y - self.board_eval)  # positive means position improvement ex -2 --> -4
    #     return y  # variation in evaluation + means good - means bad
    #     # y = np.nan_to_num(y, nan=0)  # replace np.nan with the original_prediction
    #     # return y

    def shap_values(self, nsamples):
        self.shap_values_, self.explainer_ = chess_shap_values(self.board, self, nsamples)
        return self.shap_values_

    def plot(self, size=350):
        # norm = colors.CenteredNorm(clip=True)
        # norm = colors.SymLogNorm(vmin=-100, vmax=100, clip=True)
        norm = colors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
        original_board = self.board
        arrows = list()
        board_piece_map_wo_kings = dict()
        board_piece_map_only_kings = dict()
        board_piece_map = self.board.piece_map()
        for key in board_piece_map:
            if board_piece_map[key].symbol() not in ["k", "K"]:
                board_piece_map_wo_kings[key] = board_piece_map[key]
            else:
                board_piece_map_only_kings[key] = board_piece_map[key]
        for i, shap_value in enumerate(self.shap_values_[0]):
            if shap_value >= 0:
                color = list(GREEN)
            else:
                color = list(RED)
            color.append(norm(np.abs(shap_value)))
            color = matplotlib.colors.to_hex(color, keep_alpha=True)
            mapped_square = sorted(list(board_piece_map_wo_kings.keys()))[i]
            arrows.append(chess.svg.Arrow(tail=mapped_square, head=mapped_square, color=color))
        # [chess.svg.Arrow(tail=chess.E4, head=chess.E4, color='#dddcdc9a')]
        board = chess.svg.board(original_board, arrows=arrows, size=size)
        return board

    # def plot(self, size=350):
    #     norm = colors.CenteredNorm(clip=True)
    #     # norm = colors.SymLogNorm(vmin=-100, vmax=100, clip=True)
    #     # norm = colors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    #     original_board = self.board
    #     arrows = list()
    #     board_piece_map = self.board.piece_map()
    #     for i, shap_value in enumerate(self.shap_values_[0]):
    #         if shap_value >= 0:
    #             color = list(GREEN)
    #         else:
    #             color = list(RED)
    #         color.append(norm(np.abs(shap_value)))
    #         color = matplotlib.colors.to_hex(color, keep_alpha=True)
    #         mapped_square = sorted(list(board_piece_map.keys()))[i]
    #         arrows.append(chess.svg.Arrow(tail=mapped_square, head=mapped_square, color=color))
    #     # [chess.svg.Arrow(tail=chess.E4, head=chess.E4, color='#dddcdc9a')]
    #     board = chess.svg.board(original_board, arrows=arrows, size=size)
    #     return board


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


# def mask_chessboard(zs, original_board):
#     # for shap
#     # def mask_chessboard that takes a chessboard and a set of arrays that tell the function which piece to remove
#     # ex. if there are 4 pieces in the board
#     # zs [[0, 1, 1, 0], [1, 1, 0, 1]...]
#     # where 0 means remove and 1 keep
#     # 1. convert z to position
#     # 2. check if valid
#     # 3. evaluate position (maybe relative to full board eval to see if pos is improved or not)
#     # this function is passed to shap
#     # explainer = shap.KernelExplainer(mask_chessboard, data=np.zeros((1, n. of pieces on the full board))))
#     board_piece_map = original_board.piece_map()
#     boards = list()
#     for z in zs:  # for each combination of presence/absence of pieces
#         z = z.ravel()
#         masked_piece_map = dict()
#         # board = board.copy()
#         for square_idx, square in enumerate(z):  # for each occupied square
#             if square:  # if the square is kept
#                 mapped_square = sorted(list(board_piece_map.keys()))[square_idx]
#                 masked_piece_map[mapped_square] = board_piece_map[mapped_square]  # add the corresponding piece
#         board = original_board.copy()
#         board.set_piece_map(masked_piece_map)
#         boards.append(board)
#         # if board.is_valid():
#         #     boards.append(board)
#         # else:
#         #     boards.append(chess.Board("R7/5k2/8/8/2r5/8/5K2/8 w - - 0 1"))  # append a draw
#     return boards


def chess_shap_values(board, engine, nsamples, verbose=True):
    n_pieces = len(board.piece_map()) - 2

    def f(zs):
        boards = mask_chessboard(zs, board)
        return engine.predict(boards)

    explainer = MyKernelExplainer(f, data=np.zeros((1, n_pieces)))
    shap_values = explainer.shap_values(np.ones((1, n_pieces)), nsamples=nsamples, silent=verbose)
    return shap_values, explainer


if __name__ == "__main__":
    board = chess.Board("5k2/4ppp1/8/8/2K5/5R2/8/8 w - - 0 1")
    engine = StockfishWrapper()
    chesshap = ChessShap(board, engine)
    print(chesshap.board_eval)
    shap_values = chesshap.shap_values(100)

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
