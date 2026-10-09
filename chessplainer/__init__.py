from importlib.metadata import PackageNotFoundError, version

from chessplainer.plot import board_svg
from chessplainer.positions import POSITIONS
from chessplainer.wrapper import ChessExplainer, EngineWrapper

try:
    __version__ = version("chessplainer")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0"

__all__ = ["ChessExplainer", "EngineWrapper", "POSITIONS", "board_svg"]
