"""Positions used in the chessplainer paper.

POSITIONS maps a short name to a dict with the position's title, FEN and source
(when the position comes from a game, a puzzle or an article). Use them with
chess.Board(POSITIONS[name]["fen"]).
"""

import chess

POSITIONS = {
    "example": {
        "title": "Two rooks against a queen",
        "fen": "8/2k5/2q5/8/4R3/4RK2/8/8 w - - 0 1",
        "source": None,
    },
    "self_blocking_pawn": {
        "title": "Self-blocking pawn",
        "fen": "Q7/3rpkp1/4p1p1/8/2K2P2/5R2/8/6q1 w - - 0 1",
        "source": None,
    },
    "trapped_rook": {
        "title": "Trapped rook",
        "fen": "8/4k3/1p1p4/pr1Pp3/4P3/PP1K4/1R6/8 w - - 0 31",
        "source": None,
    },
    "good_knight_bad_bishop": {
        "title": "Good knight against bad bishop",
        "fen": "8/4B3/1K6/2Pknp2/4p1p1/4P1P1/8/8 b - - 0 133",
        "source": "https://en.chessbase.com/post/endgame-blog-karsten-mueller-2020_38e",
    },
    "active_king": {
        "title": "Active king",
        "fen": "2b1rrk1/2pR1p2/1pq1pQp1/p3P2p/P1PR1K1P/5N2/2P2PP1/8 w - - 6 34",
        "source": "https://www.chess.com/forum/view/general/king-as-offensive-piece",
    },
    "active_king_2": {
        "title": "Active king (Khismatullin vs Eljanov)",
        "fen": "5Q2/5p1p/1pPr2p1/6k1/8/3pP2P/2q2PP1/3R1K2 w - - 4 44",
        "source": "https://www.chess.com/article/view/the-craziest-king-moves-ever",
    },
    "too_many_pieces": {
        "title": "Too many pieces (starting position)",
        "fen": chess.STARTING_FEN,
        "source": None,
    },
    "tactic_1": {
        "title": "Tactic: mate in 2",
        "fen": "rk2K3/NPR5/8/8/8/8/8/4Q3 w - - 0 1",
        "source": "https://www.reddit.com/r/ChessPuzzles/comments/1lf799h/white_to_play_and_mate_in_2/",
    },
    "tactic_2": {
        "title": "Tactic 2",
        "fen": "k7/rqQ3K1/8/8/8/5B2/8/8 w - - 0 1",
        "source": None,
    },
    "engines": {
        "title": "Engines disagree (Stockfish and Lc0)",
        "fen": "5rkq/3prp1p/5RpP/p1p5/5Q2/1B4P1/P4PK1/8 w - - 0 1",
        "source": "https://lichess.org/study/wxrovYNH/hBLBToEN",
    },
    "knight_vs_bishop_endgame": {
        "title": "Knight against bishop endgame puzzle",
        "fen": "8/3k4/3np2p/p5p1/P5P1/1P2B3/2K4P/8 w - - 0 1",
        "source": None,
    },
    "knight_vs_bishop_endgame_draw": {
        "title": "Knight against bishop endgame (draw)",
        "fen": "4B3/8/8/7n/8/8/p1PK4/bkB5 w - - 0 1",
        "source": None,
    },
}
