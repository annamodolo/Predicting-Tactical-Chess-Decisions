import numpy as np
import pandas as pd
import chess
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

BASE_COLS = ["is_capture", "captured_value", "gives_check"]
MATE_COLS = ["gives_mate"]
DYNAMIC_COLS = ["opp_replies", "to_attacked", "from_attacked"]
POSITION_COLS = ["is_promotion", "material_balance", "n_legal"]
NEW_COLS = ["moved_value", "n_attackers", "n_defenders", "hangs_piece", "king_dist"]
CAT_COLS = ["piece_type"]
ORIGINAL_NUM = BASE_COLS + MATE_COLS + DYNAMIC_COLS + POSITION_COLS


def material(board, color):
    return sum(VAL[p.piece_type] for p in board.piece_map().values() if p.color == color)


def candidate_rows(row):
    """One row per legal move after the opponent's first move; label=1 for the solution."""
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    board.push_uci(moves[0])              # the opponent's move
    correct = moves[1]                    # the correct move

    me, opp = board.turn, not board.turn
    balance = material(board, me) - material(board, opp)
    n_legal = board.legal_moves.count()

    out = []
    for m in list(board.legal_moves):
        piece = board.piece_at(m.from_square)

        if board.is_en_passant(m):
            captured_value = 1
        elif board.is_capture(m):
            captured_value = VAL[board.piece_at(m.to_square).piece_type]
        else:
            captured_value = 0

        is_capture = int(board.is_capture(m))
        gives_check = int(board.gives_check(m))
        from_attacked = int(board.is_attacked_by(opp, m.from_square))

        board.push(m)
        n_att = len(board.attackers(opp, m.to_square))
        n_def = len(board.attackers(me, m.to_square))
        gives_mate = int(board.is_checkmate())
        opp_replies = board.legal_moves.count()
        king_dist = chess.square_distance(m.to_square, board.king(opp))
        board.pop()

        out.append({
            "PuzzleId": row["PuzzleId"],
            "uci": m.uci(),
            "label": int(m.uci() == correct),
            "piece_type": piece.piece_type,
            "is_capture": is_capture,
            "captured_value": captured_value,
            "gives_check": gives_check,
            "gives_mate": gives_mate,
            "is_promotion": int(m.promotion is not None),
            "from_attacked": from_attacked,
            "to_attacked": int(n_att > 0),
            "opp_replies": opp_replies,
            "material_balance": balance,
            "n_legal": n_legal,
            "moved_value": VAL[piece.piece_type],
            "n_attackers": n_att,
            "n_defenders": n_def,
            "hangs_piece": int(n_att > 0 and n_def == 0),
            "king_dist": king_dist,
        })
    return out


def build_candidates(df):
    return pd.DataFrame([c for _, r in df.iterrows() for c in candidate_rows(r)])


def split_by_puzzle(cand, test_size=0.2, seed=42):
    """Train/test split by puzzle, so moves of one position never appear in both."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(cand, groups=cand["PuzzleId"]))
    return cand.iloc[train_idx].copy(), cand.iloc[test_idx].copy()


def topk(d, score, ks=(1, 3), seed=42):
    """Share of puzzles where the correct move is ranked within the top k."""
    d = d.copy()
    d["_s"] = np.asarray(score)
    d = d.sample(frac=1, random_state=seed)   # shuffle rows: ties are broken at random
    d["_r"] = d.groupby("PuzzleId")["_s"].rank(ascending=False, method="first")
    good = d[d["label"] == 1]
    return {f"top{k}": (good["_r"] <= k).mean() for k in ks}


def heuristic_score(d):
    return 3 * d["gives_check"] + d["is_capture"] * (1 + d["captured_value"])


def make_model(num, cat, kind="logit"):
    transformers = [("num", StandardScaler(), num)]
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    if kind == "logit":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200,
                            early_stopping=True, random_state=42)
    return Pipeline([("pre", ColumnTransformer(transformers)), ("clf", clf)])


def fit_eval(train, test, num, cat, kind="logit"):
    """Fit on train, return (top-k results on test, fitted model)."""
    model = make_model(num, cat, kind)
    model.fit(train[num + cat], train["label"])
    p = model.predict_proba(test[num + cat])[:, 1]
    return topk(test, p), model