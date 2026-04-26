"""
Feature engineering for player props models.
Builds features for pitcher strikeouts and batter hits predictions.
"""

import numpy as np
from typing import Optional
from src.utils.logging import get_logger

log = get_logger(__name__)


# ── Pitcher Strikeout Features ──────────────────────────────────────────

PITCHER_K_FEATURES = [
    "k_per_start_avg",       # Rolling avg Ks per start
    "k_per_start_std",       # Consistency (lower = more predictable)
    "k_per_9",               # K/9 rate
    "ip_per_start",          # Average innings per start (more IP = more K chances)
    "pitches_per_start",     # Pitch count tendency
    "k_rate",                # K/BF rate
    "recent_k_trend",        # Last 3 starts avg vs season avg (momentum)
    "opp_team_k_rate",       # Opposing team's strikeout rate
    "is_home",               # Home/away
    "era",                   # Pitcher ERA (proxy for quality)
    "whip",                  # Walks + Hits per IP
    "k_bb_ratio",            # Strikeout to walk ratio
]


def build_pitcher_k_features(
    pitcher_log: list[dict],
    game_index: int,
    opp_k_rate: float = 0.22,
    is_home: bool = False,
    min_starts: int = 5,
) -> Optional[dict]:
    """
    Build features for a pitcher strikeout prediction.
    Uses only data BEFORE game_index (no look-ahead).
    """
    prior_starts = pitcher_log[:game_index]
    if len(prior_starts) < min_starts:
        return None

    ks = [g["strikeouts"] for g in prior_starts]
    ips = [g["ip"] for g in prior_starts]
    pitches = [g["pitches"] for g in prior_starts]
    bfs = [g["batters_faced"] for g in prior_starts]
    ers = [g["earned_runs"] for g in prior_starts]
    walks = [g["walks"] for g in prior_starts]
    hits = [g["hits"] for g in prior_starts]

    total_ip = sum(ips)
    total_bf = sum(bfs)
    total_k = sum(ks)

    k_per_9 = (total_k / max(total_ip, 1)) * 9
    k_rate = total_k / max(total_bf, 1)
    era = (sum(ers) / max(total_ip, 1)) * 9
    whip = (sum(walks) + sum(hits)) / max(total_ip, 1)
    k_bb_ratio = total_k / max(sum(walks), 1)

    # Recent trend: last 3 starts vs overall
    recent_3 = ks[-3:] if len(ks) >= 3 else ks
    recent_k_avg = np.mean(recent_3)
    season_k_avg = np.mean(ks)
    recent_k_trend = recent_k_avg - season_k_avg

    return {
        "k_per_start_avg": np.mean(ks),
        "k_per_start_std": np.std(ks) if len(ks) > 1 else 2.0,
        "k_per_9": k_per_9,
        "ip_per_start": np.mean(ips),
        "pitches_per_start": np.mean(pitches),
        "k_rate": k_rate,
        "recent_k_trend": recent_k_trend,
        "opp_team_k_rate": opp_k_rate,
        "is_home": 1 if is_home else 0,
        "era": era,
        "whip": whip,
        "k_bb_ratio": k_bb_ratio,
    }


# ── Batter Hits Features ───────────────────────────────────────────────

BATTER_HITS_FEATURES = [
    "hits_per_game_avg",     # Rolling avg hits per game
    "hits_per_game_std",     # Consistency
    "batting_avg",           # BA
    "ab_per_game",           # At-bats per game (lineup spot proxy)
    "recent_hits_trend",     # Last 5 games vs season avg
    "opp_sp_era",            # Opposing starter ERA
    "opp_sp_whip",           # Opposing starter WHIP
    "opp_sp_hits_per_9",     # Opposing starter H/9
    "is_home",               # Home/away
    "extra_base_pct",        # Extra base hit %
    "k_rate",                # Batter K rate (lower = better contact)
    "bb_rate",               # Walk rate (plate discipline)
]


def build_batter_hits_features(
    batter_log: list[dict],
    game_index: int,
    opp_sp_stats: dict = None,
    is_home: bool = False,
    min_games: int = 15,
) -> Optional[dict]:
    """
    Build features for a batter hits prediction.
    Uses only data BEFORE game_index (no look-ahead).
    """
    prior_games = batter_log[:game_index]
    if len(prior_games) < min_games:
        return None

    hits = [g["hits"] for g in prior_games]
    abs_ = [g["at_bats"] for g in prior_games]
    ks = [g["strikeouts"] for g in prior_games]
    bbs = [g["walks"] for g in prior_games]
    doubles = [g["doubles"] for g in prior_games]
    triples = [g["triples"] for g in prior_games]
    hrs = [g["home_runs"] for g in prior_games]

    total_ab = sum(abs_)
    total_hits = sum(hits)
    total_xbh = sum(doubles) + sum(triples) + sum(hrs)
    ba = total_hits / max(total_ab, 1)

    # Recent trend: last 5 games
    recent_5 = hits[-5:] if len(hits) >= 5 else hits
    recent_avg = np.mean(recent_5)
    season_avg = np.mean(hits)

    opp = opp_sp_stats or {}
    opp_era = opp.get("era", 4.5)
    opp_whip = opp.get("whip", 1.30)
    opp_ip = opp.get("ip", 100)
    opp_hits = opp.get("hits_allowed", 0)
    opp_h_per_9 = (opp_hits / max(opp_ip, 1)) * 9 if opp_ip > 0 else 9.0

    return {
        "hits_per_game_avg": np.mean(hits),
        "hits_per_game_std": np.std(hits) if len(hits) > 1 else 0.8,
        "batting_avg": ba,
        "ab_per_game": np.mean(abs_),
        "recent_hits_trend": recent_avg - season_avg,
        "opp_sp_era": opp_era,
        "opp_sp_whip": opp_whip,
        "opp_sp_hits_per_9": opp_h_per_9,
        "is_home": 1 if is_home else 0,
        "extra_base_pct": total_xbh / max(total_hits, 1),
        "k_rate": sum(ks) / max(total_ab, 1),
        "bb_rate": sum(bbs) / max(total_ab + sum(bbs), 1),
    }


# ── Pitcher Outs Recorded Features ────────────────────────────────────

PITCHER_OUTS_FEATURES = [
    "outs_per_start_avg",     # Rolling avg outs per start
    "outs_per_start_std",     # Consistency
    "ip_per_start",           # Average innings per start
    "pitches_per_start",      # Pitch count tendency (more pitches = deeper into game)
    "pitches_per_ip",         # Efficiency (fewer pitches per IP = goes deeper)
    "k_rate",                 # K/BF rate (efficient outs)
    "bb_rate",                # Walk rate (inefficient = shorter outings)
    "hits_per_9",             # Hits allowed rate (traffic = shorter outings)
    "recent_outs_trend",      # Last 3 starts vs season avg
    "era",                    # ERA (quality proxy)
    "whip",                   # WHIP (baserunners = shorter outings)
    "is_home",                # Home pitchers tend to go slightly deeper
]


def build_pitcher_outs_features(
    pitcher_log: list[dict],
    game_index: int,
    is_home: bool = False,
    min_starts: int = 5,
) -> Optional[dict]:
    """
    Build features for pitcher outs recorded prediction.
    Outs recorded = IP converted to outs (e.g., 6.0 IP = 18 outs, 5.2 IP = 17 outs).
    Uses only data BEFORE game_index (no look-ahead).
    """
    prior_starts = pitcher_log[:game_index]
    if len(prior_starts) < min_starts:
        return None

    def ip_to_outs(ip: float) -> int:
        full = int(ip)
        partial = ip - full
        return full * 3 + round(partial * 10)

    ips = [g["ip"] for g in prior_starts]
    outs = [ip_to_outs(ip) for ip in ips]
    pitches = [g["pitches"] for g in prior_starts]
    ks = [g["strikeouts"] for g in prior_starts]
    bfs = [g["batters_faced"] for g in prior_starts]
    walks = [g["walks"] for g in prior_starts]
    hits = [g["hits"] for g in prior_starts]
    ers = [g["earned_runs"] for g in prior_starts]

    total_ip = sum(ips)
    total_bf = sum(bfs)
    total_k = sum(ks)
    total_walks = sum(walks)
    total_hits = sum(hits)

    era = (sum(ers) / max(total_ip, 1)) * 9
    whip = (total_walks + total_hits) / max(total_ip, 1)
    k_rate = total_k / max(total_bf, 1)
    bb_rate = total_walks / max(total_bf, 1)
    hits_per_9 = (total_hits / max(total_ip, 1)) * 9
    pitches_per_ip = sum(pitches) / max(total_ip, 1)

    recent_3 = outs[-3:] if len(outs) >= 3 else outs
    recent_outs_avg = np.mean(recent_3)
    season_outs_avg = np.mean(outs)

    return {
        "outs_per_start_avg": season_outs_avg,
        "outs_per_start_std": np.std(outs) if len(outs) > 1 else 3.0,
        "ip_per_start": np.mean(ips),
        "pitches_per_start": np.mean(pitches),
        "pitches_per_ip": pitches_per_ip,
        "k_rate": k_rate,
        "bb_rate": bb_rate,
        "hits_per_9": hits_per_9,
        "recent_outs_trend": recent_outs_avg - season_outs_avg,
        "era": era,
        "whip": whip,
        "is_home": 1 if is_home else 0,
    }
