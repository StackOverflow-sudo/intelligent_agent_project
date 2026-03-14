# group1.py
from __future__ import annotations

import math
import random
import statistics
from typing import Dict, List, Tuple, Optional
import os
import time

_LOG_PATH = "company1_debug_wrq.log"


def force_log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

from mable.cargo_bidding import TradingCompany, Bid
from mable.transport_operation import ScheduleProposal


class DirichletBinnedCDF:

    def __init__(self, bin_edges: List[float], prior: float = 1.0, decay: float = 1.0):
        self.bin_edges = list(bin_edges)
        self.alpha = [float(prior)] * len(self.bin_edges)
        self.decay = float(decay)

    @staticmethod
    def _br(a: List[float], x: float) -> int:
        lo, hi = 0, len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    @classmethod
    def with_uniform_prior(cls, bin_edges: List[float], prior: float = 1.0, decay: float = 1.0):
        return cls(bin_edges=bin_edges, prior=prior, decay=decay)

    def _maybe_decay(self):
        if self.decay < 1.0:
            self.alpha = [a * self.decay for a in self.alpha]

    def _bin_index(self, x: float) -> int:
        i = self._br(self.bin_edges, x)
        if i <= 0:
            return 0
        if i >= len(self.bin_edges):
            return len(self.bin_edges) - 1
        return i

    def pmf(self) -> List[float]:
        s = float(sum(self.alpha))
        if s <= 1e-12:
            n = len(self.alpha)
            return [1.0 / n] * n
        return [a / s for a in self.alpha]

    def quantile(self, p: float) -> float:
        p = max(0.0, min(1.0, float(p)))
        pmf = self.pmf()
        edges = self.bin_edges
        acc = 0.0
        for i, mass in enumerate(pmf):
            if acc + mass >= p:
                L = 0.0 if i == 0 else edges[i - 1]
                U = edges[i]
                if mass <= 1e-12:
                    return 0.5 * (L + U)
                frac = (p - acc) / mass
                return L + frac * (U - L)
            acc += mass
        return edges[-1]

    def update_exact(self, x: float, weight: float = 1.0):
        self._maybe_decay()
        self.alpha[self._bin_index(x)] += float(weight)

    def update_left_censored(self, u: float, weight: float = 1.0):
        self._maybe_decay()
        last = self._br(self.bin_edges, u)
        if last <= 0:
            self.alpha[0] += float(weight)
            return
        denom = sum(self.alpha[:last])
        if denom <= 1e-12:
            share = float(weight) / float(last)
            for i in range(last):
                self.alpha[i] += share
            return
        w = float(weight)
        for i in range(last):
            self.alpha[i] += w * (self.alpha[i] / denom)


class Company1(TradingCompany):

    # Debug
    FORCE_LOG = True
    LOG_EVERY_K_ROUNDS = 1

    # Market: EMWA and Left-censored
    RATIO_EDGES = [
        0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.96, 0.98,
        1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20, 1.30, 1.40,
        1.60, 2.00, 3.00, 5.00, 8.00, 12.0, 20.0, 35.0, 60.0
    ]
    _F_ratio = DirichletBinnedCDF.with_uniform_prior(RATIO_EDGES, prior=1.0, decay=0.999)

    _last_bids: Dict[str, float] = {}

    # Top-K Scheduling
    TOPK_K = 8
    MULTISTART_M_BASE = 12
    MAX_IP = 12

    POST_M = 8
    POST_MAX_IP = 12

    MAX_DT_MULT = 2.0
    MAX_MC_OVER_C = 3.0
    PENALTY_BONUS_TIGHT = 6000.0
    PENALTY_BONUS_LOOSE = 2000.0

    # no try-catch, but need to ensure there is no bug
    def _ensure_state(self):
        if not hasattr(self, "_rng") or self._rng is None:
            self._rng = random.Random()
        if not hasattr(self, "_future_trades") or self._future_trades is None:
            self._future_trades = []
        if not hasattr(self, "_mkt_ratio_ewma") or self._mkt_ratio_ewma is None:
            self._mkt_ratio_ewma = 2.0
        if not hasattr(self, "_low_ratio_ewma") or self._low_ratio_ewma is None:
            self._low_ratio_ewma = 1.25
        if not hasattr(self, "_round_idx"):
            self._round_idx = 0
        if not hasattr(self, "_bid_scale") or self._bid_scale is None:
            self._bid_scale = 1.0
        if not hasattr(self, "_hist_bids") or self._hist_bids is None:
            self._hist_bids = []
        if not hasattr(self, "_hist_wins") or self._hist_wins is None:
            self._hist_wins = []
        if not hasattr(self, "_last_round_bid_n"):
            self._last_round_bid_n = 0
        if not hasattr(self, "_round"):
            self._round = 0

    # tense-market strategy
    DBG_MAX_FILTER_FAIL = 12
    DBG_MAX_BID_SKIP = 10
    DBG_MAX_BID_SAMPLE = 12
    DBG_MAX_WIN_SAMPLE = 10
    DBG_MAX_INS_FAIL_SAMPLE = 8
    DBG_MAX_INS_WARN_SAMPLE = 6

    def _dbg_reset_round(self, stage: str):
        # stage: "INFORM" or "RECEIVE"
        self._dbg_stage = str(stage)
        self._dbg_filter_fail_samples = []
        self._dbg_bid_skip_samples = []
        self._dbg_bid_samples = []
        self._dbg_win_samples = []
        self._dbg_ins_fail_samples = []
        self._dbg_ins_warn_samples = []
        self._dbg_ins_fail_n = 0
        self._dbg_ins_warn_n = 0

    def _dbg_add(self, lst, item, limit: int):
        if len(lst) < int(limit):
            lst.append(item)

    def _dbg_record_filter_fail(self, **kw):
        self._dbg_add(self._dbg_filter_fail_samples, kw, int(self.DBG_MAX_FILTER_FAIL))

    def _dbg_record_bid_skip(self, **kw):
        self._dbg_add(self._dbg_bid_skip_samples, kw, int(self.DBG_MAX_BID_SKIP))

    def _dbg_record_bid_sample(self, **kw):
        # store full; we will rank & take top N later
        self._dbg_bid_samples.append(kw)

    def _dbg_record_win_sample(self, **kw):
        self._dbg_add(self._dbg_win_samples, kw, int(self.DBG_MAX_WIN_SAMPLE))

    def _dbg_record_ins_fail(self, **kw):
        self._dbg_ins_fail_n += 1
        # only sample in INFORM stage (tight focus)
        if getattr(self, "_dbg_stage", "") in ("INFORM", "INFORM_PLAN"):
            self._dbg_add(self._dbg_ins_fail_samples, kw, int(self.DBG_MAX_INS_FAIL_SAMPLE))

    def _dbg_record_ins_warn(self, **kw):
        self._dbg_ins_warn_n += 1
        if getattr(self, "_dbg_stage", "") in ("INFORM", "INFORM_PLAN"):
            self._dbg_add(self._dbg_ins_warn_samples, kw, int(self.DBG_MAX_INS_WARN_SAMPLE))
    def _log(self, msg: str):
        if not getattr(self, "FORCE_LOG", False):
            return
        print(f"[{self.name}] {msg}")

    def _vessels(self):
        return list(self.fleet) if self.fleet is not None else []

    def _tkey(self, trade) -> str:
        for attr in ("id", "trade_id", "uid"):
            if hasattr(trade, attr):
                v = getattr(trade, attr)
                if v is not None:
                    return str(v)

        parts = []
        for attr in (
            "origin_port", "destination_port", "earliest_pickup_time", "latest_dropoff_time",
            "earliest_pickup", "latest_delivery", "amount"
        ):
            if hasattr(trade, attr):
                parts.append(str(getattr(trade, attr)))
        return "|".join(parts) if parts else repr(trade)

    # Convert fuel consumption to money
    def _fuel_to_money(self, vessel, fuel: float) -> float:
        if hasattr(vessel, "get_cost"):
            return float(vessel.get_cost(float(fuel)))
        return float(fuel)

    def predict_direct_cost(self, vessel, trade) -> float:
        handling = 0.0
        loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
        handling += float(vessel.get_loading_consumption(loading_time))
        handling += float(vessel.get_unloading_consumption(loading_time))

        start_port = getattr(vessel, "current_port", None)
        if start_port is None:
            start_port = getattr(vessel, "location", None)
        if start_port is None:
            start_port = getattr(self, "headquarters", None)

        rep_km = float(self.headquarters.get_network_distance(start_port, trade.origin_port))
        voy_km = float(self.headquarters.get_network_distance(trade.origin_port, trade.destination_port))

        fuel = 0.0

        if rep_km > 0:
            rep_t = float(vessel.get_travel_time(rep_km))
            spd = getattr(vessel, "speed", 1.0)
            if hasattr(vessel, "get_ballast_consumption"):
                fuel += float(vessel.get_ballast_consumption(rep_t, spd))
            else:
                # fallback: laden * 1.15
                fuel += float(vessel.get_laden_consumption(rep_t, spd)) * 1.15

        if voy_km > 0:
            voy_t = float(vessel.get_travel_time(voy_km))
            spd = getattr(vessel, "speed", 1.0)
            fuel += float(vessel.get_laden_consumption(voy_t, spd))

        fuel_money = self._fuel_to_money(vessel, float(fuel))
        return float(handling + fuel_money)

    def predict_voyage_cost(self, vessel, trade) -> float:
        loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
        handling = float(vessel.get_loading_consumption(loading_time)) + float(vessel.get_unloading_consumption(loading_time))

        voy_km = float(self.headquarters.get_network_distance(trade.origin_port, trade.destination_port))
        travel_cost = 0.0
        if voy_km > 0:
            voy_t = float(vessel.get_travel_time(voy_km))
            spd = getattr(vessel, "speed", 1.0)
            travel_cost = float(vessel.get_laden_consumption(voy_t, spd))
        total = handling + travel_cost
        return float(total) if math.isfinite(total) and total > 0 else 1.0

    def voyage_cost_proxy(self, trade) -> float:
        best = float("inf")
        for v in self._vessels():
            c = self.predict_voyage_cost(v, trade)
            if c < best:
                best = c
        if (not math.isfinite(best)) or best <= 0:
            return 1.0
        return float(best)

    def direct_cost_proxy(self, trade) -> float:
        best = float("inf")
        for v in self._vessels():
            c = self.predict_direct_cost(v, trade)
            if c < best:
                best = c
        if (not math.isfinite(best)) or best <= 0:
            return 1.0
        return float(best)

    def _time_to_cost(self, vessel, dt: float) -> float:
        dt = max(0.0, float(dt))

        # idling
        for fn_name in ("get_idling_consumption", "get_idle_consumption", "get_idling_cost"):
            if hasattr(vessel, fn_name):
                fn = getattr(vessel, fn_name)
                fuel = float(fn(dt))
                if math.isfinite(fuel):
                    return self._fuel_to_money(vessel, fuel)

        # ballast
        if hasattr(vessel, "get_ballast_consumption"):
            fuel = float(vessel.get_ballast_consumption(dt, vessel.speed))
            if math.isfinite(fuel):
                return self._fuel_to_money(vessel, fuel)

        # laden
        if hasattr(vessel, "get_laden_consumption"):
            fuel = float(vessel.get_laden_consumption(dt, vessel.speed))
            if math.isfinite(fuel):
                return self._fuel_to_money(vessel, fuel)

        return float(dt)

    # -------------------------
    # Insertion search (limited & guarded)
    # -------------------------
    def _ip_subset(self, schedule, max_ip: int) -> List[int]:
        ips = list(schedule.get_insertion_points())
        if len(ips) <= max_ip:
            return ips
        return ips[:2] + ips[-(max_ip - 2):]
    def _best_insertion(self, vessel, base_schedule, trade, max_ip: int, C: float) -> Tuple[Optional[object], float]:
        base_ct = float(base_schedule.completion_time())
        ips = self._ip_subset(base_schedule, max_ip=max_ip)
        ips_len = int(len(ips))

        best_s = None
        best_mc = float("inf")

        tried = 0
        verify_false = 0
        dt_prune = 0

        def consider(cand):
            nonlocal best_s, best_mc, tried, verify_false, dt_prune
            tried += 1
            if not cand.verify_schedule():
                verify_false += 1
                return
            ct = float(cand.completion_time())
            if base_ct > 1e-6 and ct > self.MAX_DT_MULT * base_ct:
                dt_prune += 1
                return
            dt = max(0.0, ct - base_ct)

            mc_time = self._time_to_cost(vessel, dt)

            loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
            mc_hand = float(vessel.get_loading_consumption(loading_time) + vessel.get_unloading_consumption(loading_time))

            # reposition-aware proxy (money)
            mc_proxy = float(self.predict_direct_cost(vessel, trade))

            mc = float(max(mc_time + mc_hand, mc_proxy))
            if mc < best_mc:
                best_s, best_mc = cand, float(mc)

        # default append
        cand = base_schedule.copy()
        cand.add_transportation(trade)
        consider(cand)

        # limited enumeration
        for a_i, idx_pick in enumerate(ips):
            for idx_drop in ips[a_i:]:
                cand = base_schedule.copy()
                cand.add_transportation(trade, idx_pick, idx_drop)
                consider(cand)

        if best_s is None:
            self._dbg_record_ins_fail(
                trade=self._tkey(trade),
                vessel=str(getattr(vessel, "name", vessel)),
                base_ct=float(base_ct),
                ips=int(ips_len),
                max_ip=int(max_ip),
                tried=int(tried),
                verify_false=int(verify_false),
                dt_prune=int(dt_prune),
            )
        else:
            # Warn on fragile / expensive insertions (tight diagnosis)
            ratio = (float(best_mc) / float(C)) if float(C) > 1e-9 else 0.0
            if ips_len < 3 or ratio > 2.0:
                self._dbg_record_ins_warn(
                    trade=self._tkey(trade),
                    vessel=str(getattr(vessel, "name", vessel)),
                    base_ct=float(base_ct),
                    ips=int(ips_len),
                    ratio=float(ratio),
                    mc=float(best_mc),
                    C=float(C),
                )

        return best_s, best_mc
    def _best_insertion_tail(self, vessel, base_schedule, trade, max_ip: int, C: float):
        """Tail insertion points only. No try/except. (Used mainly in post-scheduling.)"""
        base_ct = float(base_schedule.completion_time())
        ips_all = list(base_schedule.get_insertion_points())
        ips = ips_all[-max_ip:] if len(ips_all) > max_ip else ips_all
        ips_len = int(len(ips))

        best_s = None
        best_mc = float("inf")

        tried = 0
        verify_false = 0
        dt_prune = 0

        def consider(cand):
            nonlocal best_s, best_mc, tried, verify_false, dt_prune
            tried += 1
            if not cand.verify_schedule():
                verify_false += 1
                return
            ct = float(cand.completion_time())
            if base_ct > 1e-6 and ct > self.MAX_DT_MULT * base_ct:
                dt_prune += 1
                return
            dt = max(0.0, ct - base_ct)
            mc = self._time_to_cost(vessel, dt)

            loading_time = vessel.get_loading_time(trade.cargo_type, trade.amount)
            fuel = float(vessel.get_loading_consumption(loading_time) + vessel.get_unloading_consumption(loading_time))
            mc += self._fuel_to_money(vessel, fuel)

            if mc < best_mc:
                best_s, best_mc = cand, float(mc)

        cand = base_schedule.copy()
        cand.add_transportation(trade)
        consider(cand)

        for a_i, idx_pick in enumerate(ips):
            for idx_drop in ips[a_i:]:
                cand = base_schedule.copy()
                cand.add_transportation(trade, idx_pick, idx_drop)
                consider(cand)

        if best_s is None:
            self._dbg_record_ins_fail(
                trade=self._tkey(trade),
                vessel=str(getattr(vessel, "name", vessel)),
                base_ct=float(base_ct),
                ips=int(ips_len),
                max_ip=int(max_ip),
                tried=int(tried),
                verify_false=int(verify_false),
                dt_prune=int(dt_prune),
                tail=True,
            )
        else:
            ratio = (float(best_mc) / float(C)) if float(C) > 1e-9 else 0.0
            if ips_len < 3 or ratio > 2.0:
                self._dbg_record_ins_warn(
                    trade=self._tkey(trade),
                    vessel=str(getattr(vessel, "name", vessel)),
                    base_ct=float(base_ct),
                    ips=int(ips_len),
                    ratio=float(ratio),
                    mc=float(best_mc),
                    C=float(C),
                    tail=True,
                )

        return best_s, best_mc

    # -------------------------
    # Top-K multi-start(planning) 
    # -------------------------
    def _plan_once(self, trades_order: List, max_ip: int) -> Tuple[Dict, Dict[str, float], List]:
        vessels = self._vessels()
        schedules: Dict = {}
        mc_by_key: Dict[str, float] = {}
        scheduled_trades: List = []

        for tr in trades_order:
            C = float(self.direct_cost_proxy(tr))

            best_v = None
            best_s = None
            best_mc = float("inf")

            for v in vessels:
                base = schedules.get(v, v.schedule)
                cand, mc = self._best_insertion(v, base, tr, max_ip=max_ip, C=C)
                if cand is None:
                    continue
                if mc < best_mc:
                    best_v, best_s, best_mc = v, cand, mc

            if best_v is not None and best_s is not None:
                schedules[best_v] = best_s
                mc_by_key[self._tkey(tr)] = float(best_mc)
                scheduled_trades.append(tr)

        return schedules, mc_by_key, scheduled_trades

    def propose_schedules_topk(self, trades: List, K: int) -> Tuple[List[Tuple], Dict[str, float]]:
        trades = list(trades) if trades else []
        if not trades:
            return [], {}

        M = int(self.MULTISTART_M_BASE)
        if len(trades) > 90:
            M = max(6, M // 2)

        plans: List[Tuple] = []
        for _ in range(max(1, M)):
            order = trades[:]
            self._rng.shuffle(order)
            schedules, mc_by_key, scheduled_trades = self._plan_once(order, max_ip=int(self.MAX_IP))
            total_mc = sum(mc_by_key.values()) if mc_by_key else float("inf")
            plans.append((len(scheduled_trades), -total_mc, schedules, mc_by_key, scheduled_trades))

        plans.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top = plans[:max(1, K)]

        cnt: Dict[str, int] = {}
        for _, _, _, mc_by_key, _ in top:
            for tk in mc_by_key.keys():
                cnt[tk] = cnt.get(tk, 0) + 1
        p = {tk: cnt[tk] / float(len(top)) for tk in cnt}
        return top, p

    def marginal_cost_from_topk(self, top_plans: List[Tuple]) -> Dict[str, float]:
        agg: Dict[str, List[float]] = {}
        for _, _, _, mc_by_key, _ in top_plans:
            for tk, c in mc_by_key.items():
                agg.setdefault(tk, []).append(float(c))
        return {tk: (sum(vals) / len(vals)) for tk, vals in agg.items()}

    # -------------------------
    # Market parameters
    # -------------------------
    def _market_ratio(self) -> float:
        return float(getattr(self, "_mkt_ratio_ewma", 1.0))

    def market_strength(self) -> float:
        r = self._market_ratio()
        if r <= 2.0:
            return 0.0
        if r >= 8.0:
            return 1.0
        return (r - 2.0) / 6.0


    def _opp_target_price(self, C: float, s: float, undercut_eps: float) -> float:
        r_mkt = float(getattr(self, "_mkt_ratio_ewma", 1.0))
        try:
            r_q = float(self._F_ratio.quantile(0.35))
        except Exception:
            r_q = r_mkt
        r = max(1.05, min(r_mkt, r_q) if math.isfinite(r_q) else r_mkt)
        return float(C * r * (1.0 - float(undercut_eps)))
        
    def _params(self, s: float):
        # Candidate selection threshold. Lower => more candidates.
        # Empirically good defaults for your tight-market experiments.
        p_select = 0.45 - 0.25 * s   # tight≈0.45, loose≈0.20
        max_bids = int(round(20 + 20 * s))  # tight≈20, loose≈40
        markup = 0.10 - 0.08 * s
        safety = 0.14 - 0.10 * s
        low_eps = 0.0 if s < 0.60 else (0.006 + 0.010 * (s - 0.60) / 0.40)
        min_profit = 0.06 - 0.04 * s
        return p_select, max_bids, markup, safety, low_eps, min_profit




    # ---------------------------------------------------------------------------
    # Frameworklifecycle
    # ---------------------------------------------------------------------------
    def pre_inform(self, trades, time_auction):
        self._ensure_state()
        self._future_trades = list(trades) if trades else []
    def inform(self, trades, *args, **kwargs):
        self._ensure_state()
        self._dbg_reset_round(stage="INFORM")
        self._round_idx += 1

        trades = list(trades) if trades else []
        t_now = int(getattr(self, "_round", 0))
        mkt_ewma = float(getattr(self, "_mkt_ratio_ewma", 1.0))
        low_ewma = float(getattr(self, "_low_ratio_ewma", 1.25))
        s = float(self.market_strength())

        force_log(
            f"[ROUND] t={t_now} r={self._round_idx} trades={len(trades)} "
            f"mkt_ewma={mkt_ewma:.3f} low_ewma={low_ewma:.3f} s={s:.2f}"
        )

        if not trades:
            return []

        p_select, max_bids, markup, safety, low_eps, min_profit = self._params(s)
        undercut_eps = float(low_eps)
        bid_scale = float(getattr(self, "_bid_scale", 1.0))

        q10 = float(self._F_ratio.quantile(0.10))
        q50 = float(self._F_ratio.quantile(0.50))
        q90 = float(self._F_ratio.quantile(0.90))

        force_log(
            f"[PARAM] s={s:.2f} p_sel={p_select:.2f} max_bids={max_bids} "
            f"markup={markup:.3f} safety={safety:.3f} undercut={undercut_eps:.3f} "
            f"min_profit={min_profit:.3f} bid_scale={bid_scale:.3f} "
            f"TOPK_K={int(self.TOPK_K)} M_BASE={int(self.MULTISTART_M_BASE)} MAX_IP={int(self.MAX_IP)} "
            f"q10={q10:.2f} q50={q50:.2f} q90={q90:.2f}"
        )
        if s < 0.45:
            force_log(
                "[TIGHT_OVR] tight market"
            )

        # ---------- Planning (Top-K multi-start) ----------
        self._dbg_stage = "INFORM_PLAN"
        top_plans, p = self.propose_schedules_topk(trades, K=int(self.TOPK_K))
        self._dbg_stage = "INFORM"

        if not top_plans:
            force_log("[TOPK] got=0 (no plans)")
            force_log("[BIDS] n=0 s={:.2f} (no feasible plans)".format(s))
            return []

        best_scheduled = int(top_plans[0][0])
        best_total_mc = float(-top_plans[0][1]) if math.isfinite(float(top_plans[0][1])) else float("inf")
        avg_sched = float(sum(int(x[0]) for x in top_plans)) / float(len(top_plans))

        p_vals = list(p.values()) if p else []
        p_med = float(statistics.median(p_vals)) if p_vals else 0.0
        p_max = float(max(p_vals)) if p_vals else 0.0

        force_log(
            f"[TOPK] got={len(top_plans)} best_sched={best_scheduled} best_total_mc={best_total_mc:.1f} "
            f"avg_sched={avg_sched:.2f} p_med={p_med:.2f} p_max={p_max:.2f}"
        )

        mc = self.marginal_cost_from_topk(top_plans)

        # ---------- Filtering candidates ----------
        N = int(len(trades))
        n_have_mc = 0
        n_pass_p = 0
        n_snipes = 0

        mc_over_c_vals: List[float] = []
        pr_vals_seen: List[float] = []

        cand: List = []
        snipes: List = []

        no_mc_logged = 0
        for tr in trades:
            tk = self._tkey(tr)

            if tk not in mc:
                if no_mc_logged < 5:
                    self._dbg_record_filter_fail(reason="no_mc", trade=tk)
                    no_mc_logged += 1
                continue

            n_have_mc += 1
            pr = float(p.get(tk, 0.0))
            C = float(self.direct_cost_proxy(tr))
            m = float(mc[tk])
            mc_over_c = (m / C) if C > 1e-9 else 0.0
            mc_over_c_vals.append(float(mc_over_c))
            pr_vals_seen.append(float(pr))

            if pr >= p_select:
                cand.append(tr)
                n_pass_p += 1
            elif pr < 0.20:
                self._dbg_record_filter_fail(reason="low_pr", trade=tk, pr=float(pr), p_sel=float(p_select), mc_over_c=float(mc_over_c))

            if C > 1e-9 and mc_over_c > float(self.MAX_MC_OVER_C):
                self._dbg_record_filter_fail(reason="mc_over_c", trade=tk, pr=float(pr), C=float(C), mc=float(m), mc_over_c=float(mc_over_c))

            if s >= 0.55 and C > 1e-9 and mc_over_c <= 0.65:
                snipes.append(tr)
                n_snipes += 1

        if mc_over_c_vals:
            xs = sorted(mc_over_c_vals)
            nxs = len(xs)
            p90 = xs[int(0.90 * (nxs - 1))]
            force_log(
                f"[MC_STATS] schedulable={len(mc)}/{N} mc/C min={xs[0]:.2f} med={xs[nxs//2]:.2f} "
                f"p90={p90:.2f} max={xs[-1]:.2f} pr_med={(statistics.median(pr_vals_seen) if pr_vals_seen else 0.0):.2f} "
                f"pr_max={(max(pr_vals_seen) if pr_vals_seen else 0.0):.2f}"
            )
        else:
            force_log(f"[MC_STATS] schedulable={len(mc)}/{N} (no mc/C stats)")

        pre_merge = int(len(cand))
        if s >= 0.55:
            seen = set()
            merged = []
            for tr in snipes + sorted(cand, key=lambda t: float(p.get(self._tkey(t), 0.0)), reverse=True):
                tk = self._tkey(tr)
                if tk in seen:
                    continue
                seen.add(tk)
                merged.append(tr)
            cand = merged
        else:
            cand.sort(key=lambda t: float(p.get(self._tkey(t), 0.0)), reverse=True)

        cand = cand[:min(len(cand), int(max_bids))]
        force_log(
            f"[FILTER_SUM] total={N} have_mc={n_have_mc} pass_p={n_pass_p} snipes={n_snipes} "
            f"cand_pre_merge={pre_merge} final={len(cand)}/{int(max_bids)}"
        )

        # placeholder: debugging
        for srec in getattr(self, "_dbg_filter_fail_samples", []):
            if srec.get("reason") == "no_mc":
                force_log(f"[FILTER_FAIL] reason=no_mc trade={srec.get('trade')}")
            elif srec.get("reason") == "low_pr":
                force_log(
                    f"[FILTER_FAIL] reason=low_pr trade={srec.get('trade')} pr={float(srec.get('pr',0.0)):.2f} "
                    f"p_sel={float(srec.get('p_sel',0.0)):.2f} mc/C={float(srec.get('mc_over_c',0.0)):.2f}"
                )
            elif srec.get("reason") == "mc_over_c":
                force_log(
                    f"[FILTER_FAIL] reason=mc_over_c trade={srec.get('trade')} pr={float(srec.get('pr',0.0)):.2f} "
                    f"C={float(srec.get('C',0.0)):.1f} mc={float(srec.get('mc',0.0)):.1f} mc/C={float(srec.get('mc_over_c',0.0)):.2f}"
                )

        # Insertion summary (bounded samples)
        if int(getattr(self, "_dbg_ins_fail_n", 0)) > 0 or int(getattr(self, "_dbg_ins_warn_n", 0)) > 0:
            force_log(
                f"[INS_SUM] ins_fail={int(getattr(self,'_dbg_ins_fail_n',0))} ins_warn={int(getattr(self,'_dbg_ins_warn_n',0))} "
                f"fail_samples={len(getattr(self,'_dbg_ins_fail_samples',[]))} warn_samples={len(getattr(self,'_dbg_ins_warn_samples',[]))}"
            )
            for rec in getattr(self, "_dbg_ins_fail_samples", []):
                force_log(
                    f"[INS_FAIL] trade={rec.get('trade')} vessel={rec.get('vessel')} base_ct={float(rec.get('base_ct',0.0)):.1f} "
                    f"ips={int(rec.get('ips',0))}/{int(rec.get('max_ip',0))} tried={int(rec.get('tried',0))} "
                    f"verify_false={int(rec.get('verify_false',0))} dt_prune={int(rec.get('dt_prune',0))} tail={bool(rec.get('tail',False))}"
                )
            for rec in getattr(self, "_dbg_ins_warn_samples", []):
                force_log(
                    f"[INS_WARN] trade={rec.get('trade')} vessel={rec.get('vessel')} ips={int(rec.get('ips',0))} "
                    f"mc/C={float(rec.get('ratio',0.0)):.2f} mc={float(rec.get('mc',0.0)):.1f} C={float(rec.get('C',0.0)):.1f} "
                    f"tail={bool(rec.get('tail',False))}"
                )

        # -------------------- Bidding --------------------
        bids: List[Bid] = []
        w_abs = 0.65 - 0.25 * s
        low_ratio = float(getattr(self, "_low_ratio_ewma", 1.35))

        skip_logged = 0
        for tr in cand:
            tk = self._tkey(tr)
            pr = float(p.get(tk, 0.0))
            m = float(mc.get(tk, 0.0))
            if (not math.isfinite(m)) or m <= 0:
                self._dbg_record_bid_skip(reason="bad_mc", trade=tk, pr=float(pr), mc=float(m))
                continue

            C = float(self.direct_cost_proxy(tr))
            Cv = float(self.voyage_cost_proxy(tr))
            abs_c = float(C)

            c_hat = w_abs * abs_c + (1.0 - w_abs) * m

            is_tight = (s < 0.6)
            markup_eff = markup
            safety_eff = safety
            min_profit_eff = min_profit
            # Tight: prioritize winning (close-to-cost bids) over margin; avoid over-raising the floor
            if is_tight:
                markup_eff = 0.0
                safety_eff = 0.0
                min_profit_eff = 0.0

            base_bid = c_hat * (1.0 + markup_eff)
            lowball = m * (1.0 + min_profit_eff)

            undercut_used = 0
            if is_tight:
                # Tight market (reverse 2nd-price): payment >= our bid.
                # Best response is to bid as close as possible to our true (marginal) execution cost.
                eps = 0.003  # tiny buffer above mc
                b = m * (1.0 + eps)
                floor_bid = m
            elif s >= 0.60:
                opp_target = self._opp_target_price(abs_c, s, undercut_eps)
                b = min(base_bid, opp_target)
                undercut_used = 1 if b < base_bid - 1e-9 else 0
                floor_bid = lowball
            else:
                b = base_bid
                floor_bid = max(c_hat * (1.0 + safety_eff), lowball)

            b *= bid_scale
            b = max(float(b), float(floor_bid))

            bids.append(Bid(amount=float(b), trade=tr))
            Company1._last_bids[tk] = float(b)

            # record bid sample metrics (rank later)
            bid_over_c = (float(b) / float(C)) if C > 1e-9 else 0.0
            mc_over_c = (float(m) / float(C)) if C > 1e-9 else 0.0
            margin = (float(b) / float(m) - 1.0) if m > 1e-9 else 0.0

            self._dbg_record_bid_sample(
                trade=tk, pr=float(pr), C=float(C), mc=float(m),
                mc_over_c=float(mc_over_c), bid=float(b), bid_over_c=float(bid_over_c),
                margin=float(margin), undercut=int(undercut_used)
            )

        # placeholder: summary
        self._last_round_bid_n = int(len(bids))

        if bids:
            bs = [float(b.amount) for b in bids]
            bs_sorted = sorted(bs)
            force_log(
                f"[BIDS] n={len(bs)} min={min(bs):.2f} med={bs_sorted[len(bs_sorted)//2]:.2f} "
                f"max={max(bs):.2f} s={s:.2f} p_sel={p_select:.2f} "
                f"markup={markup:.3f} safety={safety:.3f} undercut={undercut_eps:.3f}"
            )
        else:
            force_log(f"[BIDS] n=0 s={s:.2f} (no feasible candidates after filters)")

        # Print bounded bid-skip samples
        for rec in getattr(self, "_dbg_bid_skip_samples", []):
            r = rec.get("reason")
            if r == "bad_mc":
                force_log(f"[BID_SKIP] reason=bad_mc trade={rec.get('trade')} pr={float(rec.get('pr',0.0)):.2f} mc={float(rec.get('mc',0.0)):.1f}")
            elif r == "mc_gt_exp_pay":
                force_log(
                    f"[BID_SKIP] reason=mc_gt_exp_pay trade={rec.get('trade')} pr={float(rec.get('pr',0.0)):.2f} "
                    f"C={float(rec.get('C',0.0)):.1f} mc={float(rec.get('mc',0.0)):.1f} exp_pay={float(rec.get('exp_pay',0.0)):.1f}"
                )
            elif r == "mc_over_abs_guard":
                force_log(
                    f"[BID_SKIP] reason=mc_over_abs_guard trade={rec.get('trade')} pr={float(rec.get('pr',0.0)):.2f} "
                    f"C={float(rec.get('C',0.0)):.1f} mc={float(rec.get('mc',0.0)):.1f} mc/C={float(rec.get('mc_over_abs',0.0)):.2f} exp_ratio={float(rec.get('exp_ratio',0.0)):.2f}"
                )
            else:
                force_log(
                    f"[BID_SKIP] reason={r} trade={rec.get('trade')} pr={float(rec.get('pr',0.0)):.2f} "
                    f"C={float(rec.get('C',0.0)):.1f} Cv={float(rec.get('Cv',0.0)):.1f} "
                    f"mc={float(rec.get('mc',0.0)):.1f}"
                )


        # placeholder: bid sample
        bid_samples = list(getattr(self, "_dbg_bid_samples", []))
        bid_samples.sort(key=lambda d: (float(d.get("bid_over_c", 0.0)), float(d.get("margin", 0.0))))
        for rec in bid_samples[:int(self.DBG_MAX_BID_SAMPLE)]:
            force_log(
                f"[BID_SAMPLE] trade={rec.get('trade')} pr={float(rec.get('pr',0.0)):.2f} "
                f"C={float(rec.get('C',0.0)):.1f} mc={float(rec.get('mc',0.0)):.1f} mc/C={float(rec.get('mc_over_c',0.0)):.2f} "
                f"bid={float(rec.get('bid',0.0)):.1f} bid/C={float(rec.get('bid_over_c',0.0)):.2f} "
                f"margin={float(rec.get('margin',0.0)):.3f} undercut={int(rec.get('undercut',0))}"
            )

        if self.FORCE_LOG and (self._round_idx % self.LOG_EVERY_K_ROUNDS == 0):
            pr_med2 = statistics.median([float(p.get(self._tkey(t), 0.0)) for t in cand]) if cand else 0.0
            mc_over_c2 = []
            for t in cand:
                tk = self._tkey(t)
                C2 = self.direct_cost_proxy(t)
                if C2 > 1e-9 and tk in mc:
                    mc_over_c2.append(float(mc[tk]) / C2)
            mcoc_med = statistics.median(mc_over_c2) if mc_over_c2 else 0.0
            self._log(
                f"inform round={self._round_idx} trades={len(trades)} cand={len(cand)} "
                f"bids={len(bids)} mkt_ratio={self._market_ratio():.3f} s={s:.2f} "
                f"p_select={p_select:.2f} w_abs={w_abs:.2f} markup={markup:.2f} "
                f"min_profit={min_profit:.2f} safety={safety:.2f} "
                f"bid_scale={getattr(self, '_bid_scale', 1.0):.3f} "
                f"low_ratio={low_ratio:.2f} pr_med={pr_med2:.2f} mc/C_med={mcoc_med:.2f}"
            )

        return bids
    def receive(self, contracts, auction_ledger=None, *args, **kwargs):
        self._ensure_state()
        self._dbg_reset_round(stage="RECEIVE")

        t_now = int(getattr(self, "_round", 0))
        n_contracts = (len(contracts) if contracts else 0)
        force_log(f"[RECEIVE] t={t_now} contracts={n_contracts}")

        trades = []
        payment_by_key: Dict[str, float] = {}
        penalty_by_key: Dict[str, float] = {}

        win_infos = []

        for c in (contracts or []):
            tr = c.trade
            pay = float(c.payment)
            tk = self._tkey(tr)
            pen = float(getattr(c, "penalty", 0.0) or 0.0)

            trades.append(tr)
            payment_by_key[tk] = pay
            penalty_by_key[tk] = pen

            C = float(self.direct_cost_proxy(tr))
            r = (pay / C) if C > 1e-9 else 0.0
            myb = Company1._last_bids.get(tk)

            win_infos.append(
                dict(trade=tk, pay=float(pay), pen=float(pen), C=float(C), r=float(r), my_bid=(float(myb) if myb is not None else None))
            )

        if win_infos:
            win_infos.sort(key=lambda d: float(d.get("r", 0.0)))
            for rec in win_infos[:int(self.DBG_MAX_WIN_SAMPLE)]:
                mb = rec.get("my_bid")
                mb_s = "None" if mb is None else f"{float(mb):.1f}"
                force_log(
                    f"[WIN_DETAIL] trade={rec.get('trade')} pay={float(rec.get('pay',0.0)):.1f} pen={float(rec.get('pen',0.0)):.1f} "
                    f"C={float(rec.get('C',0.0)):.1f} r={float(rec.get('r',0.0)):.2f} my_bid={mb_s}"
                )

        scheduled_count = 0
        if trades:
            score, schedules, scheduled_trades = self._post_schedule_v5(trades, payment_by_key, penalty_by_key)
            scheduled_count = len(scheduled_trades)

            ret = self.apply_schedules(schedules)

            if ret is None:
                force_log("[POST_DROP] apply_schedules returned None")
            elif isinstance(ret, (list, tuple, set)):
                rej = list(ret)
                sample = []
                for x in rej[:5]:
                    if hasattr(x, "trade"):
                        sample.append(self._tkey(x.trade))
                    else:
                        sample.append(self._tkey(x))
                force_log(f"[POST_DROP] rejected={len(rej)} sample={sample}")
            else:
                force_log(f"[POST_DROP] apply_schedules_ret={type(ret).__name__}")

            force_log(f"[POST_V5] won={len(trades)} scheduled={scheduled_count} score={score:.1f}")

        # -------------------- Market: Ledger learning --------------------
        if auction_ledger is None or not isinstance(auction_ledger, dict):
            if self.FORCE_LOG:
                self._log(f"receive wins={len(trades)} scheduled={scheduled_count} (no ledger)")
            self._round = int(getattr(self, "_round", 0)) + 1
            return

        my_name = getattr(self, "name", None)
        if my_name is None:
            self._round = int(getattr(self, "_round", 0)) + 1
            return

        ratios_level: List[float] = []
        ratios_low: List[float] = []
        cost_cache: Dict[str, float] = {}

        exact_n = 0
        left_n = 0

        old_mkt = float(getattr(self, "_mkt_ratio_ewma", 1.0))
        old_low = float(getattr(self, "_low_ratio_ewma", 1.25))

        for company_name, contract_list in auction_ledger.items():
            if not isinstance(contract_list, (list, tuple)):
                continue
            is_me = (company_name == my_name)

            for c in contract_list:
                tr = c.trade
                payment = float(c.payment)
                tk = self._tkey(tr)

                C = cost_cache.get(tk)
                if C is None:
                    C = float(self.direct_cost_proxy(tr))
                    cost_cache[tk] = C
                if C <= 1e-9:
                    continue

                r = payment / C
                ratios_level.append(float(r))

                if is_me:
                    self._F_ratio.update_exact(r, weight=1.0)
                    exact_n += 1
                    if r < 5.0:
                        ratios_low.append(float(r))
                else:
                    myb = Company1._last_bids.get(tk)
                    if myb is None:
                        continue
                    u = min(float(myb), float(payment))
                    self._F_ratio.update_left_censored(u / C, weight=1.0)
                    left_n += 1

        # EWMAs
        if ratios_level:
            med = float(statistics.median(ratios_level))
            self._mkt_ratio_ewma = 0.90 * float(self._mkt_ratio_ewma) + 0.10 * float(med)

        if ratios_low:
            med_low = float(statistics.median(ratios_low))
            self._low_ratio_ewma = 0.85 * float(self._low_ratio_ewma) + 0.15 * float(med_low)

        # placeholder : ledgers
        if ratios_level:
            xs = sorted(ratios_level)
            nxs = len(xs)
            low20 = xs[int(0.20 * (nxs - 1))]
            med = xs[nxs // 2]
            q50 = float(self._F_ratio.quantile(0.50))
            force_log(
                f"[LEDGER] exact={exact_n} left={left_n} samples={len(ratios_level)} low20={low20:.2f} med={med:.2f} "
                f"mkt_ewma {old_mkt:.2f}->{float(self._mkt_ratio_ewma):.2f} low_ewma {old_low:.2f}->{float(self._low_ratio_ewma):.2f} "
                f"F_q50={q50:.2f}"
            )
        else:
            force_log(f"[LEDGER] exact={exact_n} left={left_n} samples=0 mkt_ewma {old_mkt:.2f}->{float(self._mkt_ratio_ewma):.2f}")

        # -------------------- !! Bid scale control !! --------------------
        bid_n = int(getattr(self, "_last_round_bid_n", 0))
        win_n = int(len(trades))

        self._hist_bids.append(bid_n)
        self._hist_wins.append(win_n)
        H = 20
        if len(self._hist_bids) > H:
            self._hist_bids = self._hist_bids[-H:]
        if len(self._hist_wins) > H:
            self._hist_wins = self._hist_wins[-H:]

        sum_b = int(sum(self._hist_bids)) if self._hist_bids else 0
        sum_w = int(sum(self._hist_wins)) if self._hist_wins else 0

        if sum_b > 0:
            win_rate = float(sum_w) / float(max(1, sum_b))
            s_now = float(self.market_strength())
            win_target = 0.25 + 0.65 * s_now

            if (sum_b >= 3) and (sum_w == 0):
                self._bid_scale *= 0.95
            elif win_rate < 0.85 * win_target:
                self._bid_scale *= 0.985
            elif win_rate > min(0.98, win_target + 0.15):
                self._bid_scale *= 1.010

            lower = 0.85 if ((sum_b >= 3) and (sum_w == 0)) else 0.90
            self._bid_scale = float(min(1.12, max(lower, self._bid_scale)))

            if self.FORCE_LOG:
                force_log(
                    f"ctrl win_rate={win_rate:.3f} target={win_target:.3f} "
                    f"bid_scale={self._bid_scale:.3f} (hist_b={sum_b}, hist_w={sum_w})"
                )

        if self.FORCE_LOG:
            force_log(
                f"receive wins={len(trades)} scheduled={scheduled_count} "
                f"mkt_ratio_ewma={self._mkt_ratio_ewma:.3f} low_ratio_ewma={self._low_ratio_ewma:.3f} "
                f"ledger_samples={len(ratios_level)}"
            )

        self._round = int(getattr(self, "_round", 0)) + 1

    # -------------------------Post scheduling -------------------------
    def _rebuild_from_order(self, order, payment_by_key, penalty_by_key, tail_only, risk_w):
        vessels = self._vessels()
        schedules = {}
        scheduled_trades = []
        total_score = 0.0

        def get_sched(v):
            return schedules.get(v, v.schedule)

        for tr in order:
            tk = self._tkey(tr)
            pay = float(payment_by_key.get(tk, 0.0))
            pen = float(penalty_by_key.get(tk, 0.0))
            C = float(self.direct_cost_proxy(tr))

            best = None  # (delta, v, cand, mc)
            for v in vessels:
                base = get_sched(v)
                if tail_only:
                    cand, mc = self._best_insertion_tail(v, base, tr, max_ip=int(self.POST_MAX_IP), C=C)
                else:
                    cand, mc = self._best_insertion(v, base, tr, max_ip=int(self.POST_MAX_IP), C=C)

                if cand is None or not math.isfinite(mc):
                    continue

                mc_over_c = (float(mc) / C) if C > 1e-9 else 0.0

                delta = (pay + pen - float(mc))
                delta -= risk_w * float(mc) * max(0.0, mc_over_c - 1.0)

                if best is None or delta > best[0]:
                    best = (delta, v, cand, float(mc))

            if best is None:
                continue

            delta, v, cand, mc = best
            schedules[v] = cand
            scheduled_trades.append(tr)
            total_score += float(delta)

        return float(total_score), schedules, scheduled_trades

    def _post_schedule_v5(self, trades, payment_by_key, penalty_by_key):
        s = float(self.market_strength())
        risk_w = 0.75 if s < 0.60 else 0.20

        trades = list(trades) if trades else []
        if not trades:
            return 0.0, {}, []

        M = max(6, int(getattr(self, "POST_M", 8)))
        best = None  # (n_scheduled, score, schedules, scheduled_trades)

        for _ in range(M):
            order = trades[:]
            self._rng.shuffle(order)

            order.sort(
                key=lambda tr: float(payment_by_key.get(self._tkey(tr), 0.0))
                + float(penalty_by_key.get(self._tkey(tr), 0.0))
                - 0.6 * self.direct_cost_proxy(tr),
                reverse=True,
            )
            score1, sch1, sel1 = self._rebuild_from_order(
                order, payment_by_key, penalty_by_key, tail_only=True, risk_w=risk_w
            )

            remaining = [tr for tr in trades if tr not in sel1]
            if remaining:
                order2 = sel1 + sorted(
                    remaining,
                    key=lambda tr: float(payment_by_key.get(self._tkey(tr), 0.0))
                    + float(penalty_by_key.get(self._tkey(tr), 0.0))
                    - 0.4 * self.direct_cost_proxy(tr),
                    reverse=True,
                )
                score2, sch2, sel2 = self._rebuild_from_order(
                    order2, payment_by_key, penalty_by_key, tail_only=False, risk_w=risk_w
                )
            else:
                score2, sch2, sel2 = score1, sch1, sel1

            score_best, sch_best, sel_best = score2, sch2, sel2

            for _iter in range(3):
                if not sel_best:
                    break
                rem = [tr for tr in trades if tr not in sel_best]
                if not rem:
                    break

                sel_sc = []
                for tr in sel_best:
                    tk = self._tkey(tr)
                    surplus = float(payment_by_key.get(tk, 0.0)) + float(penalty_by_key.get(tk, 0.0)) - self.direct_cost_proxy(tr)
                    sel_sc.append((surplus, tr))
                sel_sc.sort(key=lambda x: x[0])
                worst = [tr for _, tr in sel_sc[:2]]

                rem_sc = []
                for tr in rem:
                    tk = self._tkey(tr)
                    surplus = float(payment_by_key.get(tk, 0.0)) + float(penalty_by_key.get(tk, 0.0)) - 0.5 * self.direct_cost_proxy(tr)
                    rem_sc.append((surplus, tr))
                rem_sc.sort(key=lambda x: x[0], reverse=True)
                best_add = [tr for _, tr in rem_sc[:3]]

                improved = False
                for drop in worst:
                    for add in best_add:
                        trial = [tr for tr in sel_best if tr is not drop] + [add]
                        trial.sort(
                            key=lambda tr: float(payment_by_key.get(self._tkey(tr), 0.0))
                            + float(penalty_by_key.get(self._tkey(tr), 0.0))
                            - 0.5 * self.direct_cost_proxy(tr),
                            reverse=True,
                        )
                        sc, sch, sel = self._rebuild_from_order(
                            trial, payment_by_key, penalty_by_key, tail_only=False, risk_w=risk_w
                        )
                        if (sc > score_best + 1e-6) or (
                            len(sel) > len(sel_best)
                            and sc >= score_best - 0.02 * max(1.0, abs(score_best))
                        ):
                            score_best, sch_best, sel_best = sc, sch, sel
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break

            cand = (len(sel_best), score_best, sch_best, sel_best)
            if best is None or cand[:2] > best[:2]:
                best = cand

        if best is None:
            return 0.0, {}, []
        _, score, schedules, scheduled_trades = best
        return float(score), schedules, list(scheduled_trades)