"""Движок правил: факты по сегментам + шкала людей + яркость -> нарушения с таймкодом и обоснованием.
Каждое правило опирается только на наблюдаемые факты; описание строится шаблоном из этих фактов."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .facts import Segment
from .persons import PersonTimeline
from .video import VideoFrames

log = logging.getLogger(__name__)


@dataclass
class Violation:
    type: str
    t: float
    description: str
    confidence: float = 0.8
    evidence_ts: list[float] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def to_dict(self):
        return {"type": self.type, "t": self.t, "description": self.description, "confidence": self.confidence,
                "evidence_ts": self.evidence_ts, "facts": self.facts}


@dataclass
class Context:
    vf: VideoFrames
    segs: list[Segment]
    timeline: PersonTimeline | None
    params: dict = field(default_factory=dict)
    at_desk: list[bool] | None = None   # по кадрам шкалы: есть ли человек у стола (из детектора + ROI стола)
    deviations: list[dict] = field(default_factory=list)  # подтверждённые вторым проходом отклонения
    leave_checks: dict[float, dict] = field(default_factory=dict)  # t_empty -> точечная проверка рабочего места
    safe_checks: dict[int, dict] = field(default_factory=dict)     # индекс сегмента -> проверка кропа сейфа
    desk_checks: dict[int, dict] = field(default_factory=dict)     # индекс сегмента -> проверка кропа стола
    id_checks: dict[float, dict] = field(default_factory=dict)     # начало визита -> проверка предъявления документа
    count_checks: dict[int, dict] = field(default_factory=dict)    # индекс сегмента -> проверка пересчёта (машинка/руки)
    change_checks: dict[float, dict] = field(default_factory=dict)  # начало следующего визита -> проверка смены клиента
    door_checks: list[dict] = field(default_factory=list)          # проверки двери после входа кассира: [{t, door_open}]
    left_checks: dict[float, dict] = field(default_factory=dict)    # конец визита -> проверка «клиент ушёл до пересчёта»
    scene_rois: dict = field(default_factory=dict)
    machine_active: list[tuple[float, float]] = field(default_factory=list)  # интервалы активности в зоне машинки (пиксели)

    # --- helпер по фактам ---
    def f(self, seg: Segment, key, default=None):
        v = seg.facts.get(key)
        return default if v is None else v

    def acts(self, seg: Segment) -> list[str]:
        return list(self.f(seg, "cashier_actions", []) or [])

    def seg_at(self, t: float) -> Segment | None:
        for s in self.segs:
            if s.t0 <= t < s.t1:
                return s
        return self.segs[-1] if self.segs and t >= self.segs[-1].t1 else None

    def seg_index(self, seg: Segment) -> int:
        return self.segs.index(seg)

    # --- шкала людей ---
    def count_at(self, t: float) -> int | None:
        return self.timeline.count_at(t) if self.timeline else None

    def intervals(self, pred) -> list[tuple[float, float]]:
        """Интервалы, где pred(count) истинно (по сглаженной шкале)."""
        if not self.timeline:
            return []
        out = []
        start = None
        ts = self.timeline.t
        for i, c in enumerate(self.timeline.count):
            ok = pred(c)
            if ok and start is None:
                start = ts[i]
            if not ok and start is not None:
                out.append((start, ts[i]))
                start = None
        if start is not None:
            out.append((start, ts[-1] + 0.5))
        return out

    def visits(self, min_len: float = 1.5) -> list[tuple[float, float]]:
        """Интервалы присутствия посетителя (людей >= 2)."""
        iv = self.intervals(lambda c: c >= 2)
        # сливаем короткие разрывы (< 1.5 с)
        merged = []
        for a, b in iv:
            if merged and a - merged[-1][1] < 1.5:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        return [(a, b) for a, b in merged if b - a >= min_len]

    def empty_intervals(self, min_len: float = 2.0) -> list[tuple[float, float]]:
        iv = self.intervals(lambda c: c == 0)
        end = self.vf.duration - 0.6
        # интервал, доходящий до конца записи, засчитываем уже от 1 с (кассир ушёл перед самым концом видео)
        return [(a, b) for a, b in iv if b - a >= min_len or (b >= end and b - a >= 1.0)]

    def lights_off_time(self) -> float | None:
        br = np.asarray([f.brightness for f in self.vf.frames])
        if len(br) < 6:
            return None
        base = float(np.median(br[: max(3, len(br) // 3)]))
        thr = self.params.get("lights_off_ratio", 0.78) * base
        for i in range(1, len(br)):
            if br[i] < thr and (i + 2 >= len(br) or br[i + 1] < thr):
                return self.vf.frames[i].t
        return None

    def machine_used_near(self, t0: float, t1: float, pad: float = 15.0) -> bool:
        return any(b >= t0 - pad and a <= t1 + pad for a, b in self.machine_active)

    def person_near(self, roi, t: float, window: float = 1.5, frac: float = 0.1) -> bool:
        """Есть ли человек, чей бокс пересекает ROI, в окне ±window вокруг t (по детектору)."""
        if not self.timeline or not roi:
            return False
        for tt, dets in zip(self.timeline.t, self.timeline.dets):
            if abs(tt - t) > window:
                continue
            for d in dets:
                ix = max(0.0, min(d.box[2], roi[2]) - max(d.box[0], roi[0]))
                iy = max(0.0, min(d.box[3], roi[3]) - max(d.box[1], roi[1]))
                if ix * iy >= frac * (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]):
                    return True
        return False

    def leave_time(self, t_empty: float) -> float:
        """Момент, когда кассир отошёл от стола перед тем, как помещение опустело в t_empty."""
        # 1) время, названное VLM по подписям кадров
        cands = [s.facts.get("cashier_leaves_desk_t") for s in self.segs
                 if s.facts.get("cashier_leaves_desk_t") is not None and s.t0 <= t_empty and s.t1 >= t_empty - 20.0]
        cands = [c for c in cands if t_empty - 20.0 <= c <= t_empty + 0.5]
        if cands:
            return round(min(t_empty, min(cands)), 1)
        if not self.at_desk or not self.timeline:
            return t_empty
        ts = self.timeline.t
        # конец последнего длинного (>= 2 с) периода присутствия у стола перед t_empty
        runs = []
        start = None
        for i, t in enumerate(ts):
            if t > t_empty:
                break
            if self.at_desk[i] and start is None:
                start = t
            if not self.at_desk[i] and start is not None:
                runs.append((start, t))
                start = None
        if start is not None:
            runs.append((start, t_empty))
        long_runs = [r for r in runs if r[1] - r[0] >= 2.0]
        if not long_runs:
            return t_empty
        last_end = long_runs[-1][1]
        if t_empty - last_end > 20.0:
            return t_empty
        return round(min(t_empty, last_end + 0.5), 1)

    def segs_in(self, t0: float, t1: float) -> list[Segment]:
        return [s for s in self.segs if s.t1 > t0 and s.t0 < t1]


def _ev(t: float, dur: float, before: float = 1.0, after: float = 1.0) -> list[float]:
    ts = [t, t + after, t - before]
    return [round(max(0.0, min(dur - 0.1, x)), 1) for x in ts]


# ---------------------------------------------------------------- правила

def r_stranger_in_room(cx: Context) -> list[Violation]:
    out = []
    iv = cx.intervals(lambda c: c >= 3)
    for a, b in iv:
        if b - a < cx.params.get("stranger_min_len", 1.5):
            continue
        segs = cx.segs_in(a, b)
        vlm_ok = any((cx.f(s, "people_count_max", 0) or 0) >= 3 for s in segs)
        if not vlm_ok and cx.params.get("stranger_need_vlm", True):
            continue
        t = round(a + 0.5, 1)
        out.append(Violation(
            "stranger_in_room", t,
            f"С {a:.1f} с в кассовом помещении одновременно находятся три и более человек: кроме кассира и "
            f"обслуживаемого клиента в кассу вошёл ещё один человек. Следующий посетитель должен входить после "
            f"ухода предыдущего.",
            0.8, _ev(t, cx.vf.duration), {"people_interval": [a, b]}))
        break
    return out


def r_door_not_closed(cx: Context) -> list[Violation]:
    out = []
    segs = cx.segs
    # точечные проверки кропа двери после входа кассира: дверь открыта в обеих проверках (+3 с и +6 с)
    dc = [c for c in cx.door_checks if c and isinstance(c.get("door_open"), bool)]
    # дверь открыта в двух последних валидных проверках (через ~6 и ~9 с после входа): кассир её не закрыл
    if len(dc) >= 2 and all(c.get("door_open") for c in dc[-2:]):
        t = round(float(dc[0]["t"]), 1)
        return [Violation("door_not_closed", t,
                          "Кассир вошёл в кассу и не закрыл за собой дверь: дверь остаётся открытой и через несколько "
                          "секунд после входа.", 0.7, _ev(t, cx.vf.duration, 0.5, 3.0),
                          {"door_checks": dc})]
    for i, s in enumerate(segs):
        if not cx.f(s, "door_open_end"):
            continue
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        long_enough = (s.t1 - s.t0) >= 6.0 and i == 0
        confirmed = (nxt is not None and cx.f(nxt, "door_open_end")) or (long_enough and cx.f(s, "cashier_at_desk_end") and (cx.count_at(s.t1 - 0.5) or 0) <= 1)
        if not confirmed:
            continue
        t = round(min(s.t0 + 3.0, s.t1), 1)
        early = s.t0 <= 3.0 or i == 0
        typ = "door_not_closed" if early or (cx.count_at(s.t0) or 0) <= 1 else "door_open_during_operations"
        desc = ("Кассир вошёл в кассу и не закрыл за собой дверь: дверь остаётся открытой в последующих кадрах."
                if typ == "door_not_closed" else
                "Дверь кассы остаётся открытой во время работы кассира с клиентом/наличными.")
        out.append(Violation(typ, t, desc, 0.7, _ev(t, cx.vf.duration, 0.5, 2.0),
                             {"segment": [s.t0, s.t1]}))
        break
    return out


def r_leave_checks(cx: Context) -> list[Violation]:
    """Уход кассира из кассы: ценности на столе, ноутбук не заблокирован, ключ в сейфе."""
    out = []
    for a, b in cx.empty_intervals(cx.params.get("empty_min_len", 2.0)):
        if a < 1.0:
            continue  # касса пуста с самого начала записи — кассир ещё не пришёл
        present = [(p0, p1) for p0, p1 in cx.intervals(lambda c: c >= 1) if p1 <= a + 0.01]
        if not present or (present[-1][1] - present[-1][0]) < 3.0:
            continue  # перед «уходом» кассир не был в кассе хотя бы 3 с (мелькнул в дверях)
        seg_after = cx.seg_at(a + 0.5) or cx.seg_at(a)
        seg_before = cx.seg_at(max(0.0, a - 1.0))
        cands = [s for s in (seg_after, seg_before) if s]
        t = cx.leave_time(a)
        lc = cx.leave_checks.get(a)
        if lc:
            # точечная проверка кропов надёжнее сегментных фактов — используем её как источник истины
            if lc.get("money_on_desk"):
                out.append(Violation("valuables_left_on_leave", t,
                                     "Кассир покинул кассу (в помещении никого нет), при этом на столе остались наличные деньги. "
                                     "Ранее принятые деньги должны храниться в запираемом устройстве (сейф, шкаф, ящик).",
                                     0.8, _ev(t, cx.vf.duration, 1.0, 1.5), {"empty_from": a, "check": lc}))
            elif lc.get("bag_on_desk"):
                out.append(Violation("valuables_left_on_leave", t,
                                     "Кассир покинул кассу, оставив на столе инкассаторскую сумку с ценностями.",
                                     0.75, _ev(t, cx.vf.duration, 1.0, 1.5), {"empty_from": a, "check": lc}))
            elif lc.get("documents_on_desk"):
                t_off = cx.lights_off_time()
                if t_off is None or t_off > a + 5.0:  # не конец смены (свет не выключается после ухода)
                    out.append(Violation("valuables_left_on_leave", t,
                                         "Кассир покинул кассу, оставив на столе документы/предметы операции.",
                                         0.55, _ev(t, cx.vf.duration, 1.0, 1.5), {"empty_from": a, "check": lc}))
            if lc.get("laptop_screen") == "work":
                out.append(Violation("laptop_not_locked", t,
                                     "Кассир покинул кассу, а экран ноутбука остался включённым с открытым рабочим окном "
                                     "(не заблокирован).",
                                     0.7, _ev(t, cx.vf.duration, 1.0, 1.5), {"empty_from": a, "check": lc}))
            if lc.get("key_in_safe_lock"):
                out.append(Violation("safe_key_left", t, "Кассир покинул кассу, а ключ остался в замке сейфа.",
                                     0.65, _ev(t, cx.vf.duration, 1.0, 1.5), {"empty_from": a, "check": lc}))
            if lc.get("safe_door_open"):
                out.append(Violation("safe_open_during_work", t, "Кассир покинул кассу, оставив дверцу сейфа открытой.",
                                     0.7, _ev(t, cx.vf.duration, 1.0, 1.5), {"empty_from": a, "check": lc}))
            break
        if any(cx.f(s, "money_on_desk_end") for s in cands if s.t1 >= a - 0.5):
            out.append(Violation("valuables_left_on_leave", t,
                                 "Кассир покинул кассу (в помещении никого нет), при этом на столе остались наличные деньги. "
                                 "Ранее принятые деньги должны храниться в запираемом устройстве (сейф, шкаф, ящик).",
                                 0.8, _ev(t, cx.vf.duration, 1.0, 1.5), {"empty_from": a}))
        elif any((cx.f(s, "bag_on_desk_end") or cx.f(s, "collector_bag_any")) for s in cands if s.t1 >= a - 0.5):
            out.append(Violation("valuables_left_on_leave", t,
                                 "Кассир покинул кассу, оставив на столе инкассаторскую сумку с ценностями.",
                                 0.75, _ev(t, cx.vf.duration, 1.0, 1.5), {"empty_from": a}))
        if any(cx.f(s, "laptop_screen_end") == "work" for s in cands if s.t1 >= a - 0.5):
            out.append(Violation("laptop_not_locked", t,
                                 "Кассир покинул кассу, а экран ноутбука остался включённым с открытым рабочим окном "
                                 "(не заблокирован).",
                                 0.7, _ev(t, cx.vf.duration, 1.0, 1.5), {"empty_from": a}))
        if any(cx.f(s, "key_in_safe_lock_end") for s in cands if s.t1 >= a - 0.5):
            out.append(Violation("safe_key_left", t,
                                 "Кассир покинул кассу, а ключ остался в замке сейфа.",
                                 0.7, _ev(t, cx.vf.duration, 1.0, 1.5), {"empty_from": a}))
        break
    return out


def r_safe_open_during_work(cx: Context) -> list[Violation]:
    out = []
    segs = cx.segs
    for i, s in enumerate(segs):
        if cx.f(s, "safe_door_open_end") and cx.f(s, "cashier_at_desk_end"):
            chk = cx.safe_checks.get(i)
            if chk is None:
                continue  # без точечной проверки кропа не утверждаем
            if not chk.get("safe_door_open") or chk.get("cashier_handling_safe"):
                continue  # кроп не подтвердил, либо кассир прямо сейчас работает с сейфом
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            prv = segs[i - 1] if i > 0 else None
            # сейф должен оставаться открытым продолжительно: открыт на концах двух соседних сегментов
            confirmed = (nxt is not None and cx.f(nxt, "safe_door_open_end")) or (prv is not None and cx.f(prv, "safe_door_open_end"))
            if not confirmed:
                continue
            # момент нарушения — когда кассир уже за столом при открытом сейфе: кадр точечной проверки (конец сегмента)
            t = round(float(chk.get("t", s.t1 - 0.5)), 1)
            out.append(Violation("safe_open_during_work", t,
                                 "Дверца сейфа остаётся открытой, пока кассир работает за столом. Наличные деньги в сейфе "
                                 "при этом не защищены.",
                                 0.7, _ev(t, cx.vf.duration, 0.5, 2.0), {"segment": [s.t0, s.t1]}))
            break
    return out


def r_money_before_next_client(cx: Context) -> list[Violation]:
    out = []
    visits = cx.visits()
    for k in range(1, len(visits)):
        prev_end = visits[k - 1][1]
        start = visits[k][0]
        if start - prev_end < cx.params.get("client_change_min_gap", 1.5):
            continue
        between = cx.segs_in(prev_end, start)
        seg_prev = cx.seg_at(max(0.0, start - 0.5))
        money = any(cx.f(s, "money_on_desk_end") for s in between) or (seg_prev and cx.f(seg_prev, "money_on_desk_end"))
        secured = any("at_safe" in cx.acts(s) or cx.f(s, "safe_door_open_any") for s in between)
        chk = cx.change_checks.get(start)
        if chk is None:
            continue  # без подтверждения смены клиента не утверждаем
        if not chk.get("visitor_changed") or chk.get("cashier_put_money_in_safe"):
            continue
        # деньги на столе в момент смены клиента: по окну смены или по кропу стола при входе следующего
        money = bool(chk.get("money_on_desk_when_cashier_alone")) or bool((chk.get("desk_check") or {}).get("money_on_desk"))
        if money and not secured:
            t = round(start, 1)
            out.append(Violation("money_not_secured_before_next_client", t,
                                 "В кассу входит следующий клиент, а наличные деньги предыдущего клиента всё ещё лежат на столе: "
                                 "между клиентами сейф не открывался и деньги не убирались.",
                                 0.75, _ev(t, cx.vf.duration, 1.5, 1.0), {"prev_visit_end": prev_end, "next_visit_start": start}))
            break
    return out


def r_client_left_before_count(cx: Context) -> list[Violation]:
    out = []
    for a, b in cx.visits():
        if b >= cx.vf.duration - 1.0:
            continue
        after = cx.segs_in(b, b + 6.0)
        during = cx.segs_in(a, b)
        handed = any("receiving_cash" in cx.acts(s) or cx.f(s, "money_on_desk_any") for s in during)
        counting_after = any(("counting_machine" in cx.acts(s) or "counting_hand" in cx.acts(s)) and cx.f(s, "cashier_at_desk_end", True) for s in after)
        chk = cx.left_checks.get(b)
        if chk is not None:
            act_after = str(chk.get("cashier_action_after") or "").lower()
            counting_confirmed = chk.get("cashier_counting_after") and act_after.startswith("counting")
            if chk.get("client_left") and counting_confirmed and handed:
                t = round(float(chk.get("client_left_time") or b), 1)
                out.append(Violation("client_left_before_count", t,
                                     "Клиент передал наличные и покинул кассу, а кассир продолжает пересчёт уже без клиента. "
                                     "Приём и пересчёт наличных должны проходить в присутствии клиента.",
                                     0.7, _ev(t, cx.vf.duration, 1.0, 2.0), {"visit": [a, b], "check": chk}))
                break
            continue
        if handed and counting_after and (cx.count_at(b + 1.0) or 0) == 1:
            t = round(b, 1)
            out.append(Violation("client_left_before_count", t,
                                 "Клиент передал наличные и покинул кассу, а кассир продолжает пересчёт уже без клиента. "
                                 "Приём и пересчёт наличных должны проходить в присутствии клиента.",
                                 0.7, _ev(t, cx.vf.duration, 1.0, 2.0), {"visit": [a, b]}))
            break
    return out


def r_collector_bag_no_id(cx: Context) -> list[Violation]:
    out = []
    for a, b in cx.visits():
        if b - a > cx.params.get("collector_max_visit", 12.0):
            continue
        segs = cx.segs_in(a - 1.0, b + 2.0)
        bag = any((cx.f(s, "bag_on_desk_end") or cx.f(s, "collector_bag_any")) for s in segs)
        doc = any(cx.f(s, "document_shown_any") or "checking_document" in cx.acts(s) for s in segs)
        if bag and not doc:
            t = round((a + b) / 2, 1)
            out.append(Violation("collector_bag_no_id_check", t,
                                 f"Короткий визит ({b - a:.0f} с): инкассаторская сумка принята сразу, проверка удостоверения "
                                 f"и сопроводительных документов на видео не наблюдается.",
                                 0.65, _ev(t, cx.vf.duration, 1.5, 1.5), {"visit": [a, b]}))
            break
    return out


def r_no_client_id_check(cx: Context) -> list[Violation]:
    out = []
    for a, b in cx.visits():
        segs = cx.segs_in(a - 1.0, b)
        if not segs:
            continue
        # первый сегмент с передачей наличных
        first_cash = None
        for s in segs:
            if "receiving_cash" in cx.acts(s) or cx.f(s, "money_on_desk_any"):
                first_cash = s
                break
        if first_cash is None:
            continue
        bag = any((cx.f(s, "bag_on_desk_end") or cx.f(s, "collector_bag_any")) for s in segs)
        if bag:
            continue
        chk = cx.id_checks.get(a)
        if chk is None:
            continue  # без точечной проверки начала визита не утверждаем
        if chk.get("id_document_shown"):
            continue  # документ предъявлен; факт передачи наличных уже установлен по сегментам
        # консервативно: сегментные факты тоже не должны видеть документ до первой передачи наличных
        upto = [s for s in segs if s.t0 <= first_cash.t0]
        if any(cx.f(s, "document_shown_any") or "checking_document" in cx.acts(s) for s in upto):
            continue
        if True:
            t = chk.get("cash_time") or round(max(a + 1.0, first_cash.t0), 1)
            t = round(float(t), 1)
            out.append(Violation("no_client_id_check", t,
                                 "Клиент сразу передаёт наличные: передача и проверка документа, удостоверяющего личность, "
                                 "перед операцией на видео не наблюдается.",
                                 0.6, _ev(t, cx.vf.duration, 2.0, 1.0), {"visit": [a, b]}))
            break
    return out


def _machine_used(chk: dict) -> bool:
    """Модель иногда возвращает ключ не 'machine_used', а 'machine' / 'counting_machine_used' — принимаем любой."""
    return any(bool(v) for k, v in chk.items() if "machine" in str(k).lower() and isinstance(v, bool))


def r_no_machine_count(cx: Context) -> list[Violation]:
    out = []
    segs = cx.segs
    n = len(segs)
    for i, s in enumerate(segs):
        if "counting_hand" not in cx.acts(s):
            continue
        # контекст пересчёта: клиент у стола или приём/выдача наличных, кассир остаётся в кассе
        in_service = cx.f(s, "client_at_desk_any") or "receiving_cash" in cx.acts(s) or "handing_cash" in cx.acts(s)
        if not in_service or (cx.f(s, "people_count_end", 1) or 0) == 0 or "away" in cx.acts(s):
            continue
        win = segs[max(0, i - 2): min(n, i + 3)]
        if any("counting_machine" in cx.acts(w) for w in win):
            continue
        if cx.machine_used_near(s.t0, s.t1):
            continue  # по пиксельной активности машинка использовалась рядом по времени
        chk = cx.count_checks.get(i)
        if chk is None or _machine_used(chk) or not chk.get("hand_counting"):
            continue  # точечная проверка не подтвердила ручной пересчёт без машинки
        t = round(chk.get("hand_counting_time") or (s.t0 + 1.0), 1)
        out.append(Violation("no_machine_count", t,
                             "Кассир пересчитывает банкноты вручную, счётная машинка в этом эпизоде не используется. "
                             "Пересчёт должен выполняться на оборудовании с проверкой машиночитаемых защитных признаков.",
                             0.65, _ev(t, cx.vf.duration, 1.0, 2.0), {"segment": [s.t0, s.t1]}))
        return out
    # сумка -> сейф без пересчёта
    bag_seen = sum(1 for s in segs if (cx.f(s, "bag_on_desk_end") or cx.f(s, "collector_bag_any"))) >= 2
    machine = any("counting_machine" in cx.acts(s) for s in segs) or bool(cx.machine_active)
    bag_checks = [c for k, c in cx.count_checks.items() if str(k).startswith("bag")]
    if bag_checks and any(_machine_used(c) for c in bag_checks):
        machine = True
    if bag_seen and not machine and bag_checks:
        for i, s in enumerate(segs):
            prev_bag = any((cx.f(p, "bag_on_desk_end") or cx.f(p, "collector_bag_any")) for p in segs[:i + 1])
            if prev_bag and ("at_safe" in cx.acts(s) or cx.f(s, "safe_door_open_any")):
                t = round(s.t0 + 1.0, 1)
                out.append(Violation("no_machine_count", t,
                                     "Содержимое инкассаторской сумки убирается в сейф без пересчёта на счётной машинке.",
                                     0.55, _ev(t, cx.vf.duration, 1.0, 2.0), {"segment": [s.t0, s.t1]}))
                break
    return out


def _persistent(cx: Context, key: str, min_segs: int = 2, no_client: bool = False, check_key: str | None = None):
    hits = [s for s in cx.segs if cx.f(s, key) and not (no_client and cx.f(s, "client_at_desk_any"))]
    need = min(min_segs, max(1, len(cx.segs) - 1))
    if len(hits) < need:
        return []
    if check_key:
        # подтверждение кропом стола (если проверка была сделана и не подтвердила — отбрасываем)
        checks = [cx.desk_checks[cx.seg_index(s)] for s in hits if cx.seg_index(s) in cx.desk_checks]
        if checks and not any(c.get(check_key) for c in checks):
            return []
    return hits


def _persist_t(hits) -> float:
    """Таймкод для «постоянного» факта: середина второго сегмента-кандидата (в первом объект часто ещё несут)."""
    if len(hits) > 1:
        return round((hits[1].t0 + hits[1].t1) / 2, 1)
    return round(hits[0].t0 + 1.0, 1)


def r_personal_items(cx: Context) -> list[Violation]:
    hits = _persistent(cx, "personal_items_on_desk_any", no_client=True, check_key="personal_items_on_desk")
    if not hits:
        return []
    t = _persist_t(hits)
    return [Violation("personal_items", t,
                      "На кассовом столе находятся личные вещи (сумка / верхняя одежда). Личные вещи работников "
                      "хранятся вне кассового узла.", 0.7, _ev(t, cx.vf.duration, 0.5, 2.0),
                      {"segments": [[s.t0, s.t1] for s in hits]})]


def r_food_drink(cx: Context) -> list[Violation]:
    hits = _persistent(cx, "food_drink_on_desk_any", no_client=True, check_key="food_drink_on_desk")
    if not hits:
        return []
    t = _persist_t(hits)
    return [Violation("food_drink", t,
                      "На кассовом столе еда и/или напитки. Посторонние предметы и личные вещи на рабочем месте "
                      "кассира не допускаются.", 0.7, _ev(t, cx.vf.duration, 0.5, 2.0),
                      {"segments": [[s.t0, s.t1] for s in hits]})]


def r_phone(cx: Context) -> list[Violation]:
    hits = _persistent(cx, "cashier_phone_in_hand_any", 2)
    if not hits:
        return []
    t = round(hits[0].t0 + 1.0, 1)
    return [Violation("phone_with_camera", t,
                      "В кассе используется смартфон (устройство с функцией фото/видеофиксации).", 0.45,
                      _ev(t, cx.vf.duration, 0.5, 2.0), {"segments": [[s.t0, s.t1] for s in hits]})]


def r_devices_left_on(cx: Context) -> list[Violation]:
    t_off = cx.lights_off_time()
    if t_off is None:
        return []
    after = cx.segs_in(t_off, cx.vf.duration)
    if not after:
        return []
    screen = [cx.f(s, "laptop_screen_end") for s in after]
    bright = any(sc in ("work", "lock") for sc in screen)
    if not bright:
        return []
    t = round(t_off + 0.5, 1)
    return [Violation("devices_left_on", t,
                      "Свет в кассе выключен (конец смены), но экран ноутбука продолжает светиться — оборудование "
                      "не выключено и не заблокировано.", 0.7, _ev(t, cx.vf.duration, 0.5, 2.0),
                      {"lights_off_at": t_off, "screen_after": screen})]


def r_cashier_left_client(cx: Context) -> list[Violation]:
    out = []
    segs = cx.segs
    for i, s in enumerate(segs):
        if cx.f(s, "client_at_desk_end") and cx.f(s, "cashier_at_desk_end") is False and "away" in cx.acts(s):
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            if (s.t1 - s.t0) < 3.0 and not (nxt and cx.f(nxt, "client_at_desk_end") and cx.f(nxt, "cashier_at_desk_end") is False):
                continue
            # детектор должен подтвердить: число людей упало до 1 минимум на 2.5 с внутри сегмента (±2 с)
            drop = any(b - a >= 2.5 and b > s.t0 - 2.0 and a < s.t1 + 2.0 for a, b in cx.intervals(lambda c: c == 1))
            if not drop:
                continue
            t = round(s.t0 + 1.0, 1)
            out.append(Violation("cashier_left_during_service", t,
                                 "Клиент находится у стола, а кассир покинул рабочее место, не завершив обслуживание.",
                                 0.6, _ev(t, cx.vf.duration, 1.0, 2.0), {"segment": [s.t0, s.t1]}))
            break
    return out


def r_transfer_outside_procedure(cx: Context) -> list[Violation]:
    out = []
    for a, b in cx.visits():
        segs = cx.segs_in(a, b + 1.0)
        acts = [cx.acts(s) for s in segs]
        got = [i for i, ac in enumerate(acts) if "receiving_cash" in ac or ("counting_machine" in ac) or ("counting_hand" in ac)]
        gave = [i for i, ac in enumerate(acts) if "handing_cash" in ac]
        safe = any("at_safe" in ac for ac in acts) or any(cx.f(s, "safe_door_open_any") for s in segs)
        if got and gave and max(gave) > min(got) and not safe:
            s = segs[max(gave)]
            t = round(s.t0 + 1.0, 1)
            out.append(Violation("transfer_outside_procedure", t,
                                 "После приёма и пересчёта наличные передаются обратно клиенту, при этом сейф не открывался: "
                                 "движение денег не соответствует стандартной процедуре приёма.",
                                 0.45, _ev(t, cx.vf.duration, 1.5, 1.0), {"visit": [a, b]}))
            break
    return out


def r_other_deviation(cx: Context) -> list[Violation]:
    out = []
    for d in cx.deviations:
        if not d.get("confirmed"):
            continue
        t = round(d["t0"] + 1.0, 1)
        out.append(Violation("other_procedure_deviation", t,
                             f"Модель дважды (на разных кадрах) отметила отклонение от процедуры: {d['claim']} "
                             f"Требует проверки человеком.",
                             0.5, _ev(t, cx.vf.duration, 0.5, 2.0), {"claim": d["claim"], "visible": d.get("visible")}))
        break
    return out


ROI_FOR_TYPE = {
    "safe_open_during_work": "safe", "safe_key_left": "safe", "stranger_near_safe": "safe",
    "laptop_not_locked": "laptop", "devices_left_on": "laptop",
    "valuables_left_on_leave": "desk", "personal_items": "desk", "food_drink": "desk",
    "money_not_secured_before_next_client": "desk", "no_client_id_check": "desk", "transfer_outside_procedure": "desk",
    "no_machine_count": "counting_machine", "collector_bag_no_id_check": "desk",
    "door_not_closed": "door", "door_open_during_operations": "door", "stranger_in_room": "door",
}


RULES = [r_other_deviation, r_stranger_in_room, r_door_not_closed, r_leave_checks, r_safe_open_during_work,
         r_money_before_next_client, r_client_left_before_count, r_collector_bag_no_id,
         r_no_client_id_check, r_no_machine_count, r_personal_items, r_food_drink, r_phone,
         r_devices_left_on, r_cashier_left_client, r_transfer_outside_procedure]


def run_rules(cx: Context, min_conf: float = 0.0) -> list[Violation]:
    found: list[Violation] = []
    for rule in RULES:
        try:
            for v in rule(cx):
                if v.confidence >= min_conf:
                    found.append(v)
        except Exception as e:  # правило не должно ронять анализ
            log.exception("rule %s failed: %s", rule.__name__, e)
    # конец смены: «устройства не выключены» поглощает «ноутбук не заблокирован»
    if any(v.type == "devices_left_on" for v in found):
        found = [v for v in found if v.type != "laptop_not_locked"]
    # дедуп по типу (оставляем самое раннее)
    seen = {}
    for v in sorted(found, key=lambda v: v.t):
        if v.type not in seen:
            seen[v.type] = v
    return sorted(seen.values(), key=lambda v: v.t)
