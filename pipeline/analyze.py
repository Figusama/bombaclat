"""Оркестрация анализа одного видео: кадры -> люди -> сцена -> факты -> правила -> результат по схеме API."""
from __future__ import annotations

import logging
import os
import time

from .catalog import get_type
from .facts import CHANGE_PROMPT, COUNT_PROMPT, ID_PROMPT, LEFT_PROMPT, confirm_deviations, extract_facts, leave_check, roi_check, video_hash, window_check
from .motion import active_intervals
from .persons import PersonDetector, build_timeline, desk_presence
from .rules import ROI_FOR_TYPE, Context, run_rules
from .scene import calibrate
from .video import annotate_frame, crop_box, sample_frames, to_jpeg_b64
from .vlm.base import VLMBackend, make_backend

log = logging.getLogger(__name__)

DEFAULTS = {
    "fps": 2.0,
    "seg_len": float(os.environ.get("SEG_LEN", 8.0)),
    "frames_per_seg": int(os.environ.get("FRAMES_PER_SEG", 4)),
    "frame_w": int(os.environ.get("FRAME_W", 416)),
    "max_segs": int(os.environ.get("MAX_SEGS", 16)),
    "adaptive_frames": int(os.environ.get("ADAPTIVE_FRAMES", 48)),
    "min_conf": float(os.environ.get("MIN_CONF", 0.5)),
    "evidence_max_side": 848,
    "use_scene": os.environ.get("USE_SCENE", "1") == "1",
    "use_deviation": os.environ.get("USE_DEVIATION", "1") == "1",
    "use_cache": True,
}


class Analyzer:
    """Держит загруженные модели; analyze() потокобезопасен за счёт lock в VLM."""

    def __init__(self, vlm: VLMBackend | None = None, detector: PersonDetector | None = None):
        self.vlm = vlm or make_backend()
        self.detector = detector or PersonDetector()

    @staticmethod
    def _checks_path(key):
        from .facts import CACHE_DIR, PROMPT_VERSION
        return (CACHE_DIR / f"{key}__checks_{PROMPT_VERSION}.json") if key else None

    def _load_checks(self, key):
        import json
        p = self._checks_path(key)
        if p and p.exists():
            try:
                return json.load(open(p))
            except Exception:
                return {}
        return {}

    def _save_checks(self, key, data):
        import json
        p = self._checks_path(key)
        if p and data:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                json.dump(data, open(p, "w"), ensure_ascii=False)
            except Exception as e:
                log.warning("checks cache save failed: %s", e)

    def analyze(self, path: str, params: dict | None = None, progress=None, cache_key: str | None = None) -> dict:
        p = {**DEFAULTS, **(params or {})}
        t_start = time.time()
        timings = {}

        def report(stage, frac):
            if progress:
                progress(stage, frac)

        report("decode", 0.02)
        vf = sample_frames(path, fps=p["fps"])
        timings["decode"] = round(time.time() - t_start, 1)

        report("scene", 0.08)
        t0 = time.time()
        scene = None
        if p["use_scene"]:
            try:
                ref = vf.at(min(1.0, vf.duration / 2))
                scene = calibrate(self.vlm, ref.image)
            except Exception as e:
                log.warning("scene calibration failed: %s", e)
        timings["scene"] = round(time.time() - t0, 1)

        report("persons", 0.15)
        t0 = time.time()
        timeline = build_timeline(vf, self.detector, room_mask=scene.room_mask() if scene else None)
        timings["persons"] = round(time.time() - t0, 1)

        t0 = time.time()
        key = cache_key if cache_key is not None else (video_hash(path) if p["use_cache"] else None)
        segs = extract_facts(self.vlm, vf, scene, timeline, seg_len=p["seg_len"], frames_per_seg=p["frames_per_seg"],
                             frame_w=p["frame_w"], max_segs=p["max_segs"], cache_key=key, adaptive=p["adaptive_frames"],
                             progress=lambda i, n: report("facts", 0.2 + 0.7 * i / n))
        timings["facts"] = round(time.time() - t0, 1)

        report("confirm", 0.9)
        devs = []
        if p.get("use_deviation", True):
            try:
                devs = confirm_deviations(self.vlm, vf, segs, frame_w=p["frame_w"])
            except Exception as e:
                log.warning("deviation confirmation failed: %s", e)
        report("rules", 0.92)
        checks_cache = self._load_checks(key)

        def cached(name, *args, fn):
            ck = f"{name}:" + ":".join(str(round(a, 1)) if isinstance(a, float) else str(a) for a in args)
            if ck in checks_cache:
                return checks_cache[ck]
            val = fn()
            checks_cache[ck] = val
            return val

        at_desk = desk_presence(timeline, scene.rois["desk"]) if (scene and scene.has("desk")) else None
        machine_active = active_intervals(vf, scene.rois["counting_machine"]) if (scene and scene.has("counting_machine")) else []
        log.info("machine activity intervals: %s", machine_active)
        cx = Context(vf=vf, segs=segs, timeline=timeline, params=p, at_desk=at_desk, deviations=devs,
                     scene_rois=scene.rois if scene else {}, machine_active=machine_active)
        # подтверждение личных вещей / еды кропом стола (первый сегмент-кандидат без клиента)
        for key in ("personal_items_on_desk_any", "food_drink_on_desk_any"):
            hits = [i for i, sg in enumerate(segs) if sg.facts.get(key) and not sg.facts.get("client_at_desk_any")]
            if not hits:
                continue
            i = hits[1] if len(hits) > 1 else hits[0]  # проверяем конец второго сегмента-кандидата
            if i not in cx.desk_checks:
                try:
                    cx.desk_checks[i] = cached("desk", i, segs[i].frames[-1].t, fn=lambda: roi_check(self.vlm, segs[i].frames[-1], scene, ("desk",), frame_w=p["frame_w"]))
                    log.info("desk check seg %d: %s", i, cx.desk_checks[i])
                except Exception as e:
                    log.warning("desk check failed: %s", e)
        # подтверждение «сейф открыт во время работы» по кропу сейфа (не более 2 кандидатов)
        n_checked = 0
        for i, sg in enumerate(segs):
            if n_checked >= 2:
                break
            if sg.facts.get("safe_door_open_end") and sg.facts.get("cashier_at_desk_end"):
                try:
                    cx.safe_checks[i] = cached("safe", i, sg.frames[-1].t, fn=lambda: roi_check(self.vlm, sg.frames[-1], scene, ("safe",), frame_w=p["frame_w"]))
                    log.info("safe check seg %d: %s", i, cx.safe_checks[i])
                    n_checked += 1
                except Exception as e:
                    log.warning("safe check failed: %s", e)
        # проверка предъявления документа в начале визита (первые 2 визита с передачей наличных)
        n_id = 0
        for (a, b) in cx.visits():
            if n_id >= 2:
                break
            vsegs = cx.segs_in(a - 1.0, b)
            cash = any("receiving_cash" in (sg.facts.get("cashier_actions") or []) or sg.facts.get("money_on_desk_any") for sg in vsegs)
            if not cash or any(sg.facts.get("bag_on_desk_end") or sg.facts.get("collector_bag_any") for sg in vsegs):
                continue
            try:
                cx.id_checks[a] = cached("id", a, fn=lambda: window_check(self.vlm, vf, a, a + 9.0, ID_PROMPT, fps=1.0, frame_w=p["frame_w"]))
                log.info("id check visit @%.1f: %s", a, cx.id_checks[a])
                n_id += 1
            except Exception as e:
                log.warning("id check failed: %s", e)
        # проверка ручного пересчёта без машинки (первые 2 сегмента-кандидата)
        n_cnt = 0
        for i, sg in enumerate(segs):
            if n_cnt >= 2:
                break
            acts = sg.facts.get("cashier_actions") or []
            if "counting_hand" not in acts:
                continue
            in_service = sg.facts.get("client_at_desk_any") or "receiving_cash" in acts or "handing_cash" in acts
            if not in_service or (sg.facts.get("people_count_end") or 0) == 0 or "away" in acts:
                continue
            win = segs[max(0, i - 2): min(len(segs), i + 3)]
            if any("counting_machine" in (w.facts.get("cashier_actions") or []) for w in win):
                continue
            try:
                cx.count_checks[i] = cached("count", i, sg.t0, fn=lambda: window_check(self.vlm, vf, sg.t0 - 2.0, sg.t1 + 2.0, COUNT_PROMPT, fps=1.0, frame_w=p["frame_w"], max_frames=10))
                log.info("count check seg %d: %s", i, cx.count_checks[i])
                n_cnt += 1
            except Exception as e:
                log.warning("count check failed: %s", e)
        # дверь после входа кассира: касса пуста в начале, затем появляется человек -> проверки двери на +3 и +6 с
        if timeline and timeline.count and timeline.count[0] == 0 and scene and scene.has("door"):
            t_in = next((t for t, c in zip(timeline.t, timeline.count) if c >= 1), None)
            if t_in is not None and t_in < vf.duration - 7.0:
                for dt in (3.0, 6.0, 9.0):
                    if t_in + dt > vf.duration - 0.5:
                        break
                    fr = vf.at(t_in + dt)
                    try:
                        cx.door_checks.append(cached("door", round(fr.t, 1), fn=lambda: roi_check(self.vlm, fr, scene, ("door",), frame_w=p["frame_w"])))
                    except Exception as e:
                        log.warning("door check failed: %s", e)
                log.info("door checks after entry @%.1f: %s", t_in, cx.door_checks)
        # клиент ушёл до завершения пересчёта: окно вокруг конца визита (до 2 визитов, не у самого конца видео)
        n_left = 0
        for (a, b) in cx.visits():
            if n_left >= 2 or b >= vf.duration - 1.0:
                continue
            during = cx.segs_in(a, b)
            handed = any("receiving_cash" in (sg.facts.get("cashier_actions") or []) or sg.facts.get("money_on_desk_any") for sg in during)
            if not handed:
                continue
            try:
                cx.left_checks[b] = cached("left2", b, fn=lambda: window_check(self.vlm, vf, b - 5.0, b + 4.0, LEFT_PROMPT, fps=1.0, frame_w=p["frame_w"], max_frames=10))
                log.info("client-left check @%.1f: %s", b, cx.left_checks[b])
                n_left += 1
            except Exception as e:
                log.warning("left check failed: %s", e)
        # смена клиента при деньгах на столе: подтверждаем окном вокруг разрыва между визитами (до 2 проверок)
        vis = cx.visits()
        n_ch = 0
        for k in range(1, len(vis)):
            if n_ch >= 2:
                break
            prev_end, start = vis[k - 1][1], vis[k][0]
            if start - prev_end < 1.5:
                continue
            try:
                cx.change_checks[start] = cached("change", start, fn=lambda: window_check(self.vlm, vf, prev_end - 2.0, start + 4.0, CHANGE_PROMPT, fps=1.0, frame_w=p["frame_w"], max_frames=10))
                if cx.change_checks[start].get("visitor_changed"):
                    # второе мнение о деньгах на столе: кроп стола в момент входа следующего клиента
                    fr = vf.at(max(0.0, start - 0.5))
                    cx.change_checks[start]["desk_check"] = cached("desk_change", round(fr.t, 1), fn=lambda: roi_check(self.vlm, fr, scene, ("desk",), frame_w=p["frame_w"]))
                log.info("client change check @%.1f: %s", start, cx.change_checks[start])
                n_ch += 1
            except Exception as e:
                log.warning("change check failed: %s", e)
        # инкассаторская сумка -> сейф без машинки: подтверждаем окнами «использовалась ли машинка» (до 2 окон)
        bag_segs = [i for i, sg in enumerate(segs) if sg.facts.get("bag_on_desk_end") or sg.facts.get("collector_bag_any")]
        any_machine = any("counting_machine" in (sg.facts.get("cashier_actions") or []) for sg in segs)
        if len(bag_segs) >= 2 and not any_machine:
            safe_idx = [i for i, sg in enumerate(segs) if "at_safe" in (sg.facts.get("cashier_actions") or []) or sg.facts.get("safe_door_open_any")]
            cand = [i for i, sg in enumerate(segs) if sg.facts.get("money_on_desk_any") or "counting_hand" in (sg.facts.get("cashier_actions") or [])]
            if safe_idx:
                cand = [i for i in cand if i <= safe_idx[-1]] or cand
            # пересчёт обычно идёт сразу после появления сумки: берём первые два окна после первого сегмента с сумкой
            cand = [i for i in cand if i >= bag_segs[0]] or cand
            for i in cand[:2]:
                sg = segs[i]
                try:
                    cx.count_checks[f"bag{i}"] = cached("bagcount", i, sg.t0, fn=lambda: window_check(self.vlm, vf, sg.t0, sg.t1, COUNT_PROMPT, fps=1.0, frame_w=p["frame_w"], max_frames=10))
                    log.info("bag/machine check seg %d: %s", i, cx.count_checks[f"bag{i}"])
                except Exception as e:
                    log.warning("bag/machine check failed: %s", e)
        # точечные проверки рабочего места в момент ухода кассира (первые 2 интервала пустой кассы)
        for (a, b) in [iv for iv in cx.empty_intervals(p.get("empty_min_len", 2.0)) if iv[0] >= 1.0][:2]:
            try:
                fr = vf.at(min(a + 1.0, vf.duration - 0.1))
                cx.leave_checks[a] = cached("leave", a, fn=lambda: leave_check(self.vlm, fr, scene, frame_w=p["frame_w"]))
                log.info("leave check @%.1f: %s", a, cx.leave_checks[a])
            except Exception as e:
                log.warning("leave check failed: %s", e)
        viols = run_rules(cx, min_conf=p["min_conf"])

        self._save_checks(key, checks_cache)
        report("render", 0.96)
        out_viols = []
        for v in viols:
            typ = get_type(v.type)
            frames = []
            roi_name = ROI_FOR_TYPE.get(v.type)
            roi = scene.rois.get(roi_name) if (scene and roi_name) else None
            main = vf.at(v.t)
            # 1) полный кадр в момент события с рамкой вокруг зоны
            frames.append({"timestamp_sec": main.t,
                           "image_base64": to_jpeg_b64(annotate_frame(main.image, roi, roi_name, main.t), quality=82,
                                                       max_side=p["evidence_max_side"])})
            # 2) увеличенный кроп зоны в тот же момент
            if roi:
                frames.append({"timestamp_sec": main.t, "zoom": roi_name,
                               "image_base64": to_jpeg_b64(crop_box(main.image, roi, pad=0.3, upscale=2.0, min_size=320),
                                                           quality=85, max_side=700)})
            # 3) кадр через ~1.5 с (контекст)
            ctx = vf.at(min(vf.duration - 0.1, v.t + 1.5))
            if abs(ctx.t - main.t) >= 0.5:
                frames.append({"timestamp_sec": ctx.t,
                               "image_base64": to_jpeg_b64(annotate_frame(ctx.image, roi, roi_name, ctx.t), quality=80,
                                                           max_side=p["evidence_max_side"])})
            norms = typ["norms"]
            out_viols.append({
                "type": typ["name"],
                "type_id": v.type,
                "timecode_sec": v.t,
                "description": v.description,
                "evidence_frames": frames,
                "priority": typ["priority"],
                "confidence": v.confidence,
                "rule": {
                    "document": norms[0]["document"],
                    "clause": norms[0]["clause"],
                    "url": norms[0]["url"],
                    "quote": norms[0]["quote"],
                    "basis": norms[0]["basis"],
                    "note": norms[0].get("note"),
                },
                "rules_all": [{"document": n["document_short"], "clause": n["clause"], "url": n["url"],
                               "quote": n["quote"], "basis": n["basis"]} for n in norms],
                "observed": v.facts,
            })
        result = {
            "verdict": "violation" if out_viols else "no_violation",
            "violations": out_viols,
            "meta": {
                "duration_sec": round(vf.duration, 1),
                "resolution": [vf.width, vf.height],
                "model": getattr(self.vlm, "model_id", self.vlm.name),
                "segments": [s.to_dict() for s in segs],
                "people_timeline": {"t": timeline.t, "count": timeline.count} if timeline else None,
                "scene": scene.to_dict() if scene else None,
                "deviations": devs,
                "machine_active": machine_active,
                "leave_checks": {str(k): v for k, v in cx.leave_checks.items()},
                "safe_checks": {str(k): v for k, v in cx.safe_checks.items()},
                "desk_checks": {str(k): v for k, v in cx.desk_checks.items()},
                "id_checks": {str(k): v for k, v in cx.id_checks.items()},
                "count_checks": {str(k): v for k, v in cx.count_checks.items()},
                "change_checks": {str(k): v for k, v in cx.change_checks.items()},
                "door_checks": cx.door_checks,
                "left_checks": {str(k): v for k, v in cx.left_checks.items()},
                "timings": {**timings, "total": round(time.time() - t_start, 1)},
                "disclaimer": "Результат является подсказкой для проверки человеком, а не юридическим заключением.",
            },
        }
        report("done", 1.0)
        return result
