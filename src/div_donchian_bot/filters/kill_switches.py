from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KillSwitchCondition:
    feature: str
    op: str
    value: Any
    value2: Optional[Any] = None


@dataclass
class KillSwitchRule:
    name: str
    side: str
    all: List[KillSwitchCondition] = field(default_factory=list)
    any: List[KillSwitchCondition] = field(default_factory=list)


@dataclass
class KillSwitchConfig:
    enabled: bool = False
    mode: str = "off"
    missing_policy: str = "pass"
    feature_config: Dict[str, Any] = field(default_factory=dict)
    rules: List[KillSwitchRule] = field(default_factory=list)


@dataclass
class KillDecision:
    passed: bool
    hit_rules: List[str]
    missing_features: List[str]
    features_used: Dict[str, Any]
    mode: str = "off"


def _parse_condition(d: Dict[str, Any]) -> KillSwitchCondition:
    feature = str(d.get("feature", "")).strip()
    op = str(d.get("op", "")).strip()
    return KillSwitchCondition(
        feature=feature,
        op=op,
        value=d.get("value"),
        value2=d.get("value2"),
    )


def parse_kill_switches(cfg_dict: Dict[str, Any]) -> KillSwitchConfig:
    """Parse kill-switch config safely from a raw dict."""
    if not isinstance(cfg_dict, dict):
        return KillSwitchConfig()

    enabled = bool(cfg_dict.get("enabled", False))
    mode = str(cfg_dict.get("mode", "off")).lower()
    if mode not in ("off", "shadow", "filter"):
        mode = "off"
    missing_policy = str(cfg_dict.get("missing_policy", "pass")).lower()
    if missing_policy not in ("pass", "fail"):
        missing_policy = "pass"
    feature_cfg = cfg_dict.get("feature_config", {}) or {}
    feature_cfg = {
        "atr_len_15m": int(feature_cfg.get("atr_len_15m", 24)),
        "liq_lookback_bars": int(feature_cfg.get("liq_lookback_bars", 240)),
        "liq_min_samples": int(feature_cfg.get("liq_min_samples", 30)),
        "cache_size": int(feature_cfg.get("cache_size", 50)),
    }

    rules: List[KillSwitchRule] = []
    for r in cfg_dict.get("rules", []) or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", "rule")).strip() or "rule"
        side = str(r.get("side", "BOTH")).upper()
        if side not in ("LONG", "SHORT", "BOTH"):
            side = "BOTH"
        all_list = [_parse_condition(c) for c in (r.get("all") or []) if isinstance(c, dict)]
        any_list = [_parse_condition(c) for c in (r.get("any") or []) if isinstance(c, dict)]
        rules.append(KillSwitchRule(name=name, side=side, all=all_list, any=any_list))

    return KillSwitchConfig(enabled=enabled, mode=mode, missing_policy=missing_policy, feature_config=feature_cfg, rules=rules)


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


def _to_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
    except Exception:
        return None
    if math.isnan(f):
        return None
    return f


def _match_op(op: str, feature_val: Any, cond: KillSwitchCondition) -> bool:
    op = op.lower()
    if op in (">", "gt"):
        fv = _to_float(feature_val)
        cv = _to_float(cond.value)
        return fv is not None and cv is not None and fv > cv
    if op in (">=", "ge"):
        fv = _to_float(feature_val)
        cv = _to_float(cond.value)
        return fv is not None and cv is not None and fv >= cv
    if op in ("<", "lt"):
        fv = _to_float(feature_val)
        cv = _to_float(cond.value)
        return fv is not None and cv is not None and fv < cv
    if op in ("<=", "le"):
        fv = _to_float(feature_val)
        cv = _to_float(cond.value)
        return fv is not None and cv is not None and fv <= cv
    if op in ("==", "eq"):
        return feature_val == cond.value
    if op in ("!=", "ne"):
        return feature_val != cond.value
    if op == "between":
        fv = _to_float(feature_val)
        lo = _to_float(cond.value)
        hi = _to_float(cond.value2)
        return fv is not None and lo is not None and hi is not None and lo <= fv <= hi
    if op == "in":
        try:
            return feature_val in cond.value
        except Exception:
            return False
    if op == "not_in":
        try:
            return feature_val not in cond.value
        except Exception:
            return False
    if op == "abs_gt":
        fv = _to_float(feature_val)
        cv = _to_float(cond.value)
        return fv is not None and cv is not None and abs(fv) > cv
    if op == "abs_lt":
        fv = _to_float(feature_val)
        cv = _to_float(cond.value)
        return fv is not None and cv is not None and abs(fv) < cv
    return False


def evaluate_kill_switch(sig: Any, feature_map: Dict[str, Any], ks_cfg: KillSwitchConfig) -> KillDecision:
    """Evaluate kill-switch rules against a feature map."""
    mode = getattr(ks_cfg, "mode", "off")
    if not ks_cfg or not ks_cfg.enabled or mode == "off":
        return KillDecision(passed=True, hit_rules=[], missing_features=[], features_used={}, mode=mode)

    feature_map = feature_map or {}
    missing_policy = ks_cfg.missing_policy
    hit_rules: List[str] = []
    missing: set[str] = set()
    used: Dict[str, Any] = {}

    for rule in ks_cfg.rules:
        if rule.side not in ("BOTH", getattr(sig, "side", None)):
            continue

        # Track values for diagnostics
        for cond in rule.all + rule.any:
            if cond.feature not in used:
                used[cond.feature] = feature_map.get(cond.feature)
            if _is_missing(feature_map.get(cond.feature)):
                missing.add(cond.feature)

        def eval_condition(cond: KillSwitchCondition) -> bool:
            val = feature_map.get(cond.feature)
            if _is_missing(val):
                return missing_policy == "fail"
            return _match_op(cond.op, val, cond)

        all_pass = True
        if rule.all:
            for cond in rule.all:
                if not eval_condition(cond):
                    all_pass = False
                    break

        any_pass = True
        if rule.any:
            any_pass = False
            for cond in rule.any:
                if eval_condition(cond):
                    any_pass = True
                    break

        if all_pass and any_pass:
            hit_rules.append(rule.name)

    passed = len(hit_rules) == 0
    return KillDecision(
        passed=passed,
        hit_rules=hit_rules,
        missing_features=sorted(missing),
        features_used=used,
        mode=mode,
    )
