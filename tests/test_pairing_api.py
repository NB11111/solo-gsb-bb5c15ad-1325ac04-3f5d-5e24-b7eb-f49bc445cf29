"""End-to-end tests for the read-only rule fault sequence pairing API.

The pairing reads the stored confirmed fault intervals of two different rules
(only those whose start falls inside the half-open query window) and never
re-evaluates rules or writes fault results.
"""

import json
import os
import tempfile

import pytest

from app import db, services
from app.engine import to_epoch

BASE = to_epoch("2026-09-23T00:00:00Z")
H = 3600.0


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


@pytest.fixture()
def client(scratch_db):
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


def _create_rule(name):
    return db.execute(
        "INSERT INTO rules (name, logic, duration, created_at) VALUES (?,?,?,?)",
        (name, "AND", 0, 0.0),
    ).lastrowid


def _add_fault(rule_id, start, end):
    return db.execute(
        "INSERT INTO fault_intervals"
        " (rule_id, start_ts, end_ts, duration_required, confirmed_at, evidence_json)"
        " VALUES (?,?,?,?,?,?)",
        (rule_id, start, end, 0.0, start, "[]"),
    ).lastrowid


@pytest.fixture()
def paired_rules(scratch_db):
    """Two rules with directly stored faults (the pairing reads them verbatim).

    R1 (antecedent): faults at [00:00,00:01), [01:00,01:01), [03:00,03:01)
    R2 (subsequent): faults at [00:00,00:01) same start, [00:00:30,00:01:30),
                     [01:10,01:11)
    """
    r1 = _create_rule("前因规则")
    r2 = _create_rule("后续规则")
    f1 = _add_fault(r1, BASE, BASE + 60)
    f2 = _add_fault(r1, BASE + H, BASE + H + 60)
    f3 = _add_fault(r1, BASE + 3 * H, BASE + 3 * H + 60)
    g1 = _add_fault(r2, BASE, BASE + 60)          # same start as f1: not later
    g2 = _add_fault(r2, BASE + 30, BASE + 90)     # 30s after f1
    g3 = _add_fault(r2, BASE + H + 600, BASE + H + 660)  # 600s after f2
    return r1, r2, (f1, f2, f3), (g1, g2, g3)


# --------------------------------------------------------------------------- #
# service
# --------------------------------------------------------------------------- #


def test_service_pairs_within_max_lag(paired_rules):
    r1, r2, (f1, f2, f3), (g1, g2, g3) = paired_rules
    result = services.query_fault_pairing(r1, r2, BASE, BASE + 4 * H, 900)
    # g1 starts at the same instant as f1 (not later); f1 pairs g2 (lag 30),
    # f2 pairs g3 (lag 600); f3 finds nothing within 900s; g1 stays unmatched.
    assert [(p["antecedent_fault_id"], p["subsequent_fault_id"]) for p in result["pairs"]] == [
        (f1, g2),
        (f2, g3),
    ]
    assert [p["lag_seconds"] for p in result["pairs"]] == [30.0, 600.0]
    assert result["pair_count"] == 2
    assert result["total_lag_seconds"] == 630.0
    assert result["antecedent"]["total"] == 3
    assert result["antecedent"]["matched"] == 2
    assert result["antecedent"]["unmatched"] == 1
    assert result["antecedent"]["match_rate"] == pytest.approx(2 / 3)
    assert [f["fault_id"] for f in result["antecedent"]["unmatched_faults"]] == [f3]
    assert result["subsequent"]["total"] == 3
    assert result["subsequent"]["matched"] == 2
    assert result["subsequent"]["match_rate"] == pytest.approx(2 / 3)
    assert [f["fault_id"] for f in result["subsequent"]["unmatched_faults"]] == [g1]
    assert result["antecedent_rule"]["id"] == r1
    assert result["subsequent_rule"]["name"] == "后续规则"
    assert result["max_lag_seconds"] == 900.0


def test_service_max_lag_is_a_hard_bound(paired_rules):
    r1, r2, (f1, f2, _), (_, g2, g3) = paired_rules
    # 600s lag pairs, 599s does not
    result = services.query_fault_pairing(r1, r2, BASE, BASE + 4 * H, 600)
    assert [(p["antecedent_fault_id"], p["subsequent_fault_id"]) for p in result["pairs"]] == [
        (f1, g2),
        (f2, g3),
    ]
    result = services.query_fault_pairing(r1, r2, BASE, BASE + 4 * H, 599)
    assert [(p["antecedent_fault_id"], p["subsequent_fault_id"]) for p in result["pairs"]] == [
        (f1, g2),
    ]


def test_service_only_faults_starting_inside_window(paired_rules):
    r1, r2, (f1, f2, f3), (g1, g2, g3) = paired_rules
    # window starts at 00:30: f1/g1 (start 00:00) are excluded even though
    # their intervals overlap the window; f2 (01:00) pairs g3 (01:10)
    result = services.query_fault_pairing(r1, r2, BASE + 30, BASE + 2 * H, 900)
    assert result["antecedent"]["total"] == 1
    assert result["subsequent"]["total"] == 2  # g2 (00:00:30) and g3 (01:10)
    assert [(p["antecedent_fault_id"], p["subsequent_fault_id"]) for p in result["pairs"]] == [
        (f2, g3),
    ]
    # a window ending exactly at a fault start excludes that fault (half-open)
    result = services.query_fault_pairing(r1, r2, BASE, BASE + H, 900)
    assert result["antecedent"]["total"] == 1  # only f1
    assert result["antecedent"]["matched"] == 1


def test_service_same_rule_and_bad_args(paired_rules):
    r1, r2, *_ = paired_rules
    with pytest.raises(ValueError):
        services.query_fault_pairing(r1, r1, BASE, BASE + H, 60)
    with pytest.raises(KeyError):
        services.query_fault_pairing(r1, 999, BASE, BASE + H, 60)
    with pytest.raises(KeyError):
        services.query_fault_pairing(999, r2, BASE, BASE + H, 60)
    with pytest.raises(ValueError):
        services.query_fault_pairing(r1, r2, BASE + H, BASE, 60)
    with pytest.raises(ValueError):
        services.query_fault_pairing(r1, r2, BASE, BASE + H, 0)
    with pytest.raises(ValueError):
        services.query_fault_pairing(r1, r2, BASE, BASE + H, float("inf"))


def test_service_is_read_only(paired_rules):
    r1, r2, (f1, *_), _g = paired_rules
    before_faults = db.query("SELECT id, rule_id, start_ts, end_ts FROM fault_intervals")
    before_rules = db.query_one("SELECT COUNT(*) AS c FROM rules")["c"]
    services.query_fault_pairing(r1, r2, BASE, BASE + 4 * H, 900)
    after_faults = db.query("SELECT id, rule_id, start_ts, end_ts FROM fault_intervals")
    assert [dict(r) for r in after_faults] == [dict(r) for r in before_faults]
    assert db.query_one("SELECT COUNT(*) AS c FROM rules")["c"] == before_rules


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _params(rule_a, rule_b, **over):
    params = {
        "antecedent_rule": rule_a,
        "subsequent_rule": rule_b,
        "start": "2026-09-23T00:00:00Z",
        "end": "2026-09-23T04:00:00Z",
        "max_lag_seconds": 900,
    }
    params.update(over)
    return params


def test_api_pairing_ok(paired_rules, client):
    r1, r2, (f1, f2, _), (_, g2, g3) = paired_rules
    resp = client.get("/api/fault-pairing", params=_params(r1, r2))
    assert resp.status_code == 200
    body = resp.json()
    assert body["start"] == BASE and body["end"] == BASE + 4 * H
    assert body["pair_count"] == 2
    assert body["pairs"][0] == {
        "antecedent_fault_id": f1,
        "antecedent_start": BASE,
        "antecedent_end": BASE + 60,
        "subsequent_fault_id": g2,
        "subsequent_start": BASE + 30,
        "subsequent_end": BASE + 90,
        "lag_seconds": 30.0,
    }
    assert body["pairs"][1]["antecedent_fault_id"] == f2
    assert body["pairs"][1]["subsequent_fault_id"] == g3
    assert body["antecedent"]["match_rate"] == pytest.approx(2 / 3)
    assert body["antecedent_rule"]["name"] == "前因规则"
    assert body["subsequent_rule"]["name"] == "后续规则"


def test_api_epoch_range(paired_rules, client):
    r1, r2, (f1, *_), (_, g2, _) = paired_rules
    resp = client.get(
        "/api/fault-pairing",
        params=_params(r1, r2, start=BASE, end=BASE + 60, max_lag_seconds=60),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pair_count"] == 1
    assert body["pairs"][0]["antecedent_fault_id"] == f1
    assert body["pairs"][0]["subsequent_fault_id"] == g2


def test_api_evaluated_faults_pair(client, scratch_db):
    # full flow: import events, evaluate rules, then pair the stored faults
    rows = [
        {"device": "d1", "metric": "m", "ts": BASE, "value": 90},
        {"device": "d1", "metric": "m", "ts": BASE + 120, "value": 10},
        {"device": "d2", "metric": "m", "ts": BASE + 30, "value": 90},
        {"device": "d2", "metric": "m", "ts": BASE + 150, "value": 10},
    ]
    services.import_events("\n".join(json.dumps(r) for r in rows), reevaluate=False)
    r1 = db.execute(
        "INSERT INTO rules (name, logic, duration, created_at) VALUES (?,?,?,?)",
        ("r1", "AND", 0, 0.0),
    ).lastrowid
    db.execute(
        "INSERT INTO rule_conditions (rule_id, device, metric, op, threshold, ord)"
        " VALUES (?,?,?,?,?,?)",
        (r1, "d1", "m", ">=", 80, 0),
    )
    r2 = db.execute(
        "INSERT INTO rules (name, logic, duration, created_at) VALUES (?,?,?,?)",
        ("r2", "AND", 0, 0.0),
    ).lastrowid
    db.execute(
        "INSERT INTO rule_conditions (rule_id, device, metric, op, threshold, ord)"
        " VALUES (?,?,?,?,?,?)",
        (r2, "d2", "m", ">=", 80, 0),
    )
    services.evaluate_all_rules()
    resp = client.get(
        "/api/fault-pairing",
        params=_params(r1, r2, end="2026-09-23T01:00:00Z", max_lag_seconds=300),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pair_count"] == 1
    assert body["pairs"][0]["lag_seconds"] == 30.0
    assert body["antecedent"]["match_rate"] == 1.0
    assert body["subsequent"]["match_rate"] == 1.0


def test_api_same_rule_400(paired_rules, client):
    r1, *_ = paired_rules
    resp = client.get("/api/fault-pairing", params=_params(r1, r1))
    assert resp.status_code == 400


def test_api_unknown_rule_404(paired_rules, client):
    r1, *_ = paired_rules
    resp = client.get("/api/fault-pairing", params=_params(r1, 999))
    assert resp.status_code == 404
    resp = client.get("/api/fault-pairing", params=_params(999, r1))
    assert resp.status_code == 404


def test_api_inverted_window_400(paired_rules, client):
    r1, r2, *_ = paired_rules
    resp = client.get(
        "/api/fault-pairing",
        params=_params(r1, r2, start="2026-09-23T04:00:00Z", end="2026-09-23T00:00:00Z"),
    )
    assert resp.status_code == 400


def test_api_bad_max_lag_rejected(paired_rules, client):
    r1, r2, *_ = paired_rules
    resp = client.get("/api/fault-pairing", params=_params(r1, r2, max_lag_seconds=0))
    assert resp.status_code == 422  # Query(..., gt=0) rejects zero
    resp = client.get("/api/fault-pairing", params=_params(r1, r2, max_lag_seconds=-5))
    assert resp.status_code == 422
    resp = client.get("/api/fault-pairing", params=_params(r1, r2, max_lag_seconds="nan"))
    assert resp.status_code == 422  # NaN fails the gt=0 query validation
    resp = client.get("/api/fault-pairing", params=_params(r1, r2, max_lag_seconds="inf"))
    assert resp.status_code == 400  # non-finite lag rejected explicitly


def test_api_missing_params_422(client):
    assert client.get("/api/fault-pairing").status_code == 422
    resp = client.get(
        "/api/fault-pairing",
        params={"antecedent_rule": 1, "start": "2026-09-23T00:00:00Z",
                "end": "2026-09-23T01:00:00Z", "max_lag_seconds": 60},
    )
    assert resp.status_code == 422


def test_api_does_not_touch_faults(paired_rules, client):
    r1, r2, *_ = paired_rules
    before = db.query("SELECT id, rule_id, start_ts, end_ts, evidence_json FROM fault_intervals")
    resp = client.get("/api/fault-pairing", params=_params(r1, r2))
    assert resp.status_code == 200
    after = db.query("SELECT id, rule_id, start_ts, end_ts, evidence_json FROM fault_intervals")
    assert [dict(r) for r in after] == [dict(r) for r in before]
