"""Check the submission against the underlying forecast and its exported documents."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

from docx import Document
import numpy as np
from pptx import Presentation

from build import MODELS, REGIMES, TRANSFER_CASES, bounds, gain_ci, load_data, pct, read_csv
from puf_lab.defensive_auth import demo
from puf_lab.experiment import stream


def within(actual, expected):
    np.testing.assert_allclose(float(actual), expected, atol=1e-12, rtol=0)


def verify_errors(row, prediction, target, baseline, draws):
    absolute = np.abs(prediction - target)
    square = np.square(prediction - target)
    gain = baseline - absolute
    within(row["mae"], absolute.mean())
    within(row["rmse"], np.sqrt(square.mean()))
    within(row["mae_gain"], gain.mean())
    for prefix, estimates in (
        ("mae", absolute[draws].mean(axis=1)),
        ("rmse", np.sqrt(square[draws].mean(axis=1))),
        ("gain", gain[draws].mean(axis=1)),
    ):
        low, high = np.quantile(estimates, [.025, .975])
        within(row[f"{prefix}_ci_low"], low)
        within(row[f"{prefix}_ci_high"], high)


def verify_numeric(directory):
    rows = read_csv(directory, "metrics")
    records = read_csv(directory, "device_predictions")
    partitions = read_csv(directory, "partition_audit")
    profile = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    ids = {row["device_id"] for row in partitions}
    assert len(ids) == len(partitions), "Duplicate or overlapping device IDs"
    test_ids = {row["device_id"] for row in partitions if row["split"] == "test"}
    assert test_ids == {r["device_id"] for r in records} and len(test_ids) == len(records)
    assert len(rows) == len(REGIMES) * len(MODELS)
    counts = Counter((r["regime"], r["split"]) for r in partitions)
    for regime in REGIMES:
        for split in ("train", "validation", "test"):
            assert counts[regime, split] == profile[f"{split}_devices"]
    assert not (set(profile["features"]) & {"next_ber", "device_id", "split"})

    cross = read_csv(directory, "transfer_metrics")
    cross_predictions = read_csv(directory, "transfer_predictions")
    assert len(cross) == 6 * len(MODELS)
    assert len(cross_predictions) == 6 * profile["test_devices"]
    for row in cross + cross_predictions:
        assert row["source_regime"] in REGIMES
        assert row["regime"] in REGIMES and row["source_regime"] != row["regime"]

    for i, regime in enumerate(REGIMES):
        devices = [r for r in records if r["regime"] == regime]
        target = np.array([float(r["next_ber"]) for r in devices])
        assert all(r["baseline"] == r["current_ber"] for r in devices)
        baseline = np.abs(np.array([float(r["baseline"]) for r in devices]) - target)
        rng = stream(profile["seed"], 83, i)
        draws = rng.integers(len(devices), size=(profile["bootstrap_resamples"], len(devices)))
        for row in (r for r in rows if r["regime"] == regime):
            prediction = np.array([float(r[row["model"]]) for r in devices])
            verify_errors(row, prediction, target, baseline, draws)
        for source in REGIMES:
            if source == regime:
                continue
            transferred = {r["device_id"]: r for r in cross_predictions
                           if (r["source_regime"], r["regime"]) == (source, regime)}
            assert set(transferred) == {r["device_id"] for r in devices}
            assert all(transferred[r["device_id"]]["baseline"] == r["baseline"] for r in devices)
            pair = [r for r in cross if (r["source_regime"], r["regime"]) == (source, regime)]
            assert {r["model"] for r in pair} == set(MODELS) and len(pair) == len(MODELS)
            for row in pair:
                prediction = np.array([float(transferred[r["device_id"]][row["model"]])
                                       for r in devices])
                verify_errors(row, prediction, target, baseline, draws)
    validation = read_csv(directory, "validation")
    for regime in REGIMES:
        for model in ("ridge", "boosting"):
            candidates = [r for r in validation if (r["regime"], r["model"]) == (regime, model)]
            selected = [r for r in candidates if r["selected"] == "True"]
            assert len(selected) == 1
            within(selected[0]["validation_mae"], min(float(r["validation_mae"]) for r in candidates))
    return ids, {(r["regime"], r["model"]): r for r in rows}


def table_rows(document, header):
    table = next(item for item in document.tables
                 if [cell.text for cell in item.rows[0].cells] == header)
    return [[cell.text for cell in row.cells] for row in table.rows[1:]]


def verify_documents(directory, main, repeat, results):
    for name in ("report.docx", "slides.pptx"):
        with ZipFile(directory / name) as archive:
            assert archive.testzip() is None
    document = Document(directory / "report.docx")
    deck = Presentation(directory / "slides.pptx")
    methods = ("ridge", "boosting")
    errors = [
        [label, method_label, pct(main[regime, method]["mae"]), pct(main[regime, method]["rmse"]),
         pct(repeat[regime, method]["mae"]), pct(repeat[regime, method]["rmse"])]
        for regime, label in REGIMES.items() for method, method_label in MODELS.items()
    ]
    gains = [
        [label, MODELS[method], gain_ci(main[regime, method]), gain_ci(repeat[regime, method])]
        for regime, label in REGIMES.items() for method in methods
    ]
    assert table_rows(document, ["Режим", "Метод", "MAE, осн.", "RMSE, осн.",
                                 "MAE, повтор", "RMSE, повтор"]) == errors
    assert table_rows(document, ["Режим", "Метод", "Основной прогон", "Новый seed"]) == gains
    data = load_data(results)
    sources = table_rows(document, ["Авторы и схема", "Данные", "Результат", "Условия и место"])
    assert [(row[1], row[2]) for row in sources] == [
        (row["data_origin"], row["observed"]) for row in data["evidence"]
    ]
    cold = data["control"]

    def show(row):
        return f"{pct(row['fraction_ones'])} [{bounds(row, 'ci_low', 'ci_high')}]"

    assert table_rows(document, ["k", "Центрированная модель", "Сдвиг 0,60", "Формула при сдвиге"]) == [
        [str(k), show(cold[k, 0]), show(cold[k, .6]), pct(cold[k, .6]["theory_fraction_ones"])]
        for k in (1, 2, 3, 4)
    ]
    assert data["defense"] == demo(), "The saved protection checks differ from a fresh run"
    protection = table_rows(document, ["Проверка", "Наблюдаемый исход"])
    assert [row[1] for row in protection] == [
        "Принято", "Отклонён", "Отклонён", "Отклонено", "Принято 1 из 2",
    ]
    transfer = [
        [f"{REGIMES[origin]} → {REGIMES[target]}", MODELS[method],
         gain_ci(data["transfer"][origin, target, method]),
         gain_ci(data["transfer_repeat"][origin, target, method])]
        for origin, target, method in TRANSFER_CASES
    ]
    assert table_rows(document, ["Обучение → проверка", "Модель",
                                 "Основной, п. п.", "Повтор, п. п."]) == transfer
    assert len(deck.slides) == 14
    slide = " ".join(shape.text for shape in deck.slides[10].shapes if shape.has_text_frame)
    assert pct(data["transfer"]["volatile", "stable", "ridge"]["mae"]) in slide
    protection_slide = " ".join(
        shape.text for shape in deck.slides[12].shapes if shape.has_text_frame
    )
    assert "1 из 2" in protection_slide and "HMAC-SHA-256" in protection_slide
    sources_slide = " ".join(shape.text for shape in deck.slides[3].shapes if shape.has_text_frame)
    for value in ("≈90%", "≈87%", "99%"):
        assert value in sources_slide
    for index, table in ((8, main), (9, repeat)):
        content = " ".join(s.text for s in deck.slides[index].shapes if s.has_text_frame)
        for method in ("baseline", "ridge"):
            assert pct(table["volatile", method]["mae"]) in content
    for filename in ("report.pdf", "slides.pdf"):
        text = subprocess.check_output(
            ["pdftotext", str(directory / filename), "-"], text=True, encoding="utf-8",
        )
        assert all(pct(table["volatile", method]["mae"]) in text
                   for table in (main, repeat) for method in ("baseline", "ridge"))
        assert "открыт" in text.lower() and "HMAC-SHA-256" in text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--report", type=Path, default=Path("report"))
    args = parser.parse_args()
    main_ids, first = verify_numeric(args.results / "forecast")
    repeat_ids, second = verify_numeric(args.results / "forecast" / "repeat")
    assert main_ids.isdisjoint(repeat_ids)
    verify_documents(args.report, first, second, args.results)
    print(f"Verified {len(main_ids) + len(repeat_ids)} independent device IDs, "
          "metric intervals, Office files and PDF exports.")


if __name__ == "__main__":
    main()
