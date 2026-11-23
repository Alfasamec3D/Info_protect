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
from check_attack import check as verify_attack
from export import page_map
from check_query_study import check as verify_queries
from check_soft_labels import check as verify_soft_labels
from check_binary_information import check as verify_binary_information
from check_optimizer_control import check as verify_optimizer_control
from check_observation_study import check as verify_observation
from check_ivan import check as verify_ivan


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
    for title, page_number in page_map(directory / "report.pdf").items():
        assert f"{title}\t{page_number}" in [p.text for p in document.paragraphs]
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
    assert len(deck.slides) == 29
    from ivan_sections import confirmation_rows, stopping_rows, result_sentence
    assert table_rows(document, ['Стратегия','Чистая accuracy, %','Ожидаемая шумная, %','Разных запросов']) == confirmation_rows(data)
    assert table_rows(document, ['σ','B','M','Доли','С учётом N','Снижение, %','ДИ разности × 10³']) == stopping_rows(data)
    assert result_sentence(data) in [p.text for p in document.paragraphs]
    from observation_sections import LABELS, metric as observation_metric, contrast, interval
    assert table_rows(document, ['Метод','Основная серия, %','Повтор, %','Сошлись, осн. / повт.']) == [
        [label,pct(observation_metric(data,method)['accuracy']),pct(observation_metric(data,method,True)['accuracy']),
         f"{observation_metric(data,method)['selected_converged']} / {observation_metric(data,method,True)['selected_converged']}"]
        for method,label in LABELS.items()]
    assert table_rows(document, ['Контраст','Основная серия, п. п.','Повтор, п. п.']) == [
        [f'{LABELS[a]} − {LABELS[b]}',interval(contrast(data,a,b)),interval(contrast(data,a,b,True))]
        for a,b in (('fraction','majority_aware'),('majority_aware','majority_naive'),('fraction','single'))]
    controls=table_rows(document,["Настройка","Большинство, %","Доли, %","Разность, п. п.","Повтор, п. п."])
    assert len(controls)==4
    for row,a,b in zip(controls,data['extensions']['optimizer'],data['extensions']['optimizer_repeat']):
        assert row[1]==pct(a['majority_accuracy']) and row[2]==pct(a['fraction_accuracy'])
        for value,expected in ((row[3],a),(row[4],b)):
            assert value==f"{pct(expected['paired_gain'])} [{pct(expected['ci_low'])}; {pct(expected['ci_high'])}]"
    from extension_sections import soft, effect as soft_effect
    assert table_rows(document, ["k", "r", "ρ", "Доли − большинство, п. п.", "Повтор, п. п."]) == [
        [str(k), str(r), str(rho).replace('.', ','), soft_effect(soft(data,k,r,rho)),
         soft_effect(soft(data,k,r,rho,repeated=True))]
        for k in (1,2) for r in (3,9) for rho in (0.,.9)
    ]
    binary_rows=table_rows(document,["r","Основная серия","Другой PUF","Теория"])
    assert len(binary_rows)==3
    for row, first, second in zip(binary_rows,data['extensions']['binary'],data['extensions']['binary_repeat']):
        for value, expected in zip(row[1:3],(first,second)):
            wanted=f"{float(expected['variance_ratio']):.2f} [{float(expected['variance_ratio_low']):.2f}; {float(expected['variance_ratio_high']):.2f}]".replace('.',',')
            assert value==wanted
    from query_sections import item, effect
    assert table_rows(document, ["k", "r", "ρ", "Основная серия, п. п.", "Повтор, п. п."]) == [
        [str(k), str(r), str(rho).replace('.', ','), effect(item(data,k,r,rho)),
         effect(item(data,k,r,rho,repeated=True))]
        for k in (1,2) for r in (3,9) for rho in (0.,.9)
    ]
    assert table_rows(document, ["k", "Произведение, %", "MLP, %", "Время prod / MLP, с"]) == [
        [str(k), f"{pct(item(data,k,noise=0.)['accuracy'])} / {pct(item(data,k,noise=0.,repeated=True)['accuracy'])}",
         f"{pct(item(data,k,noise=0.,model='mlp')['accuracy'])} / {pct(item(data,k,noise=0.,model='mlp',repeated=True)['accuracy'])}",
         f"{float(item(data,k,noise=0.)['fit_seconds']):.1f} / {float(item(data,k,noise=0.,model='mlp')['fit_seconds']):.1f}"]
        for k in (1,2,3,4)
    ]
    attack_table = table_rows(document, ["k", "σ", "Основная серия, %", "Повтор, %", "Выигрыш, п. п."])
    assert len(attack_table) == 6
    from build import attack_cell
    for row, (k, noise) in zip(attack_table, ((k, n) for k in (1, 2, 4) for n in (0., .1))):
        first_attack = attack_cell(data, k, noise)
        second_attack = attack_cell(data, k, noise, repeated=True)
        assert row[2] == f"{pct(first_attack['accuracy'])} [{bounds(first_attack, 'ci_low', 'ci_high')}]"
        assert row[3] == f"{pct(second_attack['accuracy'])} [{bounds(second_attack, 'ci_low', 'ci_high')}]"
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
        assert "HMAC-SHA-256" in text
        assert pct(attack_cell(data, 2, .1)["accuracy"]) in text
        assert pct(attack_cell(data, 2, .1, repeated=True)["accuracy"]) in text
        assert pct(soft(data,2,9,0.)['accuracy']) in text
        assert pct(contrast(data)['difference']) in text
        assert pct(contrast(data,repeated=True)['difference']) in text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--report", type=Path, default=Path("report"))
    parser.add_argument("--ivan-workers", type=int, default=4)
    args = parser.parse_args()
    main_ids, first = verify_numeric(args.results / "forecast")
    repeat_ids, second = verify_numeric(args.results / "forecast" / "repeat")
    assert main_ids.isdisjoint(repeat_ids)
    attack_ids = verify_attack(args.results / "attack")
    attack_repeat_ids = verify_attack(args.results / "attack" / "repeat")
    assert attack_ids.isdisjoint(attack_repeat_ids)
    query_ids = verify_queries(args.results / "query_study")
    query_repeat_ids = verify_queries(args.results / "query_study" / "repeat")
    assert query_ids.isdisjoint(query_repeat_ids)
    soft_ids = verify_soft_labels(args.results / "soft_labels", args.results / "query_study")
    soft_repeat_ids = verify_soft_labels(args.results / "soft_labels" / "repeat", args.results / "query_study" / "repeat")
    assert soft_ids == query_ids and soft_repeat_ids == query_repeat_ids
    optimizer_ids=verify_optimizer_control(args.results / "optimizer_control",args.results / "query_study",args.results / "soft_labels")
    optimizer_repeat_ids=verify_optimizer_control(args.results / "optimizer_control" / "repeat",args.results / "query_study" / "repeat",args.results / "soft_labels" / "repeat")
    assert optimizer_ids==query_ids and optimizer_repeat_ids==query_repeat_ids
    observation_ids = verify_observation(args.results / 'observation_study')
    observation_repeat_ids = verify_observation(args.results / 'observation_study/repeat')
    assert observation_ids.isdisjoint(observation_repeat_ids)
    assert (observation_ids | observation_repeat_ids).isdisjoint(query_ids | query_repeat_ids | attack_ids | attack_repeat_ids)
    confirmation_ids = verify_ivan(args.results / 'ivan/confirmation', 'confirmation', args.ivan_workers, write_audit=True)
    stopping_ids = verify_ivan(args.results / 'ivan/stopping', 'stopping', args.ivan_workers, write_audit=True)
    assert confirmation_ids.isdisjoint(stopping_ids)
    binary_count = verify_binary_information(args.results / "binary_information")
    binary_count += verify_binary_information(args.results / "binary_information" / "repeat")
    from puf_lab.information import verify
    stored = read_csv(args.results / "reference", "information_bound")
    assert len(stored) == 9
    for row, expected in zip(stored, verify()):
        for key, value in expected.items():
            within(row[key], value)
    verify_documents(args.report, first, second, args.results)
    print(f"Verified {len(main_ids) + len(repeat_ids)} independent device IDs, "
          f"metric intervals, {len(attack_ids | attack_repeat_ids)} attack devices, "
          f"{len(query_ids | query_repeat_ids)} query-study and soft-label devices, "
          f"{binary_count} binary-information designs, Office files and PDF exports.")
    print(f'Also verified the frozen-protocol study on {len(observation_ids | observation_repeat_ids)} new devices.')
    print(f'Also verified Ivan studies: {len(confirmation_ids)} fixed-repeat and {len(stopping_ids)} adaptive-repeat devices, 3500 fitted models in full profiles.')


if __name__ == "__main__":
    main()
