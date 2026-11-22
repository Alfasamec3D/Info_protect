"""Build editable report and presentation from the completed study."""

import argparse
import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor as WordColor
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt as SlidePt


TITLE = "Предсказуемость Arbiter PUF и надёжность следующей сессии"
SUBTITLE = "Два сценария доступа и оценка BER на новых экземплярах"
AUTHORS = "Иван Шаверин · Антон Хмельницкий · Максим Рачинский"
REGIMES = {
    "stable": "Без дрейфа",
    "persistent": "Устойчивый дрейф",
    "volatile": "Изменчивый дрейф",
}
MODELS = {"baseline": "Текущий BER", "ridge": "Ridge", "boosting": "Бустинг"}
TRANSFER_CASES = (
    ("volatile", "stable", "ridge"),
    ("volatile", "stable", "boosting"),
    ("volatile", "persistent", "ridge"),
    ("stable", "volatile", "ridge"),
    ("persistent", "volatile", "boosting"),
)
SECTIONS = (
    "Введение и постановка задачи",
    "1 Сценарий без пар данного экземпляра",
    "2 Проверка симметрии на симуляторе",
    "3 Сценарий с ранее измеренными парами",
    "4 Симулятор следующей сессии",
    "5 ML-модели и разбиение по экземплярам",
    "6 Ошибки в основной и повторной сериях",
    "7 Парные интервалы и интерпретация",
    "8 Перенос между режимами дрейфа",
    "9 Как вспомогательные эксперименты объясняют выводы",
    "Заключение",
    "Приложение. Проверка одноразового запроса",
    "Приложение. Проверка требований",
    "Список источников",
)
SOURCES = [
    "Рабочая программа дисциплины «Защита информации», МФТИ. Фонд оценочных средств, разделы 3–5. Файл РУП ЗИ (3) последний.docx.",
    "Переписка группы по курсовому проекту «Защита информации». https://chatgpt.com/s/cx_6afd9a1bfd5881919dc4c94e7f42e22c",
    "Hospodar G., Maes R., Verbauwhede I. Machine Learning Attacks on 65nm Arbiter PUFs: Accurate Modeling Poses Strict Bounds on Usability. IEEE WIFS, 2012, разд. III, рис. 1(a)–(b). https://cosicdatabase.esat.kuleuven.be/backend/publications/files/conferencepaper/2264",
    "Rührmair U. et al. PUF Modeling Attacks on Simulated and Silicon Data. IEEE TIFS, 2013, vol. 8, no. 11, pp. 1876–1891, табл. XI. https://sharps.org/wp-content/uploads/RUHRMAIR-TIFS13.pdf",
    "Sayadi N. et al. Breaking XOR Arbiter PUFs without Reliability Information. Препринт, 2023, разд. 4.4 и табл. 4. https://ir.cwi.nl/pub/33755/33755.pdf",
    "Dumoulin V., Rao W., Devroye N. On the Response Entropy of APUFs. 2024. https://arxiv.org/abs/2406.19975",
    "Lim D. et al. Extracting Secret Keys From Integrated Circuits. IEEE TVLSI, 2005. https://doi.org/10.1109/TVLSI.2005.859470",
    "Maiti A., Gunreddy V., Schaumont P. A Systematic Method to Evaluate and Compare the Performance of Physical Unclonable Functions. 2011. https://eprint.iacr.org/2011/657.pdf",
    "pypuf. Additive Delay Model. https://pypuf.readthedocs.io/en/latest/simulation/delay.html",
    "NIST/SEMATECH. e-Handbook of Statistical Methods. Bootstrap Plot. https://www.itl.nist.gov/div898/handbook/eda/section3/bootplot.htm",
    "scikit-learn 1.7. Ridge and HistGradientBoostingRegressor. https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html",
    "scikit-learn 1.7. HistGradientBoostingRegressor. https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html",
    "Krawczyk H., Bellare M., Canetti R. HMAC: Keyed-Hashing for Message Authentication. RFC 2104. https://www.rfc-editor.org/rfc/rfc2104",
    "NIST SP 800-63B-4. Digital Identity Guidelines. Authentication and Authenticator Management. Разделы о криптографических аутентификаторах и одноразовых запросах. https://pages.nist.gov/800-63-4/sp800-63b.html",
]


def read_csv(directory, name):
    with (directory / f"{name}.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def pct(value, digits=2):
    return f"{100 * float(value):.{digits}f}".replace(".", ",")


def cell(rows, **keys):
    return next(row for row in rows if all(float(row[key]) == val for key, val in keys.items()))


def set_word_style(style, size, bold=False):
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = WordColor.from_string("000000")


def create_word():
    doc = Document()
    doc.core_properties.title = TITLE
    doc.core_properties.subject = SUBTITLE
    doc.core_properties.author = AUTHORS
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin, section.right_margin = Cm(3), Cm(1.5)
    section.top_margin, section.bottom_margin = Cm(2), Cm(2)
    section.different_first_page_header_footer = True
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    normal = doc.styles["Normal"]
    set_word_style(normal, 12)
    normal.paragraph_format.line_spacing = 1
    normal.paragraph_format.first_line_indent = Cm(1.25)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, size in (("Title", 22), ("Heading 1", 16), ("Heading 2", 13)):
        style = doc.styles[name]
        set_word_style(style, size, True)
        style.paragraph_format.first_line_indent = Cm(0)
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        style.paragraph_format.space_before = Pt(8)
        style.paragraph_format.space_after = Pt(7)
        style.paragraph_format.keep_with_next = True
    return doc


def centered(doc, text, size=12, before=0, bold=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(before)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    return p


def table(doc, headings, body):
    grid = doc.add_table(rows=0, cols=len(headings))
    grid.style = "Table Grid"
    for line in [headings, *body]:
        row = grid.add_row()
        for box, value in zip(row.cells, line):
            p = box.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.space_before = Pt(4)
            run = p.add_run(str(value))
            run.bold = line is headings
            run.font.size = Pt(11)


def figure(doc, image, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(image), width=Cm(16))
    centered(doc, caption, size=11)


def new_page(doc, title):
    doc.add_page_break()
    doc.add_heading(title)


def bounds(row, lo, hi, digits=2):
    return f"{pct(row[lo], digits)} … {pct(row[hi], digits)}"


def value_ci(row, metric, digits=2):
    return f"{pct(row[metric], digits)} [{bounds(row, metric + '_ci_low', metric + '_ci_high', digits)}]"


def gain_ci(row, digits=3):
    return f"{pct(row['mae_gain'], digits)} [{bounds(row, 'gain_ci_low', 'gain_ci_high', digits)}]"


def build_word(output, results, data):
    doc = create_word()
    p = doc.add_paragraph
    source = data["main"]
    repeated = data["repeat"]
    case = source["volatile", "ridge"]
    base = source["volatile", "baseline"]

    centered(doc, "Московский физико-технический институт", 14, bold=True)
    centered(doc, "(национальный исследовательский университет)")
    centered(doc, "Физтех-школа радиотехники и компьютерных технологий")
    centered(doc, "ГРУППОВОЙ КУРСОВОЙ ПРОЕКТ", 15, before=100, bold=True)
    centered(doc, "Дисциплина «Защита информации»")
    centered(doc, "Заявленная тема № 12 «Реализация атак на PUF»", before=18)
    centered(doc, TITLE, 21, before=15, bold=True)
    centered(doc, SUBTITLE, 14)
    centered(doc, "Шаверин Иван\nХмельницкий Антон\nРачинский Максим", 13, before=65)
    centered(doc, "Долгопрудный, 2026", before=75)

    new_page(doc, "РЕФЕРАТ")
    p("Работа проверяет, достаточно ли отсутствия физического доступа к Arbiter PUF или XOR Arbiter PUF для непредсказуемости ответов. Для независимого нового экземпляра без его пар запрос–ответ получено условие равновероятности и проведены проверки симулятора с нарушением симметрии. Для сценария с историческими парами сопоставлены первичные публикации на аппаратных данных.")
    p(f"В собственной ML-части целевой величиной служит агрегированная доля ошибок следующей сессии. Сравниваются правило текущего BER, Ridge и градиентный бустинг. При изменчивом дрейфе Ridge снизила MAE с {pct(base['mae'])} до {pct(case['mae'])} п. п. При устойчивом дрейфе интервалы выигрыша включают ноль. Независимый повтор выполнен с новыми экземплярами.")
    p("Проверены воспроизводимость результатов, разбиение по экземплярам и согласованность чисел отчёта с итоговыми таблицами.")
    p("Ключевые слова — PUF, Arbiter, XOR, CRP, BER, градиентный бустинг, дрейф, воспроизводимость.")

    new_page(doc, "СОДЕРЖАНИЕ")
    for number, name in enumerate(SECTIONS, 4):
        item = p(f"{name}\t{number}")
        item.paragraph_format.first_line_indent = Cm(0)
        item.paragraph_format.space_after = Pt(2)
        item.paragraph_format.tab_stops.add_tab_stop(
            Cm(16.45), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS,
        )

    new_page(doc, SECTIONS[0].upper())
    p("Рабочая программа дисциплины формулирует тему № 12 как «Реализация атак на PUF» [1]. В переписке группа предложила выяснить, достаточно ли отсутствия физического доступа, чтобы ответы PUF оставались непредсказуемыми [2].")
    p("Физически неклонируемая функция сопоставляет входному запросу ответ, который зависит от параметров конкретного устройства. В Arbiter PUF знак разности двух задержек определяет бит. XOR объединяет ответы нескольких цепочек. Открытый запрос обозначим C, итоговый бит — Y, а информацию наблюдателя — Z.")
    p("В первом сценарии наблюдатель никогда не имел пар этого экземпляра и не может его читать. Он знает текущий запрос и, возможно, статистику партии. Во втором сценарии физического доступа в момент предсказания нет, однако имеются ранее измеренные пары того же экземпляра. Эти постановки задают разные условные распределения P(Y | C, Z).")
    p("Самостоятельный вычислительный вопрос касается качества измерений. Можно ли по агрегатам калибровочной сессии прогнозировать долю ошибок следующей сессии на экземплярах, которые не использовались для обучения и выбора модели? Наблюдаемое Y в этом эксперименте — одно число BER на устройство и сессию.")
    p("Максим отвечает за симулятор и агрегирование. Иван отвечает за сравнение моделей и анализ публикаций. Антон отвечает за план эксперимента, статистику и воспроизводимость. На защите каждый показывает свой технический блок, исходные результаты и ограничения.")

    new_page(doc, SECTIONS[1].upper())
    p("Пусть c — фиксированный допустимый запрос, а Z — информация, независимая от весов и шума нового экземпляра при заданном c. Допустим, что знаки задержек цепочек при этих условиях независимы и каждый равновероятен. Это выполняется в модели независимых центрированных гауссовских весов при симметричном независимом шуме. При непрерывном распределении нулевая задержка имеет нулевую вероятность.")
    centered(doc, "P(Y = 1 | C = c, Z) = 1/2.")
    p("Для одной цепочки это следует из симметрии. Для XOR фиксируем остальные цепочки. Первый независимый равновероятный знак с одинаковой вероятностью сохраняет либо инвертирует их чётность. Отсюда следует равновероятность итогового бита. Для любой функции угадывания, использующей только c, Z и независимую случайность, условная вероятность совпадения также равна 1/2.")
    p("Точное достаточное условие — равномерность условного распределения итогового бита для каждого запроса. Независимость цепочек и центрированность служат простым способом её обеспечить. Если известна связанная с экземпляром информация или если вход выбран с учётом его скрытых параметров, доказательство теряет основание.")
    p("При общей компоненте задержек чётное число цепочек может давать смещённый XOR даже при сбалансированности каждой цепочки. При нечётном числе глобальная инверсия знаков меняет итоговую чётность; симметрия всей совместной модели тогда сохраняет баланс. Это свойства модели распределения, а не утверждения о произвольном изготовленном устройстве.")
    p("Итог первого сценария. Предел 50% доказан при явно заданной симметрии и независимости наблюдений от экземпляра. Сам факт отсутствия доступа и истории пар такого предела не гарантирует.")

    new_page(doc, SECTIONS[2].upper())
    control = data["control"]
    p("Контрольный эксперимент использует 192 независимых виртуальных устройства, 8192 общих запроса и независимый шум чтения со стандартным отклонением 0,10. Центрированная модель сравнивается с постоянным сдвигом среднего каждой цепочки 0,60. Контроль задаётся генератором, а доли единиц измеряются заново.")
    p("Таблица 1 — Доля единиц. Интервалы вычислены бутстрэпом по устройствам, единицы измерения — проценты.")
    table(doc, ("k", "Центрированная модель", "Сдвиг 0,60", "Формула при сдвиге"), [
        (k, f"{pct(control[k, 0]['fraction_ones'])} [{bounds(control[k, 0], 'ci_low', 'ci_high')}]",
         f"{pct(control[k, .6]['fraction_ones'])} [{bounds(control[k, .6], 'ci_low', 'ci_high')}]",
         pct(control[k, .6]["theory_fraction_ones"]))
        for k in (1, 2, 3, 4)
    ])
    shifted = control[1, .6]
    p(f"Для одной цепочки при сдвиге расчётная точность лучшего постоянного ответа равна {pct(shifted['theory_majority_accuracy'])}%. При центрированной модели точное среднее равно 50%. В контрольном прогоне наблюдался конечновыборочный разброс, поэтому эмпирические интервалы не трактуются как доказательство равенства.")
    corr = data["spatial"]
    p(f"В дополнительном контроле у цепочек общий центрированный фактор, а корреляция чистых задержек равна 0,60. Для 4-XOR расчётная точность лучшего постоянного ответа составила {pct(corr['theory_constant_accuracy'])}%. Доля единиц в основном прогоне — {pct(corr['fraction_ones'])}%, в повторе — {pct(data['spatial_repeat']['fraction_ones'])}%. Этот опыт проверяет достаточность предположения о балансе отдельных цепочек.")
    p("Оба контроля сохраняют отсутствие данных конкретного устройства. Первый меняет симметрию отдельной цепочки, второй меняет зависимость между цепочками. Их результаты устанавливают границы вывода о 50% в используемом ансамбле.")

    new_page(doc, SECTIONS[3].upper())
    p("При наличии старых пар задача моделирования состоит в проверке на новых входах того же экземпляра. Первичные публикации дают контрпримеры для аппаратных Arbiter и XOR Arbiter PUF. В таблице приведены показатели авторов и точное расположение каждого числа. Нумерация страниц относится к PDF по указанным ссылкам.")
    p("Таблица 2 — Литературные результаты. Расширенные условия и ссылки содержатся в report/evidence.csv.")
    table(doc, ("Авторы и схема", "Данные", "Результат", "Условия и место"), [
        (f"{row['source']} ({row['year']})\n{row['scheme']}",
         row["data_origin"], row["observed"],
         short)
        for row, short in zip(data["evidence"], (
            "Отдельный тест на новых запросах. Разд. III-A–B, рис. 1(a), PDF с. 3–4 [3].",
            "Новые запросы. Итоговая чётность измеренных простых цепочек одного кристалла. Разд. III-B, рис. 1(b), PDF с. 3–4 [3].",
            "Итоговый бит, усреднение по экземплярам. Табл. XI, строка FPGA, PDF с. 15. Условия в II-G и IX [4].",
            "Итоговый бит, усреднение по экземплярам. Табл. XI, строка ASIC, PDF с. 15. Условия в II-G и IX [4].",
            "Выбранные запросы, метрика отдельных цепочек. Разд. 4.4 и табл. 4, PDF с. 8–9 [5].",
         ))
    ])
    p("Первые четыре строки относятся к итоговому ответу. Последняя строка характеризует отдельные цепочки при другой модели доступа. Объединять эти значения в среднюю точность некорректно. Нельзя переносить цифры на произвольную архитектуру или на режим без ранее измеренных пар [6].")
    p("Итог второго сценария. Отсутствие физического доступа в момент прогноза недостаточно для непредсказуемости. Указанные результаты опровергают общее утверждение для исследованных реализаций при наличии истории CRP.")

    new_page(doc, SECTIONS[4].upper())
    p("Для собственного эксперимента используется нормированная аддитивная модель задержек [7, 9]. Один набор весов задаёт один экземпляр на обе сессии. Как метрику повторяемости применяем долю расхождений шумного чтения с чистым ответом симулятора [8]. Эталон доступен в стенде, а при аппаратном переносе потребуется отдельная процедура его формирования.")
    p("Для каждого экземпляра выбираются 1, 2 или 4 цепочки с равными вероятностями. Стандартное отклонение шума чтения выбирается равномерно от 0,04 до 0,14 и остаётся неизменным у данного экземпляра. При условии параметров устройства ошибки отдельных чтений задаются независимыми гауссовскими возмущениями.")
    p("Сдвиг S одинаков для всех чтений одной цепочки в данной сессии. Он обновляется между соседними сессиями по стационарной AR(1). Параметры режима обозначены τ для масштаба сдвига и η для корреляции соседних сессий.")
    centered(doc, "S₁ = ηS₀ + U₁,   U₁ ~ N(0, τ²(1 − η²)).")
    table(doc, ("Режим", "τ", "η", "Роль в сравнении"), [
        ("Без дрейфа", "0", "0", "Контроль только с шумом чтения"),
        ("Устойчивый дрейф", "0,22", "0,995", "Проверка сильной связи соседних сессий"),
        ("Изменчивый дрейф", "0,32", "0,12", "Проверка слабой связи при заметном сдвиге"),
    ])
    p("В калибровочной сессии симулированы два чтения для каждого из 512 входов. Для цели измеряется одно чтение по 2048 независимо сгенерированным входам следующей сессии. Параметры генерации не доступны регрессии. Режимы обучаются раздельно и задают контролируемые условия оценки, сравнение режимов само по себе не выделяет причинный вклад отдельного параметра.")
    p("Сохраняются только агрегаты по устройствам. Эксперимент не измеряет аппаратную надёжность и не проверяет неизвестные схемы или распределения.")

    new_page(doc, SECTIONS[5].upper())
    p("Три сравниваемых метода оценивают одно число от нуля до единицы. Базовое правило переносит текущий BER в следующую сессию. Ridge стандартизирует признаки и использует регуляризованную линейную регрессию. Нелинейная модель — гистограммный градиентный бустинг по деревьям [11, 12]. Прогнозы моделей ограничены диапазоном долей.")
    p("Вектор признаков включает текущий BER по двум чтениям, долю расхождений этих чтений, стандартное отклонение средних BER по восьми блокам, абсолютное смещение доли единиц и известное число цепочек. У обеих моделей и правила один и тот же текущий BER. Отдельные двоичные ответы, запросы, скрытые веса и будущие измерения в признаки не входят.")
    cfg = data["cfg"]
    p(f"В каждом режиме выделено {cfg['train_devices']} экземпляров для обучения, {cfg['validation_devices']} для выбора параметров и {cfg['test_devices']} для итогового теста. Идентификаторы и генерация экземпляров разделены между выборками. Параметр Ridge и число листьев дерева выбираются по MAE на валидации. Далее выбранные модели обучаются на объединении обучения и валидации, после чего используется тест.")
    p(f"Основная серия использует seed {cfg['seed']}, независимая серия — {data['repeat_cfg']['seed']}. Повтор заново генерирует все выборки и выполняет ту же процедуру выбора параметров. В CSV с идентификаторами можно проверить отсутствие пересечений внутри обеих серий и между ними. Отдельный тест подменяет будущую цель и проверяет неизменность калибровочных признаков.")
    p(f"Одна строка итогового теста соответствует одному устройству. MAE измеряет среднее абсолютное отклонение, RMSE сильнее взвешивает крупные ошибки. Интервалы 95% получены по {cfg['bootstrap_resamples']} бутстрэп-выборкам устройств [10]. Для выигрыша обе ошибки на одном экземпляре ресемплируются вместе. Интервалы точечные, условные на обученных моделях и отдельном режиме.")

    new_page(doc, SECTIONS[6].upper())
    figure(doc, results / "forecast" / "forecast.png",
           "Рисунок 1 — MAE и парный выигрыш на новых экземплярах, основной прогон")
    p("Таблица 3 — MAE и RMSE на новых экземплярах. Во всех ячейках единица измерения — процентный пункт.")
    table(doc, ("Режим", "Метод", "MAE, осн.", "RMSE, осн.", "MAE, повтор", "RMSE, повтор"), [
        (label, method_label, pct(source[regime, method]["mae"]), pct(source[regime, method]["rmse"]),
         pct(repeated[regime, method]["mae"]), pct(repeated[regime, method]["rmse"]))
        for regime, label in REGIMES.items() for method, method_label in MODELS.items()
    ])
    p(f"Изменчивый дрейф дал наибольшую абсолютную ошибку правила. Ridge снизила MAE с {pct(base['mae'])} до {pct(case['mae'])} п. п. В независимой серии изменение составило с {pct(repeated['volatile', 'baseline']['mae'])} до {pct(repeated['volatile', 'ridge']['mae'])} п. п. Бустинг также уменьшил MAE относительно правила. Выигрыш одной ML-модели над другой отдельно не тестировался.")

    new_page(doc, SECTIONS[7].upper())
    p("Таблица 4 — Выигрыш в MAE относительно текущего BER, п. п. В скобках точечные парные 95%-е интервалы. Положительное значение означает меньшую ошибку модели.")
    table(doc, ("Режим", "Метод", "Основной прогон", "Новый seed"), [
        (label, MODELS[method], gain_ci(source[regime, method]), gain_ci(repeated[regime, method]))
        for regime, label in REGIMES.items() for method in ("ridge", "boosting")
    ])
    figure(doc, results / "forecast" / "repeat" / "forecast.png",
           "Рисунок 2 — Независимый повтор с новым seed и новыми экземплярами")
    p("При устойчивом дрейфе правило переносит почти тот же сдвиг в следующую сессию. Оба парных интервала для Ridge и бустинга включают ноль в каждой серии. Знак оценённого выигрыша бустинга отрицателен, а интервал совместим и с небольшим улучшением. Для режима без дрейфа оценки выигрыша обеих моделей положительны, но абсолютная разница меньше, чем при изменчивом дрейфе.")
    p("Абсолютные интервалы MAE и RMSE, смещение оценок и правила выбора гиперпараметров доступны в CSV. В повторе меняются и обученные модели, и устройства. Заключение сохраняется по режимам, хотя конкретные ошибки различаются.")

    new_page(doc, SECTIONS[8].upper())
    p("Новые экземпляры из того же режима проверяют перенос между устройствами. Для применения результата важна также смена условий. Поэтому каждую модель дополнительно проверили на тестовых устройствах двух других режимов. При такой проверке параметры и стандартизация заморожены. Выбор модели использует только исходный режим. Базовое правило и ошибки вычислены на тех же целевых устройствах.")
    figure(doc, results / "forecast" / "transfer.png",
           "Рисунок 3 — Парный выигрыш по MAE. Интервалы в ячейках соответствуют 95%")
    p("Выбранные попарные сравнения приведены ниже. Положительный выигрыш означает уменьшение ошибки относительно правила текущего BER.")
    table(doc, ("Обучение → проверка", "Модель", "Основной, п. п.", "Повтор, п. п."), [
        (f"{REGIMES[origin]} → {REGIMES[target]}", MODELS[method],
         gain_ci(data["transfer"][origin, target, method]),
         gain_ci(data["transfer_repeat"][origin, target, method]))
        for origin, target, method in TRANSFER_CASES
    ])
    transfer = data["transfer"]["volatile", "stable", "ridge"]
    p(f"После переноса Ridge из изменчивого режима на случаи без дрейфа MAE стала {pct(transfer['mae'])} п. п. Правило текущего BER дало {pct(source['stable', 'baseline']['mae'])} п. п. Поэтому успешное обучение для исследованного режима ещё не обосновывает перенос на иные условия. Интервалы рассчитываются по целевым тестовым экземплярам и характеризуют симулированный сдвиг.")

    new_page(doc, SECTIONS[9].upper())
    p("Таблица 5 связывает вспомогательные эксперименты с вопросами проекта. Эти опыты не подменяют проверку качества модели на новых экземплярах.")
    table(doc, ("Эксперимент", "На какой вопрос отвечает", "Что контролирует"), [
        ("Сдвиг и корреляция цепочек", "Гарантирует ли отсутствие истории равновероятность?", "Выделяет симметрию и независимость как предположения."),
        ("Одна и отдельные сессии", "Как меняется наблюдаемая доля расхождений?", "Различает повторяемость внутри сессии и перенос во времени."),
        ("Коррелированный дрейф", "Всегда ли помогает распределённый эталон?", "Показывает, что устойчивый сдвиг меняет оценку смещения."),
        ("Калибровка по лагам", "Достаточно ли короткого наблюдения для оценки долгого дрейфа?", "Проверяет агрегированную кривую на иных лагах и устройствах."),
        ("BER следующей сессии", "Помогают ли агрегаты калибровочной сессии?", "Проверяет правило, регрессию и нелинейную модель в одной постановке."),
    ])
    data_gap = data["session_gap"]
    independent, persistent = data["sensitivity_pair"]
    p(f"В опыте на 4-XOR с сильным дрейфом переход к отдельной проверке увеличил долю расхождений на {pct(data_gap['held_minus_same'])} п. п. В отдельном контроле при одной и той же амплитуде дрейфа выигрыш распределённого эталона изменился с {pct(independent['rmse_gain'])} до {pct(persistent['rmse_gain'])} п. п. при усилении корреляции. Это мотивирует оценку прогноза BER раздельно по режимам.")
    p("На железе потребуются повторные измерения нескольких экземпляров, независимые сессии, контроль условий и эталон, качество которого измеряется самостоятельно. Численные показатели симуляции зависят от выбранного распределения дрейфа и конкретной модели чистого ответа.")

    new_page(doc, SECTIONS[10].upper())
    p("При отсутствии пар конкретного экземпляра симметричный независимый ансамбль даёт вероятность угадывания 50% даже при доступе к текущему запросу. Контроли со сдвигом и общей задержкой показывают, где исчезает эта гарантия. Эти результаты получены на собственном симуляторе.")
    p("При наличии ранее измеренных CRP опубликованные исследования предсказывают новые ответы некоторых Arbiter и XOR Arbiter PUF существенно лучше равновероятного угадывания. Поэтому отсутствие физического доступа в момент предсказания не является достаточным условием непредсказуемости. Значения точности относятся к первичным публикациям.")
    p(f"Для отдельной задачи агрегированного BER прогноз на новых экземплярах оказался полезен при изменчивом дрейфе. Основной выигрыш Ridge — {pct(case['mae_gain'])} п. п., парный 95%-й интервал {bounds(case, 'gain_ci_low', 'gain_ci_high')} п. п. При устойчивом дрейфе убедительного улучшения относительно текущего BER не получено. Результаты повторены на независимо сгенерированной выборке.")
    p("Проверка переноса показала, что модель, точная в своём режиме, может уступать простому правилу после изменения дрейфа. При внедрении надёжность прогноза следует перепроверять на новых измерительных условиях с тем же базовым сравнением.")
    p("Команды воспроизведения симуляции и экспериментов приведены в README. Статус заявленной темы и открытый пункт согласования с семинаристом указаны в таблице требований.")

    new_page(doc, SECTIONS[11].upper())
    p("Локальная демонстрация проверяет обработку одноразовых запросов при аутентификации [14]. У каждого тестового устройства свой случайный ключ.")
    p("Тег HMAC-SHA-256 вычисляется по контексту протокола, идентификатору устройства и случайному запросу проверяющего [13]. Проверяющий сверяет срок действия и тег. При успехе запрос погашается в той же критической секции.")
    checks = data["defense"]
    p("Таблица 6 — Исходы локальной демонстрации.")
    table(doc, ("Проверка", "Наблюдаемый исход"), [
        ("Первое предъявление действительного тега", "Принято" if checks["fresh_proof_accepted"] else "Ошибка"),
        ("Повтор ранее принятого тега", "Отклонён" if checks["repeated_proof_rejected"] else "Ошибка"),
        ("Тег с ключом другого устройства", "Отклонён" if checks["wrong_device_key_rejected"] else "Ошибка"),
        ("Предъявление после истечения срока", "Отклонено" if checks["expired_challenge_rejected"] else "Ошибка"),
        ("Два конкурентных предъявления", f"Принято {checks['concurrent_acceptance_count']} из 2"),
    ])
    p("Демонстрация работает в одном процессе с программными тестовыми ключами. Команда запуска — python -m puf_lab.defensive_auth --output build/defense-demo. Результаты тестов сохраняются в nonce_checks.json.")

    new_page(doc, SECTIONS[12].upper())
    p("Точная формулировка РУП и переписка разобраны в report/requirements.md. В РУП тема указана в перечне групповых заданий, критерий верхней базовой оценки требует полной реализации, а рекомендуемый размер группы составляет от трёх до пяти участников [1].")
    table(doc, ("Требование", "Подтверждение", "Статус"), [
        ("Тема № 12 и группа из трёх человек", "РУП, переписка и план подразделения работ", "Зафиксировано"),
        ("Самостоятельная атака на ответы конкретного PUF", "Раздел 3 содержит обзор опубликованных результатов", "Открыто. Требуется согласование замены"),
        ("Сценарий без CRP", "Условие 50%, симуляция и контроли", "Выполнено"),
        ("Сценарий с CRP", "Первичные работы и таблица точных указателей", "Выполнено"),
        ("Собственная ML-оценка BER", "Две независимые серии, парные интервалы, аудит идентификаторов", "Выполнено"),
        ("Проверка материалов и запуск", "README, тесты, манифесты, таблицы и файлы защиты", "Выполнено"),
    ])
    p("Дополнительные баллы оцениваются преподавателем по условиям РУП. Полученные результаты описываются в пределах симулятора. Данное исследование не устанавливает новизну результата по всем опубликованным работам.")

    new_page(doc, SECTIONS[13].upper())
    for number, item in enumerate(SOURCES, 1):
        source = p(f"{number}. {item}")
        source.alignment = WD_ALIGN_PARAGRAPH.LEFT
        source.paragraph_format.space_after = Pt(3)
    doc.save(output / "report.docx")


INK = "132736"
DEEP = "102833"
TEAL = "00756F"
LIGHT = "E5F1EE"
MUTED = "486471"
PAPER = "F7FAF9"
WHITE = "FFFFFF"


def block(slide, x, y, width, height, color):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(color)
    shape.line.fill.background()


def text(slide, value, x, y, width, height, size, color=INK, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0)
    frame.margin_top = frame.margin_bottom = Inches(0)
    for index, line in enumerate(value.split("\n")):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.space_after = SlidePt(4)
        r = p.add_run()
        r.text = line
        r.font.name = "Arial"
        r.font.size = SlidePt(size)
        r.font.bold = bold
        r.font.color.rgb = RGBColor.from_string(color)


def page(deck, section, heading):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(PAPER)
    block(slide, 0.65, 0.52, 0.32, 0.05, TEAL)
    text(slide, section.upper(), 1.10, 0.42, 11.55, 0.27, 11, TEAL, True)
    text(slide, heading, 0.65, 0.87, 12.01, 0.76, 31, INK, True)
    return slide


def tiles(slide, items, y=2.2):
    gap = 0.2
    width = (12.0 - gap * (len(items) - 1)) / len(items)
    for index, (value, explanation) in enumerate(items):
        x = 0.65 + index * (width + gap)
        block(slide, x, y, width, 1.65, LIGHT)
        text(slide, value, x + 0.19, y + 0.23, width - 0.38, 0.58, 28, TEAL, True)
        text(slide, explanation, x + 0.19, y + 0.95, width - 0.38, 0.65, 15)


def build_slides(output, results, data):
    deck = Presentation()
    deck.slide_width = Inches(13.333333)
    deck.slide_height = Inches(7.5)
    deck.core_properties.author = AUTHORS
    deck.core_properties.title = TITLE
    primary, repeated = data["main"], data["repeat"]

    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(DEEP)
    block(slide, .7, .78, .54, .07, "55D5BD")
    text(slide, "МФТИ  /  ЗАЩИТА ИНФОРМАЦИИ", 1.42, .65, 11.1, .32, 13, "85DBCB", True)
    text(slide, "Предсказуемость PUF\nи надёжность следующей сессии",
         .7, 1.65, 11.7, 2.14, 41, WHITE, True)
    text(slide, "Тема № 12 «Реализация атак на PUF»", .73, 4.16, 11.5, .55, 22, "85DBCB")
    text(slide, AUTHORS, .73, 6.62, 11.7, .40, 17, WHITE)

    slide = page(deck, "Постановка", "Что было известно наблюдателю?")
    tiles(slide, [
        ("Без истории", "Ранее измеренных пар конкретного экземпляра никогда не было"),
        ("С историей", "Ранее измеренные пары конкретного экземпляра уже доступны"),
    ])
    text(slide, "В обоих сценариях физический доступ в момент предсказания отсутствует.",
         .65, 4.5, 12, .96, 23, INK, True)
    text(slide, "Эти сценарии задают разные условные вероятности ответа.",
         .65, 5.72, 12, .71, 19, MUTED)

    slide = page(deck, "Сценарий без CRP", "Когда вероятность равна 50%")
    tiles(slide, [
        ("50%", "При условной симметрии, независимых знаках и отсутствии информации об экземпляре"),
        (f"{pct(data['control'][1, .6]['theory_majority_accuracy'])}%", "Точность константы при сдвиге одной цепочки"),
        (f"{pct(data['spatial']['theory_constant_accuracy'])}%", "Точность константы 4-XOR при корреляции 0,60"),
    ])
    text(slide, "Если хотя бы один знак независим и равновероятен, XOR остаётся равновероятным.",
         .65, 4.58, 12, .92, 22, INK, True)
    text(slide, "Два контроля на собственном симуляторе нарушают разные предположения.",
         .65, 5.75, 12, .66, 18, MUTED)

    slide = page(deck, "Сценарий с CRP", "Опубликованные контрпримеры")
    tiles(slide, [
        ("≈90%", "Arbiter PUF · измеренные CMOS"),
        ("≈87%", "2-XOR · чётность измеренных CMOS-цепочек"),
        ("99%", "5-XOR · опубликованные строки FPGA и ASIC"),
    ])
    text(slide, "Точность на новых входах при наличии ранее измеренных пар.",
         .65, 4.5, 12, .62, 22, INK, True)
    text(slide, "Hospodar et al., § III и рис. 1(a)–(b). Rührmair et al., табл. XI, строки FPGA и ASIC.",
         .65, 5.65, 12, .83, 16, MUTED)

    slide = page(deck, "Собственный вопрос", "Можно ли оценить BER следующей сессии?")
    tiles(slide, [
        ("512 входов", "Калибровка, два чтения на вход"),
        ("5 признаков", "Агрегаты текущей сессии и число цепочек"),
        ("2048 входов", "Оценка BER следующей сессии"),
    ])
    text(slide, "Цель — одна доля ошибок на устройство и сессию.",
         .65, 4.48, 12, .62, 23, INK, True)
    text(slide, "Чистый эталон известен в симуляторе. Аппаратный перенос потребует своего эталона.",
         .65, 5.55, 12, .84, 18, MUTED)

    slide = page(deck, "Симулятор", "Что меняется между сессиями")
    tiles(slide, [
        ("Без дрейфа", "Только независимый шум чтения"),
        ("Устойчивый", "Корреляция сдвигов 0,995"),
        ("Изменчивый", "Корреляция сдвигов 0,12"),
    ])
    text(slide, "Веса устройства остаются прежними, сдвиг меняется по AR(1).",
         .65, 4.49, 12, .59, 23, INK, True)
    text(slide, "Вспомогательный контроль показывает, почему условия сессии важны для оценки повторяемости.",
         .65, 5.55, 12, .85, 18, MUTED)

    slide = page(deck, "Протокол", "Новые устройства в каждой выборке")
    tiles(slide, [
        ("1024", "Обучение"),
        ("256", "Выбор параметров"),
        ("512", "Финальная проверка"),
    ])
    text(slide, "Размеры указаны для каждого режима. Повтор заново генерирует все выборки.",
         .65, 4.49, 12, .9, 21, INK, True)
    text(slide, "Признаки вычисляются до генерации будущей сессии. Интервалы парные по устройствам.",
         .65, 5.64, 12, .86, 18, MUTED)

    slide = page(deck, "Модели", "Что сравниваем на одинаковых данных")
    tiles(slide, [
        ("Текущий BER", "Простое правило переноса"),
        ("Ridge", "Регуляризованная линейная регрессия"),
        ("Бустинг", "Нелинейная модель по деревьям"),
    ])
    text(slide, "Выбор параметров — по MAE на валидации. Итоговая проверка использует отдельный тест.",
         .65, 4.5, 12, .96, 21, INK, True)
    text(slide, "Выигрыш измеряем в процентных пунктах MAE относительно простого правила.",
         .65, 5.72, 12, .80, 18, MUTED)

    slide = page(deck, "Основной прогон", "Выигрыш зависит от дрейфа")
    slide.shapes.add_picture(str(results / "forecast" / "forecast.png"),
                             Inches(.83), Inches(1.68), width=Inches(11.6))
    text(slide, f"При изменчивом дрейфе Ridge снизила MAE с {pct(primary['volatile', 'baseline']['mae'])} до {pct(primary['volatile', 'ridge']['mae'])} п. п.",
         .7, 6.56, 12, .48, 20, TEAL, True)

    slide = page(deck, "Независимый повтор", "Где правило текущего BER удержалось")
    slide.shapes.add_picture(str(results / "forecast" / "repeat" / "forecast.png"),
                             Inches(.83), Inches(1.68), width=Inches(11.6))
    text(slide, f"Изменчивый дрейф — {pct(repeated['volatile', 'baseline']['mae'])} → {pct(repeated['volatile', 'ridge']['mae'])} п. п. При устойчивом дрейфе интервалы включают ноль.",
         .7, 6.49, 12, .56, 17, TEAL, True)

    slide = page(deck, "Смена условий", "Ошибка после переноса модели")
    slide.shapes.add_picture(str(results / "forecast" / "transfer.png"),
                             Inches(1.42), Inches(1.53), width=Inches(10.49))
    transfer = data["transfer"]["volatile", "stable", "ridge"]
    text(slide, f"Ridge из изменчивого режима на тесте без дрейфа — {pct(transfer['mae'])} п. п. Правило — {pct(primary['stable', 'baseline']['mae'])} п. п.",
         .72, 6.56, 12, .52, 17, TEAL, True)

    slide = page(deck, "Границы вывода", "Что показали эксперименты")
    text(slide, "Без истории пар 50% следует из условий симметрии. Общий фактор может сместить XOR.",
         .65, 2.02, 11.9, .89, 23, INK, True)
    text(slide, "Со старыми парами известны аппаратные контрпримеры непредсказуемости.",
         .65, 3.23, 11.95, .86, 21)
    text(slide, "Наша ML-часть оценивает агрегированный BER следующей сессии на симуляторе.",
         .65, 4.45, 12, .85, 21)
    text(slide, "Открытый пункт для согласования с семинаристом указан в таблице требований.",
         .65, 5.68, 12, .90, 19, TEAL, True)

    slide = page(deck, "Демонстрация", "Проверка одноразового запроса")
    checks = data["defense"]
    tiles(slide, [
        ("Принят" if checks["fresh_proof_accepted"] else "Ошибка",
         "Действительный тег для свежего запроса"),
        ("Отклонён" if checks["repeated_proof_rejected"] else "Ошибка",
         "Повтор действительного тега"),
        (f'{checks["concurrent_acceptance_count"]} из 2', "Принято при конкурентном повторе"),
    ])
    text(slide, "Прототип HMAC-SHA-256. Проверены срок запроса и привязка к устройству.",
         .65, 4.5, 12, .88, 21, INK, True)
    text(slide, "Запуск в одном процессе с программными тестовыми ключами.",
         .65, 5.62, 12, .72, 18, MUTED)

    slide = page(deck, "Воспроизводимость", "Данные, модели и проверка")
    text(slide, "Максим — симулятор и калибровочные агрегаты.\nИван — модели и первичные публикации.\nАнтон — дизайн, статистика и повторный запуск.",
         .65, 1.98, 12, 2.27, 21)
    block(slide, .65, 5.11, 12, 1.13, LIGHT)
    text(slide, "python -m puf_lab.forecast_cli --profile smoke --output build/demo",
         .87, 5.53, 11.6, .38, 17, TEAL, True)
    deck.save(output / "slides.pptx")


def load_data(results):
    evidence = read_csv(Path(__file__).resolve().parent, "evidence")
    forecast = results / "forecast"
    repeat = forecast / "repeat"

    def metrics(directory):
        return {(row["regime"], row["model"]): row for row in read_csv(directory, "metrics")}

    def transfer(directory):
        return {(row["source_regime"], row["regime"], row["model"]): row
                for row in read_csv(directory, "transfer_metrics")}

    common = {"chains": 4, "correlation": .6}
    temporal = read_csv(results / "sensitivity", "temporal_persistence")
    return {
        "evidence": evidence,
        "defense": json.loads((results / "defense" / "nonce_checks.json").read_text(encoding="utf-8")),
        "control": {(int(r["chains"]), float(r["mean_delay_offset"])): r
                    for r in read_csv(results, "unseen_device")},
        "spatial": cell(read_csv(results / "sensitivity", "spatial_bias"), **common),
        "spatial_repeat": cell(read_csv(results / "sensitivity" / "repeat", "spatial_bias"), **common),
        "session_gap": cell(read_csv(results, "session_contrasts"),
                            chains=4, noise=.1, drift=.3, budget=4800, reads=1),
        "sensitivity_pair": tuple(cell(temporal, chains=4, drift=.3, reads=9, persistence=eta)
                                  for eta in (0, .98)),
        "main": metrics(forecast), "repeat": metrics(repeat),
        "transfer": transfer(forecast), "transfer_repeat": transfer(repeat),
        "cfg": json.loads((forecast / "manifest.json").read_text(encoding="utf-8")),
        "repeat_cfg": json.loads((repeat / "manifest.json").read_text(encoding="utf-8")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("report"))
    args = parser.parse_args()
    data = load_data(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    build_word(args.output, args.results, data)
    build_slides(args.output, args.results, data)


if __name__ == "__main__":
    main()
