"""Report the locally frozen independent-cohort observation-likelihood study."""
import csv
import json

LABELS = {'single': 'Разные запросы', 'majority_naive': 'Наивное большинство',
          'majority_aware': 'Учёт вероятности большинства', 'fraction': 'Доли ответов'}


def read(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def load(results):
    root = results / 'observation_study'
    return {part: {'metrics': read(folder / 'metrics.csv'), 'contrasts': read(folder / 'contrasts.csv'),
                   'devices': read(folder / 'per_device.csv'), 'manifest': json.loads((folder / 'manifest.json').read_text())}
            for part, folder in (('main', root), ('repeat', root / 'repeat'))}


def metric(data, method, repeated=False):
    return next(r for r in data['observation']['repeat' if repeated else 'main']['metrics'] if r['method'] == method)


def contrast(data, first='fraction', second='majority_aware', repeated=False):
    return next(r for r in data['observation']['repeat' if repeated else 'main']['contrasts']
                if r['first'] == first and r['second'] == second and r['endpoint'] == 'accuracy')


def interval(row):
    from build import pct
    return f"{pct(row['difference'])} [{pct(row['ci_low'])}; {pct(row['ci_high'])}]"


def conclusion(data):
    a, b = contrast(data), contrast(data, repeated=True)
    if float(a['ci_low']) > 0 and float(b['ci_low']) > 0:
        finding = 'Преимущество долей сохранилось в обеих независимых сериях при учёте вероятности голосования.'
    else:
        finding = 'Положительное преимущество долей не подтверждено одновременно в обеих независимых сериях.'
    return (f'На 16 + 16 новых устройствах разность «доли − учёт большинства» составила {interval(a)} п. п.; '
            f'повтор — {interval(b)} п. п. {finding}')


def add_word(doc, results, data):
    from build import SECTIONS, new_page, table, figure, pct
    p = doc.add_paragraph
    new_page(doc, SECTIONS[20].upper())
    p('После исследования раздела 19 поставлен отдельный вопрос: сохраняется ли преимущество долей, если модель учитывает распределение самого наблюдения после голосования? Старые устройства уже участвовали в выборе исследовательского направления, поэтому они не входят в новую проверку. Пилот использовал отдельный seed 190001. До генерации основных устройств локально зафиксированы протокол и хеши научного кода; внешней регистрации или независимого заверения времени не было.')
    table(doc, ('Параметр', 'Зафиксированное значение'), [
        ('Экземпляры', '16 + 16 новых 64-битных 2-XOR; seed 190031 и 190032'),
        ('Наблюдения', 'σ = 0,25; независимые чтения; r = 9'),
        ('Физический бюджет', '4608 обучающих + 512 общих валидационных обращений'),
        ('Разные запросы', '4608 при одном чтении; 512 при девяти чтениях'),
        ('Выбор модели', '4 старта × 4 L2-коэффициента; максимум 1000 итераций'),
        ('Проверка', '8192 новых одиночных ответов; 8000 бутстрэп-выборок по устройствам'),
    ])
    p('Пусть uⱼ = wⱼ/σ — неизвестные параметры, qⱼ = Φ(−uⱼᵀφ(c)). Вероятность итоговой единицы при двух независимых цепочках p(c) = q₁(1 − q₂) + (1 − q₁)q₂. Значение σ отдельно алгоритму не требуется: оно поглощено параметрами u. Все четыре атаки используют это же семейство вероятностей. Истинные u, задержки и вероятности доступны только оценщику.')
    p('Для девяти независимых чтений вероятность большинства G₉(p) = Σₖ₌₅⁹ C(9,k)pᵏ(1 − p)⁹⁻ᵏ. Сравниваются разные запросы, наивная BCE по большинству с вероятностью p, BCE по большинству с вероятностью G₉(p) и BCE по доле с вероятностью p. Последняя соответствует отрицательному логарифму биномиального правдоподобия после нормирования и отбрасывания константы. Замена p на G₉(p) учитывает голосование; эти вероятности и биномиальная формула не заявляются новыми.')
    p('Потери нормированы на физическое число обучающих чтений: BCE большинства делится на r, BCE долей уже усредняет все r битов. L2 выбирается отдельно для каждого метода из {10⁻⁵, 10⁻⁴, 10⁻³, 10⁻²}. Начальные веса одинаковы между методами и коэффициентами. Из 16 кандидатов выбирается минимум log loss на общих 512 одиночных валидационных ответах. Тест не участвует в выборе. Равенство бюджета данных не означает равенство времени вычислений.')
    p('Основной контраст заранее задан: accuracy долей минус accuracy модели G₉ на новых одиночных ответах. Другие контрасты и метрики описательные. Сохраняются все устройства и кандидаты, включая неудачные попытки. Зафиксированный протокол: configs/observation_study.json; контроль его целостности: observation_protocol_lock.json. Локальная фиксация снижает риск подгонки этой проверки, но не превращает предшествующий поиск гипотезы в заранее зарегистрированное исследование.')

    new_page(doc, SECTIONS[21].upper())
    p('Таблица 13 — Независимая проверка вероятности наблюдения. Точность относится к одному новому шумному ответу; в каждой серии 16 устройств. Все варианты имеют общий бюджет 5120 обращений с учётом валидации.')
    table(doc, ('Метод', 'Основная серия, %', 'Повтор, %', 'Сошлись, осн. / повт.'), [
        (label, pct(metric(data, method)['accuracy']), pct(metric(data, method, True)['accuracy']),
         f"{metric(data,method)['selected_converged']} / {metric(data,method,True)['selected_converged']}")
        for method, label in LABELS.items()
    ], widths=(6,3.4,3.0,4.0))
    p('Таблица 14 — Парные разности accuracy, п. п., с 95%-ми интервалами. Только первая строка — основной контраст; остальные интервалы точечные и описательные.')
    pairs=(('fraction','majority_aware'),('majority_aware','majority_naive'),('fraction','single'))
    table(doc, ('Контраст', 'Основная серия, п. п.', 'Повтор, п. п.'), [
        (f'{LABELS[a]} − {LABELS[b]}', interval(contrast(data,a,b)), interval(contrast(data,a,b,True)))
        for a,b in pairs
    ], widths=(6.4,5,5))
    figure(doc, results / 'observation_study/observation_effects.png', 'Рисунок 9 — Эффекты способов использования наблюдений на новых устройствах.')
    p(f"Доли точнее модели большинства в обеих сериях. При этом разные запросы дали на {pct(-float(contrast(data,'fraction','single')['difference']))} и {pct(-float(contrast(data,'fraction','single',True)['difference']))} п. п. больше, чем доли. Учёт вероятности большинства не дал подтверждённого выигрыша над наивным вариантом.")
    p('Вывод ограничен 2-XOR, независимым гауссовским шумом, данным бюджетом и сеткой настройки. Здесь новые устройства и другое семейство вероятностей, поэтому прямое сравнение с числами раздела 19 некорректно. Сходимость не гарантирует глобальный минимум. Мировой приоритет и превосходство над опубликованными атаками не установлены.')



def add_slides(deck, results, data):
    from build import page, text, tiles, pct, INK, TEAL
    slide = page(deck, 'Независимая проверка', 'Моделировать то, что действительно наблюдается')
    tiles(slide, [('16 + 16', 'Новые устройства после фиксации протокола'),
                  ('4 × 4', 'Старты и L2; выбор по валидации'), ('5120', 'Равный общий бюджет физических чтений')])
    text(slide, 'Для большинства девяти ответов вероятность метки — G₉(p), а не p одного чтения.', .65,4.4,12,1.,23,INK,True)
    text(slide, 'Сравнение долей с корректной моделью голосования задано до запуска новых устройств.', .65,5.8,12,1.,21,TEAL,True)
    slide = page(deck, 'Результат независимой проверки', 'Доли ответов против модели вероятности большинства')
    a,b=contrast(data),contrast(data,repeated=True)
    tiles(slide, [(f"{pct(a['difference'])} п. п.",f"95% ДИ [{pct(a['ci_low'])}; {pct(a['ci_high'])}]"),
                  (f"{pct(b['difference'])} п. п.",f"Повтор: [{pct(b['ci_low'])}; {pct(b['ci_high'])}]")])
    text(slide, '2-XOR, σ = 0,25, девять независимых чтений. Новые одиночные ответы на тесте.',.65,4.4,12,1.,23,INK,True)
    text(slide, f"Разные запросы всё же точнее долей: +{pct(-float(contrast(data,'fraction','single')['difference']))} и +{pct(-float(contrast(data,'fraction','single',True)['difference']))} п. п. Вывод ограничен проверенными условиями.",.65,5.8,12,1.,20,TEAL,True)
