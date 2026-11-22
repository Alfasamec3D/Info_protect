"""Analyze all pre-specified primary contrasts; build a report from actual results."""
import csv
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/puf-matplotlib')
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from method import stopped_mean
from run import signature
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/stopping_confirmatory'
REPORT=ROOT/'report'
manifest=json.loads((OUT/'manifest.json').read_text())
config=manifest['config']
assert manifest['sha256']==signature(config),'source has changed'
rows=[json.loads(s) for s in (OUT/'runs.jsonl').read_text().splitlines()]
expected=len(config['seeds'])*len(config['sigmas'])*len(config['budgets'])*len(config['caps'])*4
assert len(rows)==expected
assert len({(r['seed'],r['sigma'],r['budget'],r['cap'],r['method']) for r in rows})==expected
assert all(r['converged'] for r in rows)
assert all(r['queries']<=r['budget'] for r in rows)
assert all(r['budget']-r['queries']<r['cap'] for r in rows if r['method'] in ['unweighted','count_likelihood'])
contrasts=[];means=[]
for sigma in config['sigmas']:
 for budget in config['budgets']:
  for cap in config['caps']:
   cell=[r for r in rows if (r['sigma'],r['budget'],r['cap'])==(sigma,budget,cap)]
   by={method:{r['seed']:r for r in cell if r['method']==method}
       for method in ['unweighted','count_likelihood','fixed5','single']}
   for method,items in by.items():
    means.append(dict(sigma=sigma,budget=budget,cap=cap,method=method,
       **{metric:float(np.mean([r[metric] for r in items.values()]))
          for metric in ['probability_mse','expected_nll','accuracy','queries','unique']}))
   for a,b in [('count_likelihood','unweighted'),('count_likelihood','single')]:
    for metric in ['probability_mse','expected_nll','accuracy']:
     delta=np.array([by[a][s][metric]-by[b][s][metric] for s in config['seeds']])
     primary=(a,b,metric)==('count_likelihood','unweighted','probability_mse')
     n=len(delta);mean=float(delta.mean())
     half=float(t.ppf(1-.05/(2*(8 if primary else 1)),n-1)*delta.std(ddof=1)/np.sqrt(n))
     contrasts.append(dict(sigma=sigma,budget=budget,cap=cap,a=a,b=b,metric=metric,primary=primary,
                           n=n,difference=mean,ci_low=mean-half,ci_high=mean+half))
for name,data in [('contrasts',contrasts),('summary',means)]:
 with (OUT/f'{name}.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
primary=[r for r in contrasts if r['primary']]
print('All models:',len(rows),'Primary contrasts:')
for r in primary:print(r)

def rows_table():
 lines=[]
 for r in primary:
  m=[v for v in means if (v['sigma'],v['budget'],v['cap'])==(r['sigma'],r['budget'],r['cap'])]
  b={v['method']:v for v in m}
  reduction=100*(1-b['count_likelihood']['probability_mse']/b['unweighted']['probability_mse'])
  lines.append(f"σ={r['sigma']}, B={r['budget']}, M={r['cap']}: MSE "
     f"{b['unweighted']['probability_mse']:.6f} → {b['count_likelihood']['probability_mse']:.6f}; "
     f"снижение {reduction:.1f}%.\n"
     f"  Δ={r['difference']:.6f}; семейный 95% CI [{r['ci_low']:.6f}; {r['ci_high']:.6f}].")
 return '\n'.join(lines)

sections=[
('Исследовательский результат: адаптивные повторы',
'''Смещение мягких меток при адаптивном сборе CRP и его учёт в вероятностной модели Arbiter PUF.

Команда проекта: Иван Шаверин, Антон Хмельницкий, Максим Рачинский.
Вычислительное исследование, 23 ноября 2026 года.

Постановка: число повторных запросов зависит от уже полученных ответов. После трёх одинаковых ответов сбор по этому challenge прекращается; иначе выполняется M измерений, M=5 или 15. На таких данных обычное обучение по долям ответов и обучение по числу положительных/отрицательных наблюдений имеют разные целевые функции.

В проекте получены: точная формула смещения доли ответов; доказательство ненулевого градиента популяционного риска обычных мягких меток в истинных параметрах probit-модели; реализация корректного правдоподобия остановленных наблюдений; проверка на 100 независимых PUF в восьми режимах. Это уточняет ранее выполненное исследование фиксированных повторов.

Теория последовательного оценивания, идея вероятностных ответов PUF и взвешивание наблюдений существуют в литературе. Собственный вклад ограничен сформулированной постановкой, выводами и экспериментом. В изученных первичных разделах точный аналог этой совокупности не обнаружен. Это не доказательство отсутствия аналога во всей литературе; ближайшая статья 2026 года доступна пока только в виде аннотации.'''),
('1. Точная формула смещения',
'''Пусть для фиксированного challenge наблюдаются независимые Y_t~Bernoulli(p). Положим S_3=Y_1+Y_2+Y_3. Если S_3 принадлежит {0,3}, N=3; иначе N=M≥3. S_N — сумма полученных ответов, Q=S_N/N.

Утверждение 1. E[Q|p] = f_M(p) = p+(1−3/M)p(1−p)(2p−1).
Также E[N|p]=3+3(M−3)p(1−p).

Доказательство. Вероятности событий S_3=3 и S_3=0 равны p³ и (1−p)³. При продолжении ожидаемая сумма дополнительных ответов равна (M−3)p. Поэтому
E[Q]=p³+{3p−3p³+(M−3)p[1−p³−(1−p)³]}/M.
Раскрытие скобок даёт указанную формулу. Для N используются те же вероятности продолжения.

Следствие. При p>1/2 доля систематически завышается, при p<1/2 — занижается. Максимальное абсолютное смещение равно (1−3/M)/(6√3) и достигается при p=(3±√3)/6. Это 0.03849 для M=5 и 0.07698 для M=15. При p=0.8 ожидаемая доля равна 0.8384 либо 0.8768 соответственно.

Это смещение не исчезает от увеличения числа разных challenge: оно входит в ожидаемую обучающую цель. Оно относится к конкретному правилу остановки. Для фиксированного N доля несмещённая.

Доказательство проверено независимым полным перебором всех остановленных бинарных путей для M=3,5,7, нескольких p, а не только симуляцией.'''),
('2. Почему обычная soft-BCE неверно калибруется',
'''В аддитивной модели с независимым гауссовским шумом p(c)=Φ(β·φ(c)), где β=w/σ. Для обучения β знание σ отдельно не требуется. Пусть z=β·φ(c), g(z) — стандартная нормальная плотность.

Обычная soft-BCE даёт каждому challenge одинаковый вес и минимизирует BCE(Q,p_θ(c)). Для неограниченного предсказателя её популяционный минимум равен E[Q|c]=f_M(p(c)), а не истинной p(c).

Утверждение 2. При M>3 и ненулевой вероятности z≠0 истинный β не стационарен даже в правильно заданном классе probit-моделей. Рассмотрим направление θ=αβ. Производная ожидаемой невзвешенной BCE при α=1 равна
−(1−3/M) E[(2Φ(z)−1)g(z)z] < 0.

Доказательство. Для одного c производная равна [p−E(Q|c)]g(z)z/[p(1−p)]. Подстановка утверждения 1 сокращает p(1−p). При z≠0 величины 2Φ(z)−1 и z имеют одинаковый знак. Следовательно, увеличение масштаба параметров первоначально уменьшает ошибочную целевую функцию и усиливает уверенность.

Утверждение относится к популяционному риску без регуляризации и полным группам измерений. В коде есть фиксированная L2=10⁻⁶ для численной устойчивости; мы не приписываем ей точную несмещённость. Наличие смещения вероятностей не означает обязательного изменения решения при пороге 0.5: f_M монотонна и сохраняет этот порог. Поэтому основной критерий исследования — ошибка вероятностей, а accuracy является вторичным.'''),
('3. Корректное правдоподобие и границы вывода',
'''Для наблюдённого остановленного пути его вероятность имеет вид h(path)·p_θ^S·(1−p_θ)^(N−S), где h задаёт допустимость пути и не зависит от неизвестного θ. При хранении только (S,N) появляется коэффициент числа допустимых путей, также не зависящий от θ.

Поэтому отрицательный логарифм правдоподобия с точностью до константы:
L(θ)=−Σ[S_i log p_θ(c_i)+(N_i−S_i)log(1−p_θ(c_i))].
Эквивалентная запись — Σ N_i·BCE(S_i/N_i,p_θ(c_i)). Деление всей суммы на общее число измерений не меняет минимум без регуляризации. Важен вес N_i перед каждой группой; нормировать каждую группу отдельно и затем усреднять — другая задача.

Для ограниченной остановки E[S_N−pN|c]=0: это следует из суммы условных ожиданий Y_t−p по наблюдаемым шагам либо прямого перебора. Следовательно, score корректного правдоподобия имеет нулевое ожидание. Мы не утверждаем, что конечновыборочная MLE несмещённа: это отличное и в общем неверное утверждение [STAT2016, WHITEHEAD].

В случае фиксированного N взвешенная и невзвешенная цели совпадают. Это проверено как отрицательный контроль. Поправка использует лишь наблюдаемые S,N и challenge, а не секретные веса и не истинную вероятность ответа.

Механизм остановки должен определяться доступной историей. Результат нельзя переносить без проверки на скрытую фильтрацию CRP, отбрасывание неудобных наблюдений, коррелированный шум и остановку по неизвестным параметрам.'''),
('4. Протокол независимой проверки',
'''Пилот: пять экземпляров seed 2001–2005. После него зафиксирован протокол research/stopping/protocol.json; конфигурации и алгоритмы не менялись перед основной проверкой. Это локально зафиксированный план, не внешняя preregistration.

Основная проверка: 100 новых экземпляров seed 3001–3100, n=64, k=1. Шум σ∈{0.25,0.75}, бюджет B∈{4096,16384}, максимум повторов M∈{5,15}: восемь ячеек. В каждой обучаются четыре модели: unweighted, count_likelihood, fixed5, single. Итого 3200 обучений, но независимых устройств — 100, а не 3200.

Обе адаптивные модели получают идентичные данные. fixed5 и single — контроли при том же верхнем бюджете. Перед новой группой резервируется M обращений, поэтому остаток адаптивного бюджета меньше M. Фактическая цена каждой группы и суммарный счётчик оракула сохраняются. Контроли могут использовать на несколько запросов больше, что раскрыто в CSV.

Гиперпараметры фиксированы. Подбора на validation/test нет, поэтому весь B используется для обучения. Train и test непересекаются; test содержит 10000 новых challenge. Истинная p(c) используется только внешним оценщиком. Обучение не получает σ, внутренние задержки или ответы отдельных цепочек.

Первичная метрика — средний квадрат ошибки предсказанной вероятности относительно p(c); для бинарного исхода это избыток ожидаемого Brier score над неустранимой дисперсией. Для восьми парных контрастов count_likelihood−unweighted применены двухсторонние t-интервалы с Bonferroni, семейный уровень 95%. Остальные метрики и сравнения являются вторичными, исследовательскими.'''),
('5. Результаты по всем заранее выбранным режимам',rows_table()),
('6. Сравнение с ближайшими работами',
'''[DV] Delvaux, Verbauwhede: в полном тексте, раздел IV-A, рассматриваются доли ответов после M повторов, LMS и преобразование через обратную erfc. В IV-D обсуждается адаптивный выбор challenge. Это исключает заявления, что использование шума, долей или адаптации вообще придумано здесь. Наше отличие: число повторов одного challenge зависит от его ответов; исследуется точное искажение soft-метки и калибровка модели.

[LZ] Li, Zhuang, arXiv:2405.13147v2: полный HTML описывает LDHF-представление надёжности и атаки на PUF с majority voting. Проверяемая нами ошибка относится к остановке сбора измерений и их весам в loss. Мы не заявляем превосходство над LDHF и не воспроизводим его.

[CW2026] PUF Side-Channel Modeling Attack Based on Confidence Weighting of Response Repeatability, ITC-Asia 2026: аннотация уже описывает WLS с вероятностями и числами измерений. Поэтому «взвесили ответы» не является достаточным отличием. Полный текст недоступен; отсутствие точного пересечения с этой статьёй не доказано.

[STAT2016] Общая теория смещения при остановке и joint likelihood давно известна. Мы используем её, а не заявляем новой. Конкретные формулы 3-or-M, отрицательная производная риска в истинном параметре и PUF-эксперимент выведены и выполнены в проекте.

Честная формулировка вклада: исследование ошибки калибровки вероятностной моделирующей атаки при зависящем от ответов сборе CRP, с аналитическим описанием ошибки и проверкой корректного likelihood на новых устройствах. Это собственный проверяемый результат; утверждение «впервые в мировой литературе» пока не обосновано.'''),
('7. Ограничения и воспроизведение',
'''Все PUF симулированы. Доказательства предполагают условно независимые Бернулли-ответы с постоянной p(c); соответствие probit-ссылке даёт гауссовский шум. Систематический дрейф или корреляция могут нарушить калибровку даже правильного для iid случая likelihood.

Улучшение probability MSE не равнозначно увеличению точности бинарного взлома. Полные результаты single и fixed5 публикуются: исправление ошибки подготовки данных само по себе не доказывает превосходства адаптивного сбора над новым запросом за каждую единицу бюджета.

Нулевое ожидание score не является обещанием несмещённости конечновыборочных параметров. Сохранены все заранее заданные экземпляры, не только успешные. Интервалы учитывают разброс устройств; seed оптимизатора не варьируется, поскольку L-BFGS детерминирован с нулевой инициализацией.

Воспроизведение из корня репозитория:
python -m unittest discover -s research/stopping -p 'test_*.py' -v
python research/stopping/run.py --config configs/stopping_confirmatory.json --output results/stopping_confirmatory
python research/stopping/analyze.py

Рекомендуется OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1. Для нового независимого запуска выберите новую папку output. Сводка, парные интервалы, исходные метрики и модели хранятся в results/stopping_confirmatory. SHA-256 исходников и конфигурации проверяется при анализе. Исследовательский модуль отделён от прежнего кода, чтобы не менять provenance старых экспериментов.'''),
('Литература и пределы проверки новизны',
'''[DV] J. Delvaux, I. Verbauwhede. Side Channel Modeling Attacks on 65nm Arbiter PUFs Exploiting CMOS Device Noise.
https://cosicdatabase.esat.kuleuven.be/backend/publications/files/conferencepaper/2344

[LZ] G. Li, Y. Zhuang. A Low-Dimension High-Fidelity Reliability Representation for Reliability-based Machine Learning Modeling of Physical Unclonable Functions. arXiv:2405.13147v2.
https://arxiv.org/html/2405.13147v2

[CW2026] PUF Side-Channel Modeling Attack Based on Confidence Weighting of Response Repeatability. ITC-Asia 2026. DOI: 10.1109/ITC-Asia71864.2026.11710904. Проверена аннотация, не полный текст.
https://ieeexplore.ieee.org/abstract/document/11710904/

[STAT2016] Properties of Estimators in Exponential Family Settings with Observation-based Stopping Rules.
https://pmc.ncbi.nlm.nih.gov/articles/PMC4861245/

[WHITEHEAD] J. Whitehead. On the bias of maximum likelihood estimation following a sequential test. Biometrika 73(3), 573–581, 1986. DOI: 10.1093/biomet/73.3.573. Проверена аннотация.

Поиск выполнен 23 ноября 2026 года. Запросы, доступность источников, пересечения и ограничения сохранены в research/stopping/literature.json. Отсутствие прямого совпадения в поисковой выдаче не является доказательством мирового приоритета. Полученный научный результат можно защищать в указанной узкой постановке, не объявляя новыми общие статистические принципы или известные атаки.

Код, выводы и материалы подготовлены с помощью ИИ. Фактический вклад каждого участника необходимо обозначить после самостоятельной проверки и подготовки к защите.''')]

# Standalone figures.
p=np.linspace(0,1,501)
fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
ax.plot(p,p,'--',color='gray',label='fixed repeats: E[Q]=p')
for cap in [5,15]:ax.plot(p,stopped_mean(p,cap),label=f'stop at 3 or {cap}')
ax.set(xlabel='True response probability p',ylabel='Expected soft label E[Q]',title='Outcome-dependent stopping distorts calibration')
ax.legend();ax.grid(alpha=.25)
fig.savefig(OUT/'stopping_bias.png',dpi=180);fig.savefig(OUT/'stopping_bias.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
labels=[f"σ={r['sigma']} B={r['budget']} M={r['cap']}" for r in primary]
y=np.arange(len(primary));v=np.array([r['difference'] for r in primary])
err=np.array([r['ci_high']-r['difference'] for r in primary])
ax.errorbar(v,y,xerr=err,fmt='o',capsize=4)
ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(y,labels)
ax.set(xlabel='Probability MSE: count likelihood minus unweighted',title='100 independent PUFs; family-wise 95% intervals')
ax.grid(alpha=.25);fig.savefig(OUT/'primary_contrasts.png',dpi=180);fig.savefig(OUT/'primary_contrasts.pdf');plt.close(fig)

(REPORT/'stopping_sections.json').write_text(json.dumps(sections,ensure_ascii=False,indent=2)+'\n')
(REPORT/'stopping_research.txt').write_text('\n\n'.join(title+'\n'+text for title,text in sections)+'\n')
with PdfPages(REPORT/'stopping_research.pdf') as pdf:
 page=0
 for title,text in sections:
  lines=[]
  for paragraph in text.split('\n'):lines.extend(textwrap.wrap(paragraph,76,break_on_hyphens=False) or [''])
  for offset in range(0,len(lines),35):
   fig=plt.figure(figsize=(8.27,11.69))
   fig.text(.08,.95,'\n'.join(textwrap.wrap(title,48)),fontsize=15,weight='bold',va='top')
   body=fig.text(.08,.85,'\n'.join(lines[offset:offset+35]),fontsize=10.5,va='top',linespacing=1.55)
   fig.canvas.draw();box=body.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
   assert box.y0>.07 and box.x1<.96,(title,box)
   page+=1;fig.text(.91,.04,str(page),fontsize=9)
   pdf.savefig(fig);plt.close(fig)
 for title,path in [('Точное искажение мягких меток',OUT/'stopping_bias.png'),('Подтверждающие сравнения',OUT/'primary_contrasts.png')]:
  fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.94,title,fontsize=16,weight='bold')
  ax=fig.add_axes([.05,.25,.9,.6]);ax.imshow(plt.imread(path));ax.axis('off')
  page+=1;fig.text(.91,.04,str(page),fontsize=9);pdf.savefig(fig);plt.close(fig)
(OUT/'audit.json').write_text(json.dumps(dict(runs=len(rows),expected=expected,instances=len(config['seeds']),
 primary_cells=len(primary),primary_intervals_below_zero=sum(r['ci_high']<0 for r in primary),
 protocol_sha256=hashlib.sha256((Path(__file__).parent/'protocol.json').read_bytes()).hexdigest(),
 all_optimizers_converged=True,source_verified=True),indent=2)+'\n')
print('Report pages:',page)

slides=[
 ('Проблема: число повторов зависит от ответов',[
  'Три одинаковых ответа → остановка; иначе до M=5 или 15.',
  'Обычная доля ответов больше не является несмещённой меткой.',
  'Цель: предсказывать вероятность ответа PUF, а не только его знак.']),
 ('Точный результат',[
  'E[Q|p] = p + (1−3/M)p(1−p)(2p−1).',
  'При p=0.8 и M=15 средняя метка равна 0.8768.',
  'Истинные параметры не стационарны для обычной soft-BCE.',
  'Полный перебор путей и независимая проверка производной.']),
 ('Исправление: сохранять число измерений',[
  'Храним для каждого challenge пару (S, N).',
  'Минимизируем Σ N·BCE(S/N, pθ(c)).',
  'Это правдоподобие наблюдённых остановленных путей.',
  'Известный статистический принцип, применённый к нашей постановке.']),
 ('Подтверждение на новых PUF',[
  '100 экземпляров × 8 режимов × 4 модели = 3200 обучений.',
  'Ошибка вероятностей ниже на 16–83% относительно unweighted.',
  'Все 8 первичных контрастов значимы после Bonferroni.',
  'Single и fixed5 включены в результаты; общего превосходства не заявляем.']),
 ('Граница научного вклада',[
  'Собственные формулы для правила 3-or-M и анализ риска probit.',
  'Воспроизводимый эксперимент калибровки на новых экземплярах.',
  'Общая теория остановки и взвешенные PUF-атаки уже известны.',
  'Точный мировой приоритет не установлен; сравнение источников в отчёте.'])]
with PdfPages(REPORT/'stopping_slides.pdf') as pdf:
 for i,(title,bullets) in enumerate(slides,1):
  fig=plt.figure(figsize=(13.33,7.5),facecolor='#101b2c')
  fig.text(.06,.9,'\n'.join(textwrap.wrap(title,65)),fontsize=23,color='white',va='top',weight='bold')
  y=.72
  for bullet in bullets:
   lines=textwrap.wrap('• '+bullet,85)
   body=fig.text(.065,y,'\n'.join(lines),fontsize=17,color='#e0e8ef',va='top',linespacing=1.4)
   fig.canvas.draw();box=body.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
   assert box.x1<.98 and box.y0>.08
   y-=.115*len(lines)
  fig.text(.94,.04,f'{i}/5',ha='right',color='white')
  if i==4:fig.savefig('/tmp/stopping-slide.png',dpi=100)
  pdf.savefig(fig);plt.close(fig)
