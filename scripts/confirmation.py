"""Independent holdout-instance confirmation of an exploratory contrast."""
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import t
os.environ.setdefault('MPLCONFIGDIR','/tmp/puf-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/confirmation'
rows=[json.loads(x) for x in (out/'runs.jsonl').read_text().splitlines()]
assert len(rows)==300 and len({r['id'] for r in rows})==300
by={s:{r['seed']:r for r in rows if r['strategy']==s} for s in ['single','majority_5','soft_5']}
assert all(set(v)==set(range(1001,1101)) for v in by.values())
contrasts=[]
for a,b in [('soft_5','majority_5'),('soft_5','single')]:
 d=np.array([by[a][seed]['accuracy']-by[b][seed]['accuracy'] for seed in sorted(by[a])])
 mean=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(len(d)))
 # Bonferroni family-wise 95% intervals for two contrasts.
 half=float(t.ppf(1-.05/(2*2),len(d)-1)*se)
 contrasts.append(dict(comparison=f'{a} - {b}',mean=mean,low=mean-half,high=mean+half,n=len(d)))
(out/'confirmation.json').write_text(json.dumps(contrasts,indent=2)+'\n')
lines=['Независимая проверка бюджетного сравнения',
       '100 новых экземпляров, seed 1001–1100; n=64, k=1, σ=0.75.',
       'Бюджет B=4096, включая 256 validation; test=10000.',
       'Конфигурация и два контраста зафиксированы до нового запуска.',
       'Алгоритмы и гиперпараметры сохранены после исходного эксперимента.',
       'Использованы новые экземпляры, challenge и случайные измерения.', '',
       'Парные интервалы с поправкой Bonferroni на два сравнения:',
       '(семейный уровень доверия 95%; t-интервалы по экземплярам)']
for r in contrasts:
 lines.extend([r['comparison'],f"  {100*r['mean']:.3f} п.п.; интервал [{100*r['low']:.3f}; {100*r['high']:.3f}] п.п."])
lines.extend(['','Средняя accuracy:'])
for s,v in by.items():lines.append(f"{s}: {100*np.mean([r['accuracy'] for r in v.values()]):.3f}%")
lines.extend(['','Подтверждается только выбранная постановка, не все уровни шума.',
              'Независимая проверка повышает надёжность эмпирического вывода,',
              'но не доказывает научный приоритет метода и не гарантирует оценку.',
              'Мягкие ответы и использование повторов известны в литературе.',
              'Связанный первичный источник: ISLPED 2016 (измерения PUF):',
              'https://people.ece.umn.edu/groups/VLSIresearch/papers/2016/ISLPED16_PUF.pdf',
              'Внешняя аппаратная валидация нашего кода не выполнена.'])
text='\n'.join(lines)+'\n'
(ROOT/'report/confirmation.txt').write_text(text)
fig=plt.figure(figsize=(11.69,8.27))
fig.text(.06,.94,lines[0],fontsize=17,weight='bold',va='top')
fig.text(.06,.87,'\n'.join(lines[1:]),fontsize=10,linespacing=1.6,va='top')
fig.savefig(ROOT/'report/confirmation.pdf')
plt.close(fig)
print(text)
