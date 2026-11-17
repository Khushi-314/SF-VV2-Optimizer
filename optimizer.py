"""Simulation-based, time-only decision search. No maintenance scheduling or costs.

All recommendations are conditional on a finite order, a frozen calibrated model,
and explicit illustrative intervention sizes. This is NOT global optimization.
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from config import PRODUCTS, MACHINES, RESOURCES, STAGES, STAGE_RESOURCES
from simulation import ensemble

@dataclass(frozen=True)
class OptimizerConfig:
    screening_runs: int = 4
    refinement_runs: int = 12
    throughput_fraction: float = 0.95
    max_search_wip: int = 96


def _stats(results):
    m=np.asarray([r['makespan_s']/60 for r in results],float)
    th=np.asarray([r['throughput_h'] for r in results],float)
    l=np.asarray([sum(r['lead_sum'].values())/max(1,sum(r['lead_count'].values()))/60 for r in results],float)
    def ci(x):return float(1.96*np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else float('nan')
    return {'makespan_min':float(m.mean()),'makespan_ci_min':ci(m),
            'throughput_h':float(th.mean()),'throughput_ci_h':ci(th),
            'cycle_min':float(l.mean()),'cycle_ci_min':ci(l),
            'mean_wip':float(np.mean([r['wip_mean'] for r in results])),
            'p95_makespan_min':float(np.percentile(m,95))}


def theoretical_wip(parameters, quantities):
    """Idealized Factory Physics estimate for the supplied product proportions."""
    total=sum(int(v) for v in quantities.values())
    if total<=0:raise ValueError('At least one product is required')
    mix={p:int(quantities.get(p,0))/total for p in PRODUCTS}
    stage_times={}
    for stage in STAGES:
        m=STAGE_RESOURCES[stage][0]
        stage_times[stage]=sum(mix[p]*parameters[(p,m)][0] for p in PRODUCTS)
    raw=sum(stage_times.values())
    cap=min(len(STAGE_RESOURCES[s])/stage_times[s] for s in STAGES if stage_times[s]>0)
    return max(1.,float(raw*cap)),min(STAGES,key=lambda s:len(STAGE_RESOURCES[s])/stage_times[s])


def _grid(estimate, upper):
    x={1,2,3,4,5,int(upper)}
    for scale in (0.5,0.75,1,1.25,1.5,2,3,4):
        x.add(int(round(estimate*scale)))
    k=2
    while k<upper:
        x.add(k);k=int(k*1.6)+1
    return sorted(w for w in x if 1<=w<=upper)


def search_wip(parameters,quantities,buffers,reliability,seed=1313,config=None):
    config=config or OptimizerConfig()
    total=sum(int(v) for v in quantities.values())
    theory,bottleneck=theoretical_wip(parameters,quantities)
    upper=min(total,config.max_search_wip,max(24,math.ceil(4*theory),2*len(RESOURCES)))
    grid=_grid(theory,upper)
    cache={}
    def check(w):
        if w not in cache:
            cache[w]=_stats(ensemble(parameters,quantities,w,buffers,reliability,
                   runs=config.screening_runs,seed=seed,detailed_runs=0))
        return cache[w]
    for w in grid:check(w)
    best=max(v['throughput_h'] for v in cache.values())
    threshold=config.throughput_fraction*best
    feasible=[w for w,v in cache.items() if v['throughput_h']>=threshold]
    w=min(feasible)
    # Recheck integer neighbours, filling the gap between the previous coarse point and
    # the current candidate. Never exceeds the stated finite search ceiling.
    lower=max((v for v in grid if v<w),default=1)
    for j in range(max(1,lower),w+1):check(j)
    best=max(v['throughput_h'] for v in cache.values())
    candidates=[i for i,v in cache.items() if v['throughput_h']>=config.throughput_fraction*best]
    chosen=min(candidates)
    # Focused replication on candidates near the threshold, plus plateau reference.
    shortlist=sorted(set([chosen,max(cache),*range(max(1,chosen-2),min(upper,chosen+2)+1)]))
    for j in shortlist:
        cache[j]=_stats(ensemble(parameters,quantities,j,buffers,reliability,
                    runs=config.refinement_runs,seed=seed,detailed_runs=0))
    # Compare only similarly refined points when choosing an operating WIP.
    high=max(cache[j]['throughput_h'] for j in shortlist)
    good=[j for j in shortlist if cache[j]['throughput_h']>=config.throughput_fraction*high]
    chosen=min(good)
    table=pd.DataFrame([{'WIP':k,**v,'Runs':(config.refinement_runs if k in shortlist else config.screening_runs)}
                        for k,v in sorted(cache.items())])
    reached=upper>1 and (cache[upper]['throughput_h']-cache[chosen]['throughput_h'])<=0.05*max(cache[upper]['throughput_h'],1e-9)
    return {'recommended_wip':chosen,'theoretical_wip':theory,'theoretical_bottleneck':bottleneck,
            'wip_table':table,'tested_upper':upper,'saturation_supported':reached,
            'note':'Finite-order throughput plateau is an empirical approximation, not a global optimum.'}


def _variants(buffers,reliability):
    for m in MACHINES:
        for fraction in (0.25,0.50):
            c={k:v.copy() for k,v in reliability.items()}
            c[m]['mttf_s']*=1+fraction
            c[m]['eta_s']*=1+fraction  # beta held fixed, so MTTF scales with eta
            yield f'{m}: MTTF +{fraction:.0%}',f'Reliability · {m}',c,buffers.copy()
            c={k:v.copy() for k,v in reliability.items()}
            c[m]['repair_s']*=1-fraction
            yield f'{m}: repair mean -{fraction:.0%}',f'Repair · {m}',c,buffers.copy()
        for increment in (1,2):
            b=buffers.copy();b[m]=b.get(m,0)+increment
            yield f'{m}: buffer +{increment}',f'Buffer · {m}',reliability,b


def rank_interventions(parameters,quantities,wip,buffers,reliability,seed=1313,
                       screening_runs=4,refinement_runs=12,top_k=10):
    baseline_screen=ensemble(parameters,quantities,wip,buffers,reliability,
                             runs=screening_runs,seed=seed,detailed_runs=0)
    screen_mean=np.mean([r['makespan_s'] for r in baseline_screen])/60
    ranked=[]
    for label,group,conf,buf in _variants(buffers,reliability):
        trials=ensemble(parameters,quantities,wip,buf,conf,runs=screening_runs,
                        seed=seed,detailed_runs=0)
        gain=screen_mean-np.mean([r['makespan_s'] for r in trials])/60
        ranked.append((gain,label,group,conf,buf))
    ranked.sort(key=lambda x:x[0],reverse=True)
    # Publish refined estimates only: don't present tiny-screening-run ranks as facts.
    baseline=ensemble(parameters,quantities,wip,buffers,reliability,
                      runs=refinement_runs,seed=seed,detailed_runs=0)
    base=np.asarray([r['makespan_s']/60 for r in baseline])
    rows=[]
    chosen=ranked[:max(0,top_k-3)]
    for category in ('Reliability','Repair','Buffer'):
        option=next((r for r in ranked if r[2].startswith(category)),None)
        if option is not None and option[1] not in {r[1] for r in chosen}:chosen.append(option)
    for item in ranked:
        if len(chosen)>=top_k:break
        if item[1] not in {r[1] for r in chosen}:chosen.append(item)
    for _,label,group,conf,buf in chosen:
        trials=ensemble(parameters,quantities,wip,buf,conf,runs=refinement_runs,
                        seed=seed,detailed_runs=0)
        diff=base-np.asarray([r['makespan_s']/60 for r in trials])
        ci=1.96*np.std(diff,ddof=1)/np.sqrt(len(diff)) if len(diff)>1 else float('nan')
        rows.append({'Intervention':label,'Category':group,
                     'Mean time saved (min)':float(diff.mean()),
                     '95% CI ± (min)':float(ci),
                     'Improvement (%)':float(100*diff.mean()/base.mean()),
                     'P95 reduction (min)':float(np.percentile(base,95)-np.percentile([r['makespan_s']/60 for r in trials],95)),
                     'Evidence':('Positive CI' if diff.mean()-ci>0 else 'Uncertain')})
    return pd.DataFrame(rows).sort_values('Mean time saved (min)',ascending=False).reset_index(drop=True)


def optimize(parameters,quantities,buffers,reliability,seed=1313,screening_runs=4,refinement_runs=12):
    conf=OptimizerConfig(screening_runs=screening_runs,refinement_runs=refinement_runs)
    w=search_wip(parameters,quantities,buffers,reliability,seed,conf)
    w['recommendations']=rank_interventions(parameters,quantities,w['recommended_wip'],buffers,reliability,
                       seed,screening_runs,refinement_runs)
    return w
