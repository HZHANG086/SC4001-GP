"""Paired test and crossed seed/example bootstrap without pseudo-replication.

Items shared across seeds are sampled ONCE per replicate, not independently for
each model/seed. Domain and label proportions stay fixed. Grouping identical
five-seed correctness-difference patterns is exactly equivalent to multinomial
resampling individual examples, while avoiding huge index arrays.
"""
import numpy as np


def crossed_bootstrap(domain_differences,domain_labels,repetitions=10000,seed=20261110):
    rng=np.random.default_rng(seed)
    nseeds=domain_differences[0].shape[0]
    totals=np.zeros((repetitions,nseeds),dtype=float)
    observed=0.0
    for matrix,labels in zip(domain_differences,domain_labels):
        matrix=np.asarray(matrix,dtype=float);labels=np.asarray(labels)
        if matrix.shape!=(nseeds,len(labels)) or not np.isin(matrix,[-1,0,1]).all():
            raise ValueError('Expected seed-by-example correctness differences')
        domain=np.zeros_like(totals)
        for label in (0,1):
            subset=matrix[:,labels==label].T
            if len(subset)==0: raise ValueError('Both labels required')
            patterns,counts=np.unique(subset,axis=0,return_counts=True)
            draws=rng.multinomial(len(subset),counts/len(subset),size=repetitions)
            domain += draws@patterns/len(labels)
        totals += domain/len(domain_differences)
        observed += matrix.mean()/len(domain_differences)
    seed_counts=rng.multinomial(nseeds,np.full(nseeds,1/nseeds),size=repetitions)
    conditional=totals.mean(1)*100
    crossed=(totals*seed_counts).sum(1)/nseeds*100
    result={'mean_difference_pp':float(observed*100),'replicates':repetitions,'rng_seed':seed}
    for name,values in [('conditional_fixed_models',conditional),('crossed_seeds_and_examples',crossed)]:
        result[name+'_low'],result[name+'_high']=map(float,np.quantile(values,[.025,.975]))
    return result
