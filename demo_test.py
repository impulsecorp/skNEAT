from algo import *
from matplotlib.pylab import plt

num_trials = 10
CV_splits = 10

use_weights = 1

parallel = 0
precomp = None
verbose=1
use_local_search=0
display_whole_pop = 0
display_max_species = 8
initeval = 1
evaluations = 1000
display_pop_each = 80000
once = False
x_shape, ys = dx.shape[1], 1
population = params.PopulationSize

penalize_stangation = 1
penalize_stagnation_evals = 2400

max_stagnation = 5000

max_nodeslinks_to_output = 4

evhist = []
best_ever = 0
best_gs = []

def poplen(pop):
    return sum([len(x.Individuals) for x in pop.Species])

def prettydict(d, nonl=False):
    i,t,di = d
    if not nonl:
        ks = '%d - %s: ' % (i, t)
    else:
        return '%d - %s' % (i, t)
    s = []
    for k,v in sorted(list(di.items())):
        if isinstance(v, float):
            s.append( '%s %3.3f' % (k[0:1], v) )
        else:
            s.append( '%s %s' % (k[0:1], v) )
    return ks + ', '.join(s)

def decide(x):
    return x

def species_display(pop):
    genomes = [x.GetLeader() for x in pop.Species][0:display_max_species]
    f, axes = plt.subplots(1, len(genomes), figsize=(len(genomes) * 4.5, 14))
    print('Species Representatives:')
    if len(genomes)>1:
        for i,(ax, g) in enumerate(zip(axes, genomes)):
            try:
                img = NEAT.viz.Draw(g, size=(300, 400))[:, 0:230]
                ax.imshow(img)
                ax.set_title('%3.6f | %3.2f%%' % (decide(g.GetFitness()),
                                                  (len(pop.Species[i].Individuals)/poplen(pop))*100 ))
                s = '\n'.join([prettydict(x) for x in g.GetNeuronTraits() if x[1] != 'input'][0:8])
                s += '\n=============\n'
                s += '\n'.join([prettydict(x, nonl=True) for x in g.GetLinkTraits(False)][0:8])
                ax.set_xlabel(s)
            except Exception as ex:
                print(ex)
    else:
        try:
            g = pop.GetBestGenome()
            img = NEAT.viz.Draw(g, size=(300, 400))[:, 0:230]
            axes.imshow(img)
            axes.set_title('%3.6f' % (decide(g.GetFitness())))
            s = '\n'.join([prettydict(x) for x in g.GetNeuronTraits() if x[1] != 'input'][0:8])
            s += '\n=============\n'
            s += '\n'.join([prettydict(x, nonl=True) for x in g.GetLinkTraits(False)][0:8])
            axes.set_xlabel(s)
        except Exception as ex:
            print(ex)
    f.tight_layout()
    plt.show()













