import MultiNEAT as NEAT
import networkx as nx
from genes import *
from sklearn.model_selection import StratifiedKFold

########################
# NEAT parameters
# ...
use_weights = True
num_trials = 3
CV_splits = 5


params = NEAT.Parameters()

params.PopulationSize = 120
params.DynamicCompatibility = True
params.YoungAgeTreshold = 3
params.SpeciesMaxStagnation = 10000
params.OldAgeTreshold = 1000
params.MinSpecies = 2
params.MaxSpecies = 6
params.RouletteWheelSelection = False
params.ArchiveEnforcement = True
params.InnovationsForever = True

params.ConstraintTrials = 12800000

params.MutateAddNeuronProb = 0.01
params.MutateAddLinkProb = 0.03
#params.MutateRemLinkProb = 0.1/4
params.RecurrentProb = 0.0
params.MaxWeight = 3.0
params.MinWeight = -3.0

params.MutateWeightsProb = 0.25
params.MutateActivationAProb = 0.0
params.MutateActivationBProb = 0.0
params.MutateNeuronTimeConstantsProb = 0.0
params.MutateNeuronBiasesProb = 0.0

params.MutateGenomeTraitsProb = 0.0
params.MutateNeuronTraitsProb = 0.5
params.MutateLinkTraitsProb = 0.0

params.OverallMutationRate = 0.7
params.CrossoverRate = 0.7
params.MultipointCrossoverRate = 0.4
params.SurvivalRate = 0.2
params.InterspeciesCrossoverRate = 0.005
params.PreferFitterParentRate = 0.5

params.DontUseBiasNeuron = True
params.AllowLoops = False
params.AllowClones = True

params.ExcessCoeff = 1.0
params.DisjointCoeff = 1.0

params.WeightDiffCoeff = 0.1
params.TimeConstantDiffCoeff = 0.0
params.BiasDiffCoeff = 0.0
params.ActivationADiffCoeff = 0.0
params.ActivationBDiffCoeff = 0.0
params.NormalizeGenomeSize = True

params.MinCompatTreshold = 0.0
params.CompatTreshold = 1.25
params.CompatTreshChangeInterval_Evaluations = 1
params.CompatTresholdModifier = 0.2

probs = [1.0]*len(derived_list)

##################
# Genome constraints
def fails_constraints(genome):
    # """
    try:
        ts = genome.GetNeuronTraits()
        o = None
        for t in ts:
            if t[1] == 'output':
                o = t[2]
            else:
                if t[1] == 'hidden':
                    if (isinstance(t[2]['node'], geneLogisticRegression) #or
                           # isinstance(t[2]['node'], geneKNeighborsClassifier) or
                           # isinstance(t[2]['node'], geneXGBClassifier)
                    ):
                        return True

        if not (isinstance(o['node'], geneLogisticRegression) #or
               # isinstance(o['node'], geneSVC) or
               # isinstance(o['node'], geneDecisionTreeClassifier) or
               # isinstance(o['node'], geneKNeighborsClassifier) or
               # isinstance(o['node'], geneBaggingClassifier) or
               # isinstance(o['node'], geneGradientBoostingClassifier) or
               # isinstance(o['node'], geneExtraTreesClassifier) or
               # isinstance(o['node'], geneLightGBM) or
               # isinstance(o['node'], geneCatBoostClassifier) or
               # isinstance(o['node'], geneXGBClassifier)
        ):
            return True
        else:
            return False
    except:
        return True
    # """
    return False


params.CustomConstraints = fails_constraints

###############
# Traits setup
###############
# merge mode
s = ['concat',
    # when dimensions match, these can be done, otherwise it defaults to concat
     'add','mul','avg','min','max',
    ]
p = [1.0] * len(s)
mm = {'details': {'set': s, 'probs': p},
      'importance_coeff': 0.0,
      'mutation_prob': 0.3,
      'type': 'str'}

params.SetNeuronTraitParameters('mm', mm)

node = {'details': (derived_list, probs),
          'importance_coeff': 0.1,
          'mutation_prob': 0.2,
          'type': 'pyclassset'}

params.SetNeuronTraitParameters('node', node)

# Initialize population
num_inputs = 1
num_outputs = 1

num_input_dims = dx.shape[1]
num_output_dims = 1

#############################
# Activate Graph
#############################

def activate_graph(gr, inputs, targets, num_outputs=1, fit=True):
    allnodes = list(nx.dfs_postorder_nodes(gr))[::-1]
    for a in allnodes: gr.node[a]['act'] = None

    # separate input from non-input nodes
    allnodes = [x for x in allnodes if x > num_inputs]

    # input the data
    for i, inp in zip(range(1, num_inputs + 1), inputs):
        gr.node[i]['act'] = np.array(inp).reshape(inp.shape[0], -1)

    # pass through the graph
    for an in allnodes:
        # print(gr.node[a], end=' ')
        mm = gr.node[an]['mm']

        # collect the inputs to this node

        # also sort the incoming edges by id for consistency
        inedg = list(gr.in_edges(an))
        inps = [gr.node[i]['act'] for i, o in inedg]

        if use_weights:
            inedgw = list(gr.in_edges(an, data=1))
            ws = [ts['w'] for i, o, ts in inedgw]
            # weighted stack
            inps = [w * x for w, x in zip(ws, inps)]
        else:
           # not weighted stack
           inps = np.vstack(inps)

        if (mm == 'concat') or (len(inps) == 1) or (not all([x.shape[1] == inps[0].shape[1] for x in inps])):
            if len(inps) > 1:
                iii = np.concatenate(inps, axis=1)
            else:
                if isinstance(inps, list):
                    iii = inps[0]
                else:
                    iii = inps
        else:
            iii = np.array(inps)
            if mm == 'add':
                iii = np.sum(inps, axis=0)
            elif mm == 'mul':
                iii = np.prod(inps, axis=0)
            elif mm == 'avg':
                iii = np.mean(inps, axis=0)
            elif mm == 'min':
                iii = np.min(inps, axis=0)
            elif mm == 'max':
                iii = np.max(inps, axis=0)
            else:
                iii = np.array(iii)

        if fit:
            gr.node[an]['node'].fit(iii, targets)
        act = gr.node[an]['node'].transform(iii, targets).reshape(iii.shape[0], -1)

        # store activation
        gr.node[an]['act'] = act

    outputs = [gr.node[o]['act'] for o in allnodes[-num_outputs:]]
    outputs = np.array(outputs[0])
    return np.array(outputs)



########################
# Evaluate Individual
########################

# evaluation function
pred_mode = 'c'


def fitter(gr, a, b, fit):
    return activate_graph(gr, [a], b, fit=fit)


def evaluate(args):
    idx, gr, dx, dy, ltr, ntr, precomp = args
    ntr = [x for x in ntr if x[1] != 'input']

    alls = [w for i, o, tr, w in ltr]
    ws = alls[0:len(ltr)]
    for (i, o, tr, w), nw in zip(ltr, ws):
        gr.edge[i][o]['w'] = nw

    if pred_mode == 'r':
        pdy = np.digitize(dy.reshape(-1), bins=np.linspace(np.min(dy), np.max(dy), 10))
    else:
        pdy = dy.reshape(-1)

    try:
        acc = 0

        for trial in range(num_trials):

            skf = StratifiedKFold(n_splits=CV_splits, shuffle=True,
                                  # random_state = rnd.randint(0, 100000)
                                  )
            cvavg_tr = 0
            cvavg_ts = 0
            for tr, ts in (skf.split(dx, pdy)):
                x_train = dx[tr]
                y_train = dy[tr]
                x_test = dx[ts]
                y_test = dy[ts]

                a_tr = fitter(gr, x_train, y_train.reshape(-1),
                              True)  # activate_graph(gr, [x_train], y_train, fit=True)
                a_ts = fitter(gr, x_test, y_test.reshape(-1), False)  # activate_graph(gr, [x_test], y_test, fit=False)

                # print(a_tr, a_ts)

                def fixx(x):
                    x = x.reshape(x.shape[0], -1)
                    if x.shape[1] > 1:
                        x = np.mean(x, axis=1)
                    # x = np.round(x)
                    return x.reshape(x.shape[0], -1)

                a = fixx(a_tr)
                b = fixx(a_ts)

                # print(a.reshape(-1), len(a.reshape(-1)), np.sum(a.reshape(-1) == y_train.reshape(-1)),
                #      np.sum(a.reshape(-1) == y_train.reshape(-1)) / len(a.reshape(-1)))
                # print(b.reshape(-1), np.sum(b.reshape(-1) == y_test.reshape(-1)), len(b.reshape(-1)),
                #      np.sum(b.reshape(-1) == y_test.reshape(-1)) / len(b.reshape(-1)))
                # print()

                acc_tr = np.sum(a.reshape(-1) == y_train.reshape(-1)) / len(a.reshape(-1))
                acc_ts = np.sum(b.reshape(-1) == y_test.reshape(-1)) / len(b.reshape(-1))
                cvavg_tr += acc_tr
                cvavg_ts += acc_ts

            acc += float(cvavg_ts / CV_splits)

        acc /= num_trials

    except Exception as ex:
        print(ex)
        acc = 0.0

    f = acc

    return idx, f, None


