from pyomo.pysp.scenariotree.manager import ScenarioTreeManagerClientSerial
from pyomo.pysp.ef import create_ef_instance
from ReferenceSCUC_dat import *
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt

def extract_var_2dim(instance, varname):
    # This function is used to extract variables indexed by 2-dim tuple with 
    # the following format: (generator, time): value.
    tmp_col = set()
    tmp_var = getattr(instance, varname)
    for k, v in tmp_var.iteritems():
        gen, t = k
        tmp_col.add(gen)
    var = pd.DataFrame(
        index=instance.TimePeriods.values(),
        columns=list(tmp_col),
    )
    for k, v in tmp_var.iteritems():
        gen, t = k
        var.at[t, gen] = value(v)
    return var

def StageCost_rule(model, t):
    expr = sum(
        model.ProductionCost[g, t]
        + model.StartupCost[g, t]
        + model.ShutdownCost[g, t]
        for g in model.ThermalGenerators
    )
    return model.StageCost[t] == expr

def Objective_rule(model):
    expr = sum(
        model.StageCost[t] for t in model.TimePeriods
    )
    return expr

# del model.TotalProductionCost
# del model.TotalFixedCost
del model.TotalCostObjective

model.StageCost = Var(model.TimePeriods, within=NonNegativeReals)
model.StageCostConstraint = Constraint(model.TimePeriods, rule = StageCost_rule)

model.TotalCostObjective = Objective(
    rule=Objective_rule,
    sense=minimize,
)

if __name__ == "__main__":
    p_model = './ReferenceModel.py'
    p_data = './SP'
    options = ScenarioTreeManagerClientSerial.register_options()
    options.model_location = p_model
    options.scenario_tree_location = p_data
    results = defaultdict(list)
    varnames = [
        'PowerGenerated',
        'FlexibleRampUpAvailable',
        'FlexibleRampDnAvailable',
        'RegulatingReserveUpAvailable',
        'RegulatingReserveDnAvailable',
        'SpinningReserveUpAvailable',
        'UnitOn',
        'MaximumPowerAvailable',
    ]
    with ScenarioTreeManagerClientSerial(options) as manager:
        manager.initialize()
        ef_instance = create_ef_instance(manager.scenario_tree,
                                         verbose_output=options.verbose)
        with SolverFactory('cplex') as opt:
            ef_result = opt.solve(ef_instance)
        for s in manager.scenario_tree.scenarios:
            ins = s._instance
            for varname in varnames:
                results[varname].append( extract_var_2dim(ins, varname) )

    plt.plot(
        results['PowerGenerated'][1].index, 
        results['PowerGenerated'][1]['Wind 01'],
        '-k*',
        results['PowerGenerated'][0].index,
        results['PowerGenerated'][0]['Wind 01'],
        '-b*',
    )
    plt.show()