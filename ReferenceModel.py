from pyomo.pysp.scenariotree.manager import ScenarioTreeManagerClientSerial
from pyomo.pysp.ef import create_ef_instance
from ReferenceSCUC_dat import *

# del model.TotalProductionCost
# del model.TotalFixedCost
del model.TotalCostObjective

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

model.StageCost = Var(model.TimePeriods, within=NonNegativeReals)
model.StageCostConstraint = Constraint(model.TimePeriods, rule = StageCost_rule)

model.TotalCostObjective = Objective(
    rule=Objective_rule,
    sense=minimize,
)

if __name__ == "__main__":
    options = ScenarioTreeManagerClientSerial.register_options()
    p_model = './ReferenceModel.py'
    p_data = './SP'
    options = ScenarioTreeManagerClientSerial.register_options()
    options.model_location = p_model
    options.scenario_tree_location = p_data
    with ScenarioTreeManagerClientSerial(options) as manager:
        manager.initialize()
        ef_instance = create_ef_instance(manager.scenario_tree,
                                         verbose_output=options.verbose)
        with SolverFactory('cplex') as opt:
            ef_result = opt.solve(ef_instance)
    ef_instance.solutions.store_to( ef_result )
    ef_obj = value( ef_instance.EF_EXPECTED_COST.values()[0] ) 
