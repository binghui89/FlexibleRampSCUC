import os, pandas as pd
from pyomo.pysp.scenariotree.manager import ScenarioTreeManagerClientSerial
from pyomo.pysp.ef import create_ef_instance
from pyomo.environ import *
from collections import defaultdict
from matplotlib import pyplot as plt
from SCUC_RampConstraint_3 import create_model
from helper import import_scenario_data
from IPython import embed as IP

model = create_model()
sname, sdata = import_scenario_data()
PowerForecast = sdata

def FirstStageCost_rule(model):
    return model.TotalFixedCost

def SecondStageCost_rule(model):
    return model.TotalProductionCost + model.TotalCurtailmentCost + model.TotalReserveShortageCost

def pysp_instance_creation_callback(scenario_name, node_names):

    instance = model.clone()
    instance.PowerForecast.store_values(PowerForecast[scenario_name])

    return instance

del model.TotalCostObjective

# Because we have to define a first stage cost for runef/runph, here defines a 
# dummy first stage cost, which always equals to 0
# model.FirstStageCost = Var( within=NonNegativeReals)
# model.FirstStageCostConstraint = Constraint( rule = FirstStageCost_rule )
# model.SecondStageCost = Var(model.TimePeriods, within=NonNegativeReals)
# model.SecondStageCostConstraint = Constraint(model.TimePeriods, rule = SecondStageCost_rule)
model.FirstStageCost  = Expression(rule = FirstStageCost_rule)
model.SecondStageCost = Expression(rule = SecondStageCost_rule)

def Objective_rule(model):
    return model.FirstStageCost + model.SecondStageCost

model.TotalCostObjective = Objective(
    rule=Objective_rule,
    sense=minimize,
)