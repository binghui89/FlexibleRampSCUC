from pyomo.pysp.scenariotree.manager import ScenarioTreeManagerClientSerial
from pyomo.pysp.ef import create_ef_instance
from SCUC_RampConstraint_3 import *
import os, pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from IPython import embed as IP

def import_scenario_data():
    data_path = os.path.sep.join(
        ['.', 'TEXAS2k_B']
    )
    gen_df = pd.read_csv(
        os.path.sep.join( [data_path, 'generator_data_plexos_withRT.csv'] ),
        index_col=0,
    )
    wind_generator_names  =  [ x for x in gen_df.index if x.startswith('wind') ]
    genfor_df = pd.read_csv(
        os.path.sep.join( [data_path, 'generator.csv'] ),
        index_col=0,
    )

    dir_windforecast  = os.path.sep.join(
        [data_path, 'result']
    )
    df_wind_mapping   = pd.DataFrame.from_csv(
        os.path.sep.join(
            [data_path, 'wind_generator_data.csv']
        )
    )
    map_wind2site = df_wind_mapping['SITE_ID'].to_dict()

    col_names = [
        'Q10',
        'Q20',
        'Q30',
        'Q40',
        'Q50',
        'Q60',
        'Q70',
        'Q80',
        'Q90',
    ]
    WindPowerForecast = dict()
    for w in wind_generator_names:
        wid = map_wind2site[w]
        fname = os.path.sep.join(
            [
                dir_windforecast,
                'Forecast_prob_'+str(wid)+'.csv',
            ]
        )
        df_tmp = pd.read_csv(fname)
        row_chosen = (df_tmp['Year']==2012) & (df_tmp['Month']==5) & (df_tmp['Day']==10)

        # # The following code is to assure Kwami's data is the same day as my data
        # actual = df_tmp.loc[row_chosen, 'Actual'].reset_index(drop=True)
        # for i in range(0, 24):
        #     if abs(genfor_df.loc[i, w] - 0)>=1E-3:
        #         break # Assure the denominator, genfor_df.loc[i, w], > 0
        # scaler = actual[i]/genfor_df.loc[i, w]
        # delta  = actual.subtract(scaler*genfor_df.loc[:, w])
        # print scaler, sum(delta) 

        for c in col_names:
            s = c + 'Scenario' # Scenario name
            if s not in WindPowerForecast:
                WindPowerForecast[s] = dict()
            for h in range(1, 25): # They start from 1 to 24
                index_chosen = (
                    (df_tmp['Year']==2012)
                    & (df_tmp['Month']==5)
                    & (df_tmp['Day']==10)
                    & (df_tmp['Hour']==h-1) # We start from 0 to 23
                )
                tmp_data = df_tmp.loc[index_chosen, c].values[0] # Hopefully there is only one element
                if tmp_data < 0:
                    tmp_data = 0 # Floor at zero
                WindPowerForecast[s][w, h] = tmp_data
    print "Scenario data created!"
    return WindPowerForecast

# PowerForecast = dict()
# scenarios = ['WindOffScenario', 'WindOnScenario']
# for s in scenarios:
#     if s == 'WindOnScenario':
#         scenario_data = dict()
#         for w in wind_generator_names:
#             for t in range(1, 25):
#                 scenario_data[w, t] = 1
#         PowerForecast[s] = scenario_data
#     elif s == 'WindOffScenario':
#         scenario_data = dict()
#         for w in wind_generator_names:
#             for t in range(1, 25):
#                 scenario_data[w, t] = 0
#         PowerForecast[s] = scenario_data
PowerForecast = import_scenario_data()

def DummyStageCost_rule(model):
    return model.DummyStageCost == 0

def StageCost_rule(model, t):
    StageProductionCost = sum(
        model.ProductionCost[g, t]
        for g in model.ThermalGenerators
    )
    StageFixedCost = sum(
        model.StartupCost[g, t] + model.ShutdownCost[g, t]
        for g in model.ThermalGenerators
    )
    StageCurtailmentCost = sum(
        model.BusVOLL[b] * model.BusCurtailment[b,t]
        for b in model.LoadBuses
    )
    expr = (
        StageProductionCost
        + StageFixedCost
        + StageCurtailmentCost
        + model.RampingCost[t]
        + 10000000*(
            sum( model.MaxWindAvailable[g,t] for g in model.WindGenerators )
            + model.SpinningReserveShortage[t]
            + model.RegulatingReserveShortage[t]
            + model.FlexibleRampDnShortage[t]
            + model.FlexibleRampUpShortage[t]
        )
    )
    return model.StageCost[t] == expr

def Objective_rule(model):
    expr = model.DummyStageCost + sum(
        model.StageCost[t] for t in model.TimePeriods
    )
    return expr

def pysp_instance_creation_callback(scenario_name, node_names):

    instance = model.clone()
    instance.PowerForecast.store_values(PowerForecast[scenario_name])

    return instance

del model.TotalCostObjective

# Because we have to define a first stage cost for runef/runph, here defines a 
# dummy first stage cost, which always equals to 0
model.DummyStageCost = Var( within=NonNegativeReals)
model.DummyStageCostConstraint = Constraint( rule = DummyStageCost_rule )
model.StageCost = Var(model.TimePeriods, within=NonNegativeReals)
model.StageCostConstraint = Constraint(model.TimePeriods, rule = StageCost_rule)

model.TotalCostObjective = Objective(
    rule=Objective_rule,
    sense=minimize,
)