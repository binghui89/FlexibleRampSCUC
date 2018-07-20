from pyomo.pysp.scenariotree.manager import ScenarioTreeManagerClientSerial
from pyomo.pysp.ef import create_ef_instance
from ReferenceSCUC import *
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

def scenario_data_118():
    PowerForecastWind_w = import_scenario_data()
    PowerForecastWind_W = dict()
    W2w = {
        'Wind 01': 'wind1',
        'Wind 02': 'wind2',
        'Wind 03': 'wind23',
        'Wind 04': 'wind4',
        'Wind 05': 'wind5',
        'Wind 06': 'wind6',
        'Wind 07': 'wind7',
        'Wind 08': 'wind8',
        'Wind 09': 'wind9',
        'Wind 10': 'wind10',
        'Wind 11': 'wind11',
        'Wind 12': 'wind52',
        'Wind 13': 'wind43',
        'Wind 14': 'wind24',
        'Wind 15': 'wind35',
        'Wind 16': 'wind36',
        'Wind 17': 'wind37',
    }
    for W in wind_generator_names:
        w = W2w[W]
        for s in PowerForecastWind_w:
            if s not in PowerForecastWind_W:
                PowerForecastWind_W[s] = dict()
            for h in range(1, 25):
                PowerForecastWind_W[s][W, h] = PowerForecastWind_w[s][w, h]
    return PowerForecastWind_W

PowerForecastWind = scenario_data_118()

def StageCost_rule(model, t):
    expr = sum(
        model.ProductionCost[g, t]
        + model.StartupCost[g, t]
        + model.ShutdownCost[g, t]
        for g in model.ThermalGenerators
    )
    return model.StageCost[t] == expr

def Objective_rule(model):
    expr = model.DummyStageCost + sum(
        model.StageCost[t] for t in model.TimePeriods
    )
    return expr

def DummyStageCost_rule(model):
    return model.DummyStageCost == 0

def pysp_instance_creation_callback(scenario_name, node_names):

    instance = model.clone()
    instance.PowerForecast.store_values(PowerForecastWind[scenario_name])

    return instance


# del model.TotalProductionCost
# del model.TotalFixedCost
del model.TotalCostObjective

model.DummyStageCost = Var( within=NonNegativeReals)
model.DummyStageCostConstraint = Constraint( rule = DummyStageCost_rule )
model.StageCost = Var(model.TimePeriods, within=NonNegativeReals)
model.StageCostConstraint = Constraint(model.TimePeriods, rule = StageCost_rule)

model.TotalCostObjective = Objective(
    rule=Objective_rule,
    sense=minimize,
)

# if __name__ == "__main__":
#     p_model = './ReferenceModel.py'
#     p_data = './SP'
#     options = ScenarioTreeManagerClientSerial.register_options()
#     options.model_location = p_model
#     options.scenario_tree_location = p_data
#     results = defaultdict(list)
#     varnames = [
#         'PowerGenerated',
#         'FlexibleRampUpAvailable',
#         'FlexibleRampDnAvailable',
#         'RegulatingReserveUpAvailable',
#         'RegulatingReserveDnAvailable',
#         'SpinningReserveUpAvailable',
#         'UnitOn',
#         'MaximumPowerAvailable',
#     ]
#     with ScenarioTreeManagerClientSerial(options) as manager:
#         manager.initialize()
#         ef_instance = create_ef_instance(manager.scenario_tree,
#                                          verbose_output=options.verbose)
#         with SolverFactory('cplex') as opt:
#             ef_result = opt.solve(ef_instance)
#         for s in manager.scenario_tree.scenarios:
#             ins = s._instance
#             for varname in varnames:
#                 results[varname].append( extract_var_2dim(ins, varname) )

#     plt.plot(
#         results['PowerGenerated'][1].index, 
#         results['PowerGenerated'][1]['Wind 01'],
#         '-k*',
#         results['PowerGenerated'][0].index,
#         results['PowerGenerated'][0]['Wind 01'],
#         '-b*',
#     )
#     plt.show()