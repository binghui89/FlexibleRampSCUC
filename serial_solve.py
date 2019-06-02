from __future__ import division
import os, sys, platform, datetime, smtplib, multiprocessing, pandas as pd, numpy as np, matplotlib
from time import time
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from postprocessing import store_csvs
from pyomo.environ import *
from helper import import_scenario_data, extract_uniton
from email.mime.text import MIMEText
from numpy import sign
from matplotlib import pyplot as plt
from IPython import embed as IP
from SCUC_RampConstraint_3 import create_model, create_dispatch_model

def create_UC_model_season(season):
    ''' 
    This function creat the base UC model and modify it with season data
    '''
    data_path = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/'
    instance = create_model()
    if season=='season2':
        return instance
    else:
        csv_load  = data_path + 'loads_' + season + '.csv'
        csv_solar = data_path + 'solar_' + season + '.csv'
        dir_scenario='/home/bxl180002/git/WindScenarioGeneration/'+ season
        sname, sdata = import_scenario_data(dir_scenario=dir_scenario)

    df_load = pd.read_csv(csv_load, index_col=0)
    df_load.reset_index(drop=True, inplace=True)
    df_busload = df_load.loc[:, df_load.columns.difference(['LOAD'])]
    dict_loadtotal = dict()
    dict_busload = dict()
    for i, row in df_busload.iterrows():
        for b in df_busload.columns:
            dict_busload[b, i+1] = df_busload.loc[i, b]
    df_load_total = df_load.loc[:, 'LOAD'].reset_index()
    for i, row in df_load_total.iterrows():
        dict_loadtotal[i+1] = df_load_total.loc[i, 'LOAD']
    
    df_solar = pd.read_csv(csv_solar, index_col=0)
    df_solar.reset_index(drop=True, inplace=True)
    dict_solar = dict()
    for i, row in df_solar.iterrows():
        for g in df_solar.columns:
            dict_solar[g, i+1] = df_solar.loc[i, g]

    instance.PowerForecast.store_values(dict_solar)
    instance.PowerForecast.store_values(sdata['xa'])
    instance.BusDemand.store_values(dict_busload)
    instance.Demand.store_values(dict_loadtotal)

    instance.SpinningReserveRequirement.reconstruct()
    instance.RegulatingReserveRequirement.reconstruct()

    return instance

def create_ED_model_season(season, dict_uniton):
    instance = create_UC_model_season(season)
    dir_scenario='/home/bxl180002/git/WindScenarioGeneration/'+ season
    snames, sdata = import_scenario_data(dir_scenario=dir_scenario)

    # Step 1 of 3: Fix all unit commitment status
    if dict_uniton:
        instance.UnitOn.set_values(dict_uniton)
    fixed = [instance.UnitOn[k].fixed for k in instance.UnitOn.iterkeys()]
    if False in fixed: # Fix all if anyone is not fixed.
        instance.UnitOn.fix()
        print "Commitment status fixed!"

    # Step 2 of 3: Deactivate all minimum online/offline constraints
    # This may not be necessary, but may help with computational performance
    instance.EnforceUpTimeConstraintsInitial.deactivate()
    instance.EnforceUpTimeConstraintsSubsequent.deactivate()
    instance.EnforceDownTimeConstraintsInitial.deactivate()
    instance.EnforceDownTimeConstraintsSubsequent.deactivate()

    # Step 3 of 3: Update wind power forecast with actual wind power
    # For debugging purpose, comment this block out to compare the objective 
    # values of the dispatch model with the UC model. Identical solution means 
    # correct.
    instance.PowerForecast.store_values(sdata['xa']) # Dispatch model always use actual data

    return instance

def solve_deterministic_for_season(season):
    dir_scenario='/home/bxl180002/git/WindScenarioGeneration/'+ season
    snames, sdata = import_scenario_data(dir_scenario=dir_scenario)

    print 'Case starts at {}'.format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    )

    if not os.path.isdir(season):
        os.mkdir(season)
    os.chdir(season)
    dirhome = os.getcwd()

    for s in snames:
        if not os.path.isdir(s):
            os.mkdir(s)
        os.chdir(s)

        instance = create_UC_model_season(season)
        instance.PowerForecast.store_values(sdata[s]) # Use scenario data
        optimizer = SolverFactory('cplex')
        results = optimizer.solve(instance)
        instance.solutions.load_from(results)
        store_csvs(instance, 'UC')
        print "Single case {} solved at {}, value: {:>.2f}.".format(
            s,
            datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
            value( instance.TotalCostObjective ),
        )

        # Solve dispatch model
        print 'Solving dispatch model...'
        print 'Dispatch model starts at {}'.format(
            datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
        )
        fixed = [instance.UnitOn[k].fixed for k in instance.UnitOn.iterkeys()]
        if False in fixed: # Fix all if anyone is not fixed.
            instance.UnitOn.fix()
            print "Commitment status fixed!"

        # Step 2 of 3: Deactivate all minimum online/offline constraints
        # This may not be necessary, but may help with computational performance
        instance.EnforceUpTimeConstraintsInitial.deactivate()
        instance.EnforceUpTimeConstraintsSubsequent.deactivate()
        instance.EnforceDownTimeConstraintsInitial.deactivate()
        instance.EnforceDownTimeConstraintsSubsequent.deactivate()

        # Step 3 of 3: Update wind power forecast with actual wind power
        # For debugging purpose, comment this block out to compare the objective 
        # values of the dispatch model with the UC model. Identical solution means 
        # correct.
        instance.PowerForecast.store_values(sdata['xa']) # Dispatch model always use actual data

        # Now we solve it...
        instance.preprocess() # Do we really need to do this?
        results = optimizer.solve(instance)
        instance.solutions.load_from(results)

        store_csvs(instance, 'ED')
        print "Dispatch model solved at {}, value: {:>.2f}.".format(
            datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
            value( instance.TotalCostObjective ),
        )
        os.chdir(dirhome)

    IP()

# def solve_dispatch_model(instance, dict_uniton=None):
#     snames, sdata = import_scenario_data()

#     # Step 1 of 3: Fix all unit commitment status
#     if dict_uniton:
#         instance.UnitOn.set_values(dict_uniton)
#     fixed = [instance.UnitOn[k].fixed for k in instance.UnitOn.iterkeys()]
#     if False in fixed: # Fix all if anyone is not fixed.
#         instance.UnitOn.fix()
#         print "Commitment status fixed!"

#     # Step 2 of 3: Deactivate all minimum online/offline constraints
#     # This may not be necessary, but may help with computational performance
#     instance.EnforceUpTimeConstraintsInitial.deactivate()
#     instance.EnforceUpTimeConstraintsSubsequent.deactivate()
#     instance.EnforceDownTimeConstraintsInitial.deactivate()
#     instance.EnforceDownTimeConstraintsSubsequent.deactivate()

#     # Step 3 of 3: Update wind power forecast with actual wind power
#     # For debugging purpose, comment this block out to compare the objective 
#     # values of the dispatch model with the UC model. Identical solution means 
#     # correct.
#     instance.PowerForecast.store_values(sdata['xa']) # Dispatch model always use actual data

#     # Now we solve it...
#     # instance.preprocess() # Do we really need to do this?
#     optimizer = SolverFactory('cplex')
#     results = optimizer.solve(instance)
#     instance.solutions.load_from(results)
#     return instance

def solve_after_stochastic(season, csvef):
    print 'Case starts at {}'.format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    )
    # csvef = '/home/bxl180002/git/FlexibleRampSCUC/results_runef_reduced_reserve/ef.csv'
    dirwork = os.path.dirname(csvef)
    dirhome = os.getcwd()

    os.chdir(dirwork)
    dict_uniton = extract_uniton(csvef)
    print 'Creating model...'
    instance = create_ED_model_season(season, dict_uniton)
    print 'Model created at {}'.format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    )
    optimizer = SolverFactory('cplex')
    print 'Solving dispatch model...'
    results = optimizer.solve(instance)
    instance.solutions.load_from(results)
    print 'Dispatch model starts at {}'.format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    )
    store_csvs(instance, 'ED')
    print "Dispatch model solved at {}, value: {:>.2f}.".format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
        value( instance.TotalCostObjective ),
    )
    os.chdir(dirhome)

if __name__ == "__main__":
    # independent_run_10_case(2012, 5, 10, True)

    dirhome = os.getcwd()
    season = 'season1'
    solve_deterministic_for_season(season)
    os.chdir(dirhome)

    # solve_after_stochastic()