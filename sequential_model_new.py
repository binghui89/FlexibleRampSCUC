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
from unit_commitment import GroupDataFrame, MyDataFrame, NewNetwork, create_model


def build_118_network():
    csv_bus               = '/home/bxl180002/git/FlexibleRampSCUC/118bus/bus.csv'
    csv_branch            = '/home/bxl180002/git/FlexibleRampSCUC/118bus/branch.csv'
    csv_ptdf              = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ptdf.csv'
    csv_gen               = '/home/bxl180002/git/FlexibleRampSCUC/118bus/generator_data_plexos_withRT.csv'
    csv_marginalcost      = '/home/bxl180002/git/FlexibleRampSCUC/118bus/marginalcost.csv'
    csv_blockmarginalcost = '/home/bxl180002/git/FlexibleRampSCUC/118bus/blockmarginalcost.csv'
    csv_blockoutputlimit  = '/home/bxl180002/git/FlexibleRampSCUC/118bus/blockoutputlimit.csv'

    df_bus              = pd.read_csv(csv_bus,               index_col=['BUS_ID'])
    df_branch           = pd.read_csv(csv_branch,            index_col=['BR_ID'])
    df_ptdf             = pd.read_csv(csv_ptdf,              index_col=0)
    df_gen              = pd.read_csv(csv_gen,               index_col=0)
    df_margcost         = pd.read_csv(csv_marginalcost,      index_col=0)
    df_blockmargcost    = pd.read_csv(csv_blockmarginalcost, index_col=0)
    df_blockoutputlimit = pd.read_csv(csv_blockoutputlimit,  index_col=0)

    df_bus['VOLL'] = 9000
    baseMVA = 100

    # This is to fix a bug in the 118 bus system
    if 'Hydro 31' in df_gen.index:
        df_gen.drop('Hydro 31', inplace=True)
    # Geo 01 is a thermal gen, set startup and shutdown costs to non-zero to 
    # force UnitStartUp and UnitShutDn being intergers.
    if 'Geo 01' in df_gen.index:
        df_gen.at['Geo 01', 'STARTUP']  = 50
        df_gen.at['Geo 01', 'SHUTDOWN'] = 50
        # df_margcost.at['Geo 01', 'nlcost'] = 10
        # df_margcost.at['Geo 01', '1']      = 10

    # Add start-up and shut-down time in a quick and dirty way
    df_gen.loc[:, 'STARTUP_TIME']  = df_gen.loc[:, 'MINIMUM_UP_TIME']
    df_gen.loc[:, 'SHUTDOWN_TIME'] = df_gen.loc[:, 'MINIMUM_UP_TIME']
    # df_gen.loc[df_gen['STARTUP_TIME']>=12,   'STARTUP_TIME']  = 12
    # df_gen.loc[df_gen['SHUTDOWN_TIME']>=12, 'SHUTDOWN_TIME']  = 12

     # AGC related parameters, need update
    df_gen['AGC_MODE'] = 'RAW' # Possible options: NA, RAW, SMOOTH, CPS2
    df_gen['DEAD_BAND'] = 5 # MW, 5 MW from FESTIV

    # Now, build the network object for the UCED model
    ############################################################################
    network_118 = NewNetwork()
    network_118.set_bus = set(df_bus.index.tolist())
    network_118.dict_set_gens = dict()
    network_118.dict_set_gens['ALL']        = set(df_gen.index)
    network_118.dict_set_gens['THERMAL']    = set(df_gen[df_gen['GEN_TYPE']=='Thermal'].index)
    network_118.dict_set_gens['NONTHERMAL'] = set(df_gen[~(df_gen['GEN_TYPE']=='Thermal')].index)
    network_118.dict_set_gens['RENEWABLE']  = set(df_gen[df_gen['GEN_TYPE']=='Renewable'].index)
    network_118.dict_set_gens['HYDRO']      = set(df_gen[df_gen['GEN_TYPE']=='Hydro'].index)
    network_118.dict_set_gens['WIND']       = set([i for i in df_gen.index if i.startswith('Wind')])
    network_118.dict_set_gens['THERMAL_slow'] = df_gen[
        (df_gen['MINIMUM_UP_TIME'] > 1) & (df_gen['GEN_TYPE']=='Thermal')
    ].index
    network_118.dict_set_gens['THERMAL_fast'] = df_gen[
        (df_gen['MINIMUM_UP_TIME'] <= 1) & (df_gen['GEN_TYPE']=='Thermal')
    ].index

    network_118.set_block = set(df_blockmargcost.columns)

    network_118.set_branches = set(df_branch.index)
    network_118.set_branches_enforced = network_118.set_branches

    network_118.dict_genbus_by_gen = df_gen['GEN_BUS'].to_dict()

    network_118.dict_block_size_by_gen_block = MyDataFrame(df_blockoutputlimit).to_dict_2d()
    network_118.dict_block_cost_by_gen_block = MyDataFrame(df_blockmargcost).to_dict_2d()
    network_118.dict_block_size0_by_gen = df_margcost['Pmax0'].to_dict()
    network_118.dict_block_cost0_by_gen = df_margcost['nlcost'].to_dict()

    network_118.dict_ptdf_by_br_bus = MyDataFrame(df_ptdf).to_dict_2d()
    network_118.dict_busvoll_by_bus = df_bus[ df_bus['PD']>0 ][ 'VOLL' ].to_dict()

    network_118.dict_linelimits_by_br = df_branch['RATE_A'].to_dict()

    network_118.dict_pmin_by_gen         = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'PMIN'].to_dict()
    network_118.dict_pmax_by_gen         = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'PMAX'].to_dict()
    network_118.dict_rampup_by_gen       = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'RAMP_10'].to_dict() # MW/hr
    network_118.dict_rampdn_by_gen       = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'RAMP_10'].to_dict() # MW/hr
    network_118.dict_h_startup_by_gen    = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'STARTUP_TIME'].to_dict() # hr
    network_118.dict_h_shutdn_by_gen     = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'SHUTDOWN_TIME'].to_dict() # hr
    network_118.dict_h_minup_by_gen      = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'MINIMUM_UP_TIME'].to_dict() # hr
    network_118.dict_h_mindn_by_gen      = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'MINIMUM_DOWN_TIME'].to_dict() # hr
    network_118.dict_t_uniton_by_gen     = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'GEN_STATUS'].to_dict()
    network_118.dict_cost_startup_by_gen = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'STARTUP'].to_dict()
    network_118.dict_cost_shutdn_by_gen  = df_gen.loc[network_118.dict_set_gens['THERMAL'], 'SHUTDOWN'].to_dict()

    network_118.dict_reserve_margin['REGUP'] = 0.05
    network_118.dict_reserve_margin['REGDN'] = 0.05
    network_118.dict_reserve_margin['SPNUP'] = 0.1

    dfs = GroupDataFrame()
    dfs.df_bus              = df_bus
    dfs.df_branch           = df_branch
    dfs.df_ptdf             = df_ptdf
    dfs.df_gen              = df_gen
    dfs.df_margcost         = df_margcost
    dfs.df_blockmargcost    = df_blockmargcost
    dfs.df_blockoutputlimit = df_blockoutputlimit

    return dfs, network_118

def build_texas_network():
    csv_bus               = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/bus.csv'
    csv_branch            = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/branch.csv'
    csv_ptdf              = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ptdf.csv'
    csv_gen               = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/generator_data_plexos_withRT.csv'
    csv_marginalcost      = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/marginalcost.csv'
    csv_blockmarginalcost = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/blockmarginalcost.csv'
    csv_blockoutputlimit  = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/blockoutputlimit.csv'

    df_bus              = pd.read_csv(csv_bus,               index_col=['BUS_ID'])
    df_branch           = pd.read_csv(csv_branch,            index_col=['BR_ID'])
    df_ptdf             = pd.read_csv(csv_ptdf,              index_col=0)
    df_gen              = pd.read_csv(csv_gen,               index_col=0)
    df_margcost         = pd.read_csv(csv_marginalcost,      index_col=0)
    df_blockmargcost    = pd.read_csv(csv_blockmarginalcost, index_col=0)
    df_blockoutputlimit = pd.read_csv(csv_blockoutputlimit,  index_col=0)

    df_bus['VOLL'] = 9000
    baseMVA = 100

    # Enforce the 230 kV limits
    kv_level = 230
    bus_kVlevel_set = set(df_bus[df_bus['BASEKV']>=kv_level].index)
    set_branch_enforced = {
        i for i in df_branch.index
        if df_branch.loc[i,'F_BUS'] in bus_kVlevel_set 
        and df_branch.loc[i,'T_BUS'] in bus_kVlevel_set
    }
    df_ptdf = df_ptdf.loc[set_branch_enforced,:].copy()

    # Add start-up and shut-down time in a quick and dirty way
    df_gen.loc[:, 'STARTUP_TIME']  = df_gen.loc[:, 'MINIMUM_UP_TIME']
    df_gen.loc[:, 'SHUTDOWN_TIME'] = df_gen.loc[:, 'MINIMUM_UP_TIME']
    # df_gen.loc[df_gen['STARTUP_TIME']>=12,   'STARTUP_TIME']  = 12
    # df_gen.loc[df_gen['SHUTDOWN_TIME']>=12, 'SHUTDOWN_TIME']  = 12

     # AGC related parameters, need update
    df_gen['AGC_MODE'] = 'RAW' # Possible options: NA, RAW, SMOOTH, CPS2
    df_gen['DEAD_BAND'] = 5 # MW, 5 MW from FESTIV

    # Update start-up/shut-down times
    for i, row in df_gen.iterrows():
        cap    = df_gen.loc[i, 'PMAX']
        s_cost = df_gen.loc[i, 'STARTUP']
        tmin   = df_gen.loc[i, 'MINIMUM_UP_TIME']
        t0     = df_gen.loc[i, 'GEN_STATUS']
        if i.startswith('coal'):
            if cap <= 300:
                # Subcritical steam cycle
                s_cost = 16.28*cap
                tmin = 4 # Min on/offline time
                t0 = tmin # Initial online time
            else:
                # Supercritical steam cycle
                s_cost = 29.44*cap
                tmin = 12
                t0 = tmin
        elif i.startswith('ng'):
            if cap <= 50:
                # Small aeroderivative turbines
                s_cost = 8.23*cap
                tmin = 1
                t0 = tmin
            elif cap <= 100:
                # Small aeroderivative turbines
                s_cost = 8.23*cap
                tmin = 3
                t0 = tmin
            elif cap <= 450:
                # Heavy-duty GT
                s_cost = 1.70*cap
                tmin = 5
                t0 = tmin
            else:
                # CCGT
                s_cost = 1.74*cap
                tmin = 8
                t0 = tmin
        elif i.startswith('nuc'):
            # Supercritical steam cycle
            s_cost = 29.44*cap
            tmin = 100
            t0 = 1 # Nuclear units never go offline

        df_gen.loc[i, 'STARTUP']  = s_cost
        df_gen.loc[i, 'SHUTDOWN'] = s_cost
        df_gen.loc[i, 'MINIMUM_UP_TIME']   = tmin
        df_gen.loc[i, 'MINIMUM_DOWN_TIME'] = tmin
        df_gen.loc[i, 'GEN_STATUS'] = t0 # All but nuclear units are free to be go offline

    df_gen['STARTUP_RAMP']  = df_gen[['STARTUP_RAMP','PMIN']].max(axis=1)
    df_gen['SHUTDOWN_RAMP'] = df_gen[['SHUTDOWN_RAMP','PMIN']].max(axis=1)

    # Assume renewable sources cost nothing to start
    df_gen.loc[df_gen['GEN_TYPE']=='Renewable', 'STARTUP'] = 0

    # Now, build the network object for the UCED model
    ############################################################################
    network_texas = NewNetwork()
    network_texas.set_bus = set(df_bus.index.tolist())
    network_texas.dict_set_gens = dict()
    network_texas.dict_set_gens['ALL']        = set(df_gen.index)
    network_texas.dict_set_gens['THERMAL']    = set(df_gen[df_gen['GEN_TYPE']=='Thermal'].index)
    network_texas.dict_set_gens['NONTHERMAL'] = set(df_gen[~(df_gen['GEN_TYPE']=='Thermal')].index)
    network_texas.dict_set_gens['RENEWABLE']  = set(df_gen[df_gen['GEN_TYPE']=='Renewable'].index)
    network_texas.dict_set_gens['HYDRO']      = set(df_gen[df_gen['GEN_TYPE']=='Hydro'].index)
    network_texas.dict_set_gens['WIND']       = set([i for i in df_gen.index if i.startswith('wind')])
    network_texas.dict_set_gens['THERMAL_slow'] = df_gen[
        (df_gen['MINIMUM_UP_TIME'] > 1) & (df_gen['GEN_TYPE']=='Thermal')
    ].index
    network_texas.dict_set_gens['THERMAL_fast'] = df_gen[
        (df_gen['MINIMUM_UP_TIME'] <= 1) & (df_gen['GEN_TYPE']=='Thermal')
    ].index

    network_texas.set_block = set(df_blockmargcost.columns)

    network_texas.set_branches = set(df_branch.index)
    network_texas.set_branches_enforced = set_branch_enforced

    network_texas.dict_genbus_by_gen = df_gen['GEN_BUS'].to_dict()

    network_texas.dict_block_size_by_gen_block = MyDataFrame(df_blockoutputlimit).to_dict_2d()
    network_texas.dict_block_cost_by_gen_block = MyDataFrame(df_blockmargcost).to_dict_2d()
    network_texas.dict_block_size0_by_gen = df_margcost['Pmax0'].to_dict()
    network_texas.dict_block_cost0_by_gen = df_margcost['nlcost'].to_dict()

    network_texas.dict_ptdf_by_br_bus = MyDataFrame(df_ptdf).to_dict_2d()
    network_texas.dict_busvoll_by_bus = df_bus[ df_bus['PD']>0 ][ 'VOLL' ].to_dict()

    network_texas.dict_linelimits_by_br = df_branch['RATE_A'].to_dict()

    network_texas.dict_pmin_by_gen         = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'PMIN'].to_dict()
    network_texas.dict_pmax_by_gen         = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'PMAX'].to_dict()
    network_texas.dict_rampup_by_gen       = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'RAMP_10'].to_dict() # MW/hr
    network_texas.dict_rampdn_by_gen       = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'RAMP_10'].to_dict() # MW/hr
    network_texas.dict_h_startup_by_gen    = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'STARTUP_TIME'].to_dict() # hr
    network_texas.dict_h_shutdn_by_gen     = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'SHUTDOWN_TIME'].to_dict() #hr
    network_texas.dict_h_minup_by_gen      = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'MINIMUM_UP_TIME'].to_dict() # hr
    network_texas.dict_h_mindn_by_gen      = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'MINIMUM_DOWN_TIME'].to_dict() # hr
    network_texas.dict_t_uniton_by_gen     = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'GEN_STATUS'].to_dict()
    network_texas.dict_cost_startup_by_gen = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'STARTUP'].to_dict()
    network_texas.dict_cost_shutdn_by_gen  = df_gen.loc[network_texas.dict_set_gens['THERMAL'], 'SHUTDOWN'].to_dict()

    network_texas.dict_reserve_margin['REGUP'] = 0.05
    network_texas.dict_reserve_margin['REGDN'] = 0.05
    network_texas.dict_reserve_margin['SPNUP'] = 0.1

    dfs = GroupDataFrame()
    dfs.df_bus              = df_bus
    dfs.df_branch           = df_branch
    dfs.df_ptdf             = df_ptdf
    dfs.df_gen              = df_gen
    dfs.df_margcost         = df_margcost
    dfs.df_blockmargcost    = df_blockmargcost
    dfs.df_blockoutputlimit = df_blockoutputlimit

    return dfs, network_texas

def test_dauc(casename, showing_gens='problematic'):
    t0 = time()
    content = ''

    nI_DAC = 1 # Number of DAUC intervals in an hour
    nI_RTC = 4 # Number of RTUC intervals in an hour
    nI_RTD = 12 # Number of RTED intervals in an hour
    nI_AGC = 3600/6 # Number of AGC intervals in an hour

    # Time table, should we add AGC time as well?
    df_timesequence = pd.DataFrame(columns=['DAC', 'RTC', 'RTD'], dtype='int')
    for i in range(1, 25):
        for j in range(1, 5):
            for k in range(1,4):
                index = (i-1)*12+(j-1)*3 + k
                df_timesequence.loc[index, 'DAC'] = i
                df_timesequence.loc[index, 'RTC'] = (i-1)*4 + j
                df_timesequence.loc[index, 'RTD'] = index
    df_timesequence = df_timesequence.astype('int')

    # Initial conditions
    dict_UnitOnT0State = dict()
    dict_PowerGeneratedT0 = dict()

    if casename == '118':
        dfs, network = build_118_network()

        # Data files
        csv_busload           = '/home/bxl180002/git/FlexibleRampSCUC/118bus/loads.csv'
        csv_genfor            = '/home/bxl180002/git/FlexibleRampSCUC/118bus/generator.csv'
        csv_busload_RTC       = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ha_loads.csv'
        csv_genfor_RTC        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ha_generator.csv'
        csv_busload_RTD       = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ed_loads.csv'
        csv_genfor_RTD        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ed_generator.csv'
        csv_busload_AGC       = '/home/bxl180002/git/FlexibleRampSCUC/118bus/agc_loads.csv'
        csv_genfor_AGC        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/agc_generator.csv'

        # Prepare day-ahead UC data
        df_busload = pd.read_csv(csv_busload, index_col=0)
        df_genfor  = pd.read_csv(csv_genfor, index_col=0)
        df_busload = MyDataFrame(df_busload.loc[:, df_busload.columns.difference(['LOAD'])])
        df_genfor  = MyDataFrame(df_genfor)
        df_genfor.index = range(1, 25) # Kwami's convention: time starts from 1...
        df_genfor_nonthermal = df_genfor.loc[:, network.dict_set_gens['NONTHERMAL']]
        df_genfor_nonthermal.fillna(0, inplace=True)

        # Prepare real-time UC (hourly ahead) data
        df_busload_RTC = pd.read_csv(csv_busload_RTC, index_col=['Slot'])
        df_genfor_RTC  = pd.read_csv(csv_genfor_RTC, index_col=['Slot'])
        df_busload_RTC = MyDataFrame(df_busload_RTC.loc[:, df_busload_RTC.columns.difference(['LOAD'])])
        df_genfor_RTC  = MyDataFrame(df_genfor_RTC)
        df_genfor_RTC  = df_genfor_RTC.loc[:, network.dict_set_gens['NONTHERMAL']]
        df_genfor_RTC.fillna(0, inplace=True)

        # Prepare economic dispatch data
        df_busload_RTD = pd.read_csv(csv_busload_RTD, index_col=['Slot'])
        df_genfor_RTD  = pd.read_csv(csv_genfor_RTD, index_col=['Slot'])
        df_busload_RTD = MyDataFrame(df_busload_RTD.loc[:, df_busload_RTD.columns.difference(['LOAD'])])
        df_genfor_RTD  = MyDataFrame(df_genfor_RTD)
        df_genfor_RTD  = df_genfor_RTD.loc[:, network.dict_set_gens['NONTHERMAL']]
        df_genfor_RTD.fillna(0, inplace=True)

        # Prepare AGC data
        # df_busload_AGC = pd.read_csv(csv_busload_AGC, index_col='Slot')
        # df_genfor_AGC  = pd.read_csv(csv_genfor_AGC, index_col='Slot')
        # if 'LOAD' in df_busload_AGC.columns.difference(['LOAD']):
        #     df_busload_AGC = df_busload_AGC.loc[:, df_busload_AGC.columns.difference(['LOAD'])]
        # df_busload_AGC = MyDataFrame(df_busload_AGC.loc[:, df_busload_AGC.columns.difference(['LOAD'])])
        # df_busload_AGC.fillna(0, inplace=True)
        # df_busload_full_AGC = pd.DataFrame(0, index=df_busload_AGC.index, columns=network.ls_bus) # For ACE calculation, include all buses
        # df_busload_full_AGC.loc[:, df_busload_AGC.columns] = df_busload_AGC
        # df_genfor_AGC  = MyDataFrame(df_genfor_AGC)
        # df_genfor_AGC.fillna(0, inplace=True)


        # For debugging

        # Control initial conditions
        ########################################################################
        # All units are off
        # for g in network.dict_set_gens['ALL']:
        #     dict_PowerGeneratedT0[g] = 0
        #     if g in network.dict_set_gens['THERMAL']:
        #         dict_UnitOnT0State[g]    = -12

        # All units are on and at minimum generation levels
        for g in network.dict_set_gens['ALL']:
            dict_PowerGeneratedT0[g] = dfs.df_gen.at[g, 'PMIN']
            if g in network.dict_set_gens['THERMAL']:
                dict_UnitOnT0State[g]    = 12

        # All units are on and at maximum generation levels
        # for g in network.dict_set_gens['ALL']:
        #     dict_PowerGeneratedT0[g] = dfs.df_gen.at[g, 'PMAX']
        #     if g in network.dict_set_gens['THERMAL']:
        #         dict_UnitOnT0State[g]    = 12
        ########################################################################

        # Start-up/shut-down tests
        ########################################################################
        # Start-up test
        # Pmax: 595 MW, Pmin: 298.29 MW, Tsu = Tsd = 8 hrs
        dict_PowerGeneratedT0['CC NG 35'] = 298.29/8*4
        dict_UnitOnT0State['CC NG 35'] = 4

        # Shut-down test
        # Pmax: 943.5 MW, Pmin: 503.86 MW, Tsu = Tsd = 12 hrs
        dict_PowerGeneratedT0['CC NG 16'] = 503.86/8*3
        dict_UnitOnT0State['CC NG 16'] = 12

        # Start-up/shut-down test for super slow units, of which the start-up 
        # time is over the length of the model horizon.
        # Pmax: 20 MW, Pmin: 6 MW, Tsu = Tsd = 48 hrs

        # Start-up test # 1: Beginning of start-up period
        dict_PowerGeneratedT0['ST Coal 01'] = 6.0/48*3
        dict_UnitOnT0State['ST Coal 01'] = 3

        # Start-up test # 2: End of start-up period 
        # dict_PowerGeneratedT0['ST Coal 01'] = 6.0/48*40
        # dict_UnitOnT0State['ST Coal 01'] = 40

        # Shut-down test # 1: End of shut-down period
        # Sadly, we still don't know how to model the case where z_{g, t} = 1 
        # when t > te, in another word, when the ending interval of the 
        # shut-down process is beyond the end of model horizon. However, such 
        # scenario won't be the output of the UC model, since the UC model has 
        # to know then the generator shuts down (i.e., z_{g, t} = 1) first hand.
        # In another word, in the UC model results, when z_{g, t} = 1, t is 
        # always less than te.
        # dict_PowerGeneratedT0['ST Coal 01'] = 6.0/48*3
        # dict_UnitOnT0State['ST Coal 01'] = 90
        ########################################################################

    elif casename == 'TX':
        # Texas 2000 system

        dfs, network = build_texas_network()

        csv_bus               = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/bus.csv'
        csv_branch            = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/branch.csv'
        csv_ptdf              = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ptdf.csv'
        csv_gen               = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/generator_data_plexos_withRT.csv'
        csv_marginalcost      = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/marginalcost.csv'
        csv_blockmarginalcost = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/blockmarginalcost.csv'
        csv_blockoutputlimit  = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/blockoutputlimit.csv'
        csv_busload           = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/loads.csv'
        csv_genfor            = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/generator.csv'
        csv_busload_RTC        = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ha_load.csv'
        csv_genfor_RTC        = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ha_generator.csv'
        csv_busload_RTD       = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ed_load.csv'
        csv_genfor_RTD        = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ed_generator.csv'

        df_busload = pd.read_csv(csv_busload, index_col=0)
        df_genfor  = pd.read_csv(csv_genfor, index_col=0)
        df_busload = MyDataFrame(df_busload.loc[:, df_busload.columns.difference(['LOAD'])])
        df_genfor  = MyDataFrame(df_genfor)
        df_genfor.index = range(1, 25) # Kwami's convention: time starts from 1...
        df_genfor_nonthermal = df_genfor.loc[:, network.dict_set_gens['NONTHERMAL']]
        df_genfor_nonthermal.fillna(0, inplace=True)

        for g in network.dict_set_gens['ALL']:
            dict_PowerGeneratedT0[g] = dfs.df_gen.at[g, 'PMIN']
            if g in network.dict_set_gens['THERMAL']:
                dict_UnitOnT0State[g]    = 12
    ############################################################################

    # Start DAUC
    ############################################################################
    model = create_model(
        network,
        df_busload,
        df_genfor_nonthermal,
        nI=nI_DAC,
        dict_UnitOnT0State=dict_UnitOnT0State,
        dict_PowerGeneratedT0=dict_PowerGeneratedT0,
    )
    msg = 'Model created at: {:>.2f} s'.format(time() - t0)
    print(msg)
    content += msg
    content += '\n'

    instance = model
    optimizer = SolverFactory('cplex')
    results = optimizer.solve(instance, options={"mipgap":0.001})
    instance.solutions.load_from(results)
    msg = 'Model solved at: {:>.2f} s, objective: {:>.2f}'.format(
            time() - t0,
            value(instance.TotalCostObjective)
        )
    print(msg)
    content += msg
    content += '\n'

    # For debugging purpose only, normally, all slack variables should be 0
    print 'Non-zero slack variables:'
    items = [
        'Slack_startup_lower',  'Slack_startup_upper',   'Slack_shutdown_lower', 
        'Slack_shutdown_upper', 'Slack_overlap_startup', 'Slack_overlap_shutdown', 
        'Slack_rampup',         'Slack_rampdn', 
    ]
    for i in items:
        attr = getattr(instance, i)
        for k in attr.iterkeys():
            if abs(value(attr[k])) > 1e-10: # Slack variable tolerance
                print i, k, value(attr[k])
    ############################################################################
    # End of DAUC

    # Processing some DAUC results
    ############################################################################
    # For those gens showing weird results, where y or z are fractions, while 
    # supposedly they can only be 0 or 1.
    set_gens = set()
    for k in instance.UnitStartUp:
        y, z = value(instance.UnitStartUp[k]), value(instance.UnitShutDn[k])
        if ((abs(y) > 1e-3) and (abs(y-1)>1e-3)) or ((abs(z) > 1e-3) and (abs(z-1) > 1e-3)):
            print k, value(instance.UnitStartUp[k]), value(instance.UnitShutDn[k])
            set_gens.add(k[0])

    # Results container
    df_POWER_START_DAC   = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.AllGenerators)
    df_POWER_END_DAC     = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.AllGenerators)
    df_UNITON_DAC        = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_SIGMAUP_DAC       = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_SIGMADN_DAC       = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_SIGMAPOWERUP_DAC  = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_SIGMAPOWERDN_DAC  = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_REGUP_DAC         = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_REGDN_DAC         = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_SPNUP_DAC         = pd.DataFrame(np.nan, index=instance.TimePeriods, columns=instance.ThermalGenerators)
    df_UNITONT0_DAC      = pd.DataFrame(np.nan, index=[instance.TimePeriods.first()-1], columns=instance.ThermalGenerators)
    df_SIGMADNT0_DAC     = pd.DataFrame(0, index=[instance.TimePeriods.first()-1], columns=instance.ThermalGenerators)
    for g in instance.AllGenerators:
        for t in instance.TimePeriods:
            df_POWER_END_DAC.at[t, g] = value(instance.PowerGenerated[g, t])
            if t == instance.TimePeriods.first():
                df_POWER_START_DAC.at[t, g] = value(instance.PowerGeneratedT0[g])
            else:
                df_POWER_START_DAC.at[t, g] = value(instance.PowerGenerated[g, t-1])
    for g in instance.ThermalGenerators:
        df_UNITONT0_DAC.at[instance.TimePeriods.first()-1, g] = value(instance.UnitOnT0[g])
        if g in instance.SigmaDnT0:
            df_SIGMADNT0_DAC.at[instance.TimePeriods.first()-1, g] = value(instance.SigmaDnT0[g])
        for t in instance.TimePeriods:
            df_UNITON_DAC.at[t, g]       = value(instance.UnitOn[g, t])
            df_SIGMAUP_DAC.at[t, g]      = value(instance.SigmaUp[g, t])
            df_SIGMADN_DAC.at[t, g]      = value(instance.SigmaDn[g, t])
            df_SIGMAPOWERUP_DAC.at[t, g] = value(instance.SigmaPowerTimesUp[g, t])
            df_SIGMAPOWERDN_DAC.at[t, g] = value(instance.SigmaPowerTimesDn[g, t])
            df_REGUP_DAC.at[t, g]        = value(instance.RegulatingReserveUpAvailable[g, t])
            df_REGDN_DAC.at[t, g]        = value(instance.RegulatingReserveDnAvailable[g, t])
            df_SPNUP_DAC.at[t, g]        = value(instance.SpinningReserveUpAvailable[g, t])

    # Calculate unit start-up/shut-down binary indicator
    df_unitonaug = pd.concat(
        [df_UNITONT0_DAC[df_UNITON_DAC.columns], df_UNITON_DAC[df_UNITON_DAC.columns]], 
        axis=0
    ) # Augmented df_uniton
    df_UNITSTUP_DAC = pd.DataFrame(
        np.maximum(df_unitonaug.loc[1:, :].values - df_unitonaug.iloc[0:-1, :].values, 0),
        index=df_UNITON_DAC.index,
        columns=df_unitonaug.columns,
    ).astype(int)
    df_UNITSTDN_DAC = pd.DataFrame(
        np.maximum(df_unitonaug.iloc[0:-1, :].values - df_unitonaug.loc[1:, :].values, 0),
        index=df_UNITON_DAC.index,
        columns=df_unitonaug.columns,
    ).astype(int)
    df_power_mean = pd.DataFrame(
        (df_POWER_START_DAC + df_POWER_END_DAC)/2, 
        index=instance.TimePeriods, 
        columns=instance.AllGenerators,
    )

    # Collect thermal generator information
    ls_dict_therm  = list()
    for g in instance.ThermalGenerators:
        for a in [
            'PowerGenerated', 
            'UnitOn', 'UnitStartUp', 'UnitShutDn', 
            'SigmaUp', 'SigmaDn', 
            'SigmaPowerTimesUp', 'SigmaPowerTimesDn'
        ]:
            attr = getattr(instance, a)
            dict_row = {'Gen': g, 'Var': a}
            for t in instance.TimePeriods:
                dict_row[t] = value(attr[g, t])
            ls_dict_therm.append(dict_row)
    df_therm = pd.DataFrame(ls_dict_therm, columns=['Gen', 'Var']+list(instance.TimePeriods.value))

    ############################################################################
    # End of processing some DAUC results

    # Prepare for the RTUC run
    ############################################################################
    nI_RTCperDAC = nI_RTC/nI_DAC # Number of RTC intervals per DAC interval, must be integer
    # df_UNITSTUP_DAC2RTC = pd.DataFrame(0, index=df_busload_RTC.index, columns=df_UNITSTUP_DAC.columns)
    # df_UNITSTDN_DAC2RTC = pd.DataFrame(0, index=df_busload_RTC.index, columns=df_UNITSTDN_DAC.columns)
    # df_UNITSTUP_DAC2RTC.loc[(df_UNITSTUP_DAC.index-1)*nI_RTCperDAC+1, :] = df_UNITSTUP_DAC.values
    # df_UNITSTDN_DAC2RTC.loc[df_UNITSTDN_DAC.index*nI_RTCperDAC, :]       = df_UNITSTDN_DAC.values

    # df_UNITON_DAC2RTC = pd.DataFrame(
    #     (df_UNITON_DAC+df_UNITSTDN_DAC).values.astype(int).repeat(nI_RTCperDAC, axis=0) - df_UNITSTDN_DAC2RTC.values,
    #     index=df_busload_RTC.index, 
    #     columns=df_UNITON_DAC.columns,
    # )
    # df_SIGMAUP_DAC2RTC = pd.DataFrame(
    #     df_SIGMAUP_DAC.astype(int).values.repeat(nI_RTCperDAC, axis=0),
    #     index=df_busload_RTC.index, 
    #     columns=df_UNITON_DAC.columns,
    # )
    # df_SIGMADN_DAC2RTC = pd.DataFrame(
    #     pd.concat([df_SIGMADNT0_DAC, df_SIGMADN_DAC], axis=0).astype(int).values.repeat(nI_RTCperDAC, axis=0)[1:df_busload_RTC.index.size+1, :],
    #     index=df_busload_RTC.index, 
    #     columns=df_UNITON_DAC.columns,
    # )

    # NOTE: np array's index starts from 0
    ar_UNITSTUP_DAC2RTC = np.zeros((df_UNITSTUP_DAC.shape[0]*nI_RTCperDAC, df_UNITSTUP_DAC.shape[1]), dtype=int)
    ar_UNITSTDN_DAC2RTC = np.zeros((df_UNITSTUP_DAC.shape[0]*nI_RTCperDAC, df_UNITSTUP_DAC.shape[1]), dtype=int)
    ar_UNITSTUP_DAC2RTC[(df_UNITSTUP_DAC.index-1)*nI_RTCperDAC, :] = df_UNITSTUP_DAC.values
    ar_UNITSTDN_DAC2RTC[df_UNITSTDN_DAC.index*nI_RTCperDAC-1, :]   = df_UNITSTDN_DAC.values

    ar_UNITON_DAC2RTC = (df_UNITON_DAC+df_UNITSTDN_DAC).values.astype(int).repeat(nI_RTCperDAC, axis=0) - ar_UNITSTDN_DAC2RTC

    ar_SIGMAUP_DAC2RTC = df_SIGMAUP_DAC.astype(int).values.repeat(nI_RTCperDAC, axis=0)
    ar_SIGMADN_DAC2RTC = np.concatenate(
        [df_SIGMADNT0_DAC.values, df_SIGMADN_DAC.values], 
        axis=0
    ).repeat(nI_RTCperDAC, axis=0)[1:df_busload_RTC.index.size+1, :]

    # Binary indicator of a unit is up, starting-up, or  shuttind-down. 
    # A unit is up if power >= Pmin, this means nominal ramp rate applies in this interval.
    # A unit is starting up if power < Pmin and it is ramping up at its start-up ramp rate.
    # A unit is shutting down if power < Pmin and it is ramping down at its shut-down ramp rate.
    # Note the differences: 
    # ar_is_up_DAC2RTC           <--> ar_UNITON_DAC2RTC
    # ar_is_startingup_DAC2RTC   <--> ar_SIGMAUP_DAC2RTC
    # ar_is_shuttingdown_DAC2RTC <--> ar_SIGMADN_DAC2RTC
    # A unit's status can only be one of the following exclusively:
    # (1) up, (2) starting-up, (3) shutting-down or (4) offline, so the 
    # following equation always holds: 
    # ar_is_up_DAC2RTC + ar_is_startingup_DAC2RTC + ar_is_shuttingdown_DAC2RTC <= 1
    ar_is_startingup_DAC2RTC   = ar_SIGMAUP_DAC2RTC
    ar_is_shuttingdown_DAC2RTC = np.concatenate([df_SIGMADNT0_DAC.values, ar_SIGMADN_DAC2RTC], axis=0)[0:-1]
    ar_is_up_DAC2RTC = np.maximum(
        0,
        ar_UNITON_DAC2RTC + ar_UNITSTDN_DAC2RTC 
        - ar_is_startingup_DAC2RTC 
        - ar_is_shuttingdown_DAC2RTC
    )

    ar_pmin  = dfs.df_gen.loc[df_UNITON_DAC.columns, 'PMIN'].values
    ar_pmax  = dfs.df_gen.loc[df_UNITON_DAC.columns, 'PMAX'].values
    ar_p_aug = np.concatenate(
        [
            df_POWER_START_DAC[df_UNITON_DAC.columns].values[0,:].reshape(1, df_UNITON_DAC.columns.size),
            df_POWER_END_DAC[df_UNITON_DAC.columns].values
        ],
        axis=0
    ) # Arrary of power generation from DAUC, including initial power level P0 in the first row

    ar_POWER_END_DAC = np.empty((df_UNITON_DAC.shape[0]*nI_RTCperDAC, df_UNITON_DAC.shape[1]))
    xp = np.concatenate([[df_UNITON_DAC.index.values[0] - 1], df_UNITON_DAC.index.values]).astype(float)/df_UNITON_DAC.index.values.max()
    for i in np.arange(ar_p_aug.shape[1]):
        ar_POWER_END_DAC[:, i] = np.interp(
            df_busload_RTC.index.values.astype(float)/df_busload_RTC.index.max(),
            xp,
            ar_p_aug[:, i]
        )

    # All ramp rate units are in MW/interval
    # ar_ramp_up_RTC = dfs.df_gen.loc[df_UNITON_DAC.columns, 'RAMP_10'].values/nI_RTC
    ar_ramp_dn_RTC = dfs.df_gen.loc[df_UNITON_DAC.columns, 'RAMP_10'].values/nI_RTC
    # ar_ramp_stup_RTC = ar_pmin/(nI_RTC*dfs.df_gen.loc[df_UNITON_DAC.columns, 'STARTUP_TIME'].values)
    # ar_ramp_stdn_RTC = ar_pmin/(nI_RTC*dfs.df_gen.loc[df_UNITON_DAC.columns, 'SHUTDOWN_TIME'].values)

    # Upper and lower dispatch limits
    ar_dispatch_max_RTC = np.empty((df_UNITSTUP_DAC.shape[0]*nI_RTCperDAC, df_UNITSTUP_DAC.shape[1]))
    ar_dispatch_min_RTC = np.empty((df_UNITSTUP_DAC.shape[0]*nI_RTCperDAC, df_UNITSTUP_DAC.shape[1]))

    for i in np.arange(df_UNITSTUP_DAC.shape[0]*nI_RTCperDAC-1, -1, -1):
        if i == df_UNITSTUP_DAC.shape[0]*nI_RTCperDAC-1:
            ar_dispatch_max_RTC[i, :] = (
                ar_pmax*ar_is_up_DAC2RTC[-1, :] 
                + (
                    ar_is_startingup_DAC2RTC[-1, :] 
                    + ar_is_shuttingdown_DAC2RTC[-1, :]
                )*ar_POWER_END_DAC[-1, :]
            )
            ar_dispatch_min_RTC[i, :] = (
                ar_pmin*ar_is_up_DAC2RTC[-1, :] 
                + (
                    ar_is_startingup_DAC2RTC[-1, :] 
                    + ar_is_shuttingdown_DAC2RTC[-1, :]
                )*ar_POWER_END_DAC[-1, :]
            )
        else:
            i_next = i+1
            tmp = ar_is_up_DAC2RTC[i, :]*ar_is_up_DAC2RTC[i_next, :]
            ar_dispatch_max_RTC[i, :] = np.minimum(
                ar_pmax, 
                tmp*(ar_dispatch_max_RTC[i_next, :] + ar_ramp_dn_RTC) +
                (1-tmp)*ar_POWER_END_DAC[i, :]
            )
            ar_dispatch_min_RTC[i, :] = np.minimum(
                ar_pmin, 
                ar_dispatch_max_RTC[i, :]
            )

    df_UNITON_DAC2RTC = pd.DataFrame(
        ar_UNITON_DAC2RTC,
        index=df_busload_RTC.index, 
        columns=df_UNITON_DAC.columns,
    )
    df_UNITSTUP_DAC2RTC = pd.DataFrame(
        ar_UNITSTUP_DAC2RTC,
        index=df_busload_RTC.index, 
        columns=df_UNITON_DAC.columns,
    )
    df_UNITSTDN_DAC2RTC = pd.DataFrame(
        ar_UNITSTDN_DAC2RTC,
        index=df_busload_RTC.index, 
        columns=df_UNITON_DAC.columns,
    )
    df_dispatch_max_RTC = pd.DataFrame(
        ar_dispatch_max_RTC,
        index=df_busload_RTC.index, 
        columns=df_UNITON_DAC.columns,
    )
    df_dispatch_min_RTC = pd.DataFrame(
        ar_dispatch_min_RTC,
        index=df_busload_RTC.index, 
        columns=df_UNITON_DAC.columns,
    )

    # # Plot the upper and lower dispatch limits
    # ls_gens = df_UNITON_DAC.columns.tolist()
    # for i in range(0, len(ls_gens)):
    #     g = ls_gens[i]
    #     if i%9 == 0:
    #         plt.figure()
    #     ax = plt.subplot(3, 3, i%9+1)
    #     p0 = value(instance.PowerGeneratedT0[g])
    #     p = np.concatenate([[p0], df_POWER_END_DAC[g].values])
    #     ar_upper = np.concatenate([[p0], df_dispatch_max_RTC[g].values])
    #     ar_lower = np.concatenate([[p0], df_dispatch_min_RTC[g].values])
    #     ax.plot(
    #         np.concatenate(
    #             [
    #                 [0],
    #                 df_POWER_START_DAC.index.values
    #             ]
    #         ).astype(float)/df_POWER_START_DAC.index.max(),
    #         p,
    #         'k',
    #     )
    #     ax.fill_between(
    #         np.concatenate(
    #             [
    #                 [0],
    #                 df_dispatch_max_RTC.index.values
    #             ]
    #         ).astype(float)/df_dispatch_max_RTC.index.max(), 
    #         ar_lower, 
    #         ar_upper, 
    #         color='b', 
    #         alpha=0.2,
    #     )
    #     ax.plot(
    #         np.concatenate(
    #             [
    #                 [0],
    #                 df_POWER_START_DAC.index.values
    #             ]
    #         ).astype(float)/df_POWER_START_DAC.index.max(),
    #         [value(instance.MinimumPowerOutput[g])]*25,
    #         'r',
    #     )
    #     ax.set_title( g + ', Tsd = {:>g}'.format(value(instance.ShutdownHour[g])))
    #     if i%9 == 8:
    #         plt.show()
    #     elif i == len(ls_gens) - 1:
    #         plt.show()

    IP()

    for i_rtuc in range(1, 2):
        t_start = i_rtuc # 4*(i_rtuc-1) + 1
        t_end   = i_rtuc + 3 # 4*i_rtuc

        dict_uniton_slow = MyDataFrame(
            df_UNITON_DAC2RTC.loc[t_start: t_end, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()
        dict_UnitStartUp_slow = MyDataFrame(
            df_UNITSTUP_DAC2RTC.loc[t_start: t_end, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()
        dict_UnitShutDn_slow = MyDataFrame(
            df_UNITSTDN_DAC2RTC.loc[t_start: t_end, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()
        dict_DispacthLimitsUpper_slow = MyDataFrame(
            df_dispatch_max_RTC.loc[t_start: t_end, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()
        dict_DispacthLimitsLower_slow = MyDataFrame(
            df_dispatch_min_RTC.loc[t_start: t_end, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()

        # Create RTUC model
        # ins_ha = create_model(
        #     network,
        #     df_busload_ha.loc[t_start: t_end, :],
        #     df_genfor_ha.loc[t_start: t_end, :],
        #     ReserveFactor,
        #     RegulatingReserveFactor,
        #     nI_ha,
        #     dict_UnitOnT0State, 
        #     dict_PowerGeneratedT0,
        #     dict_uniton_ha,
        #     dict_DispacthLimitsUpper_slow
        # )
        ins_ha = create_model(
            network,
            df_busload_RTC.loc[t_start: t_end, :], # Only bus load, first dimension time starts from 1, no total load
            df_genfor_RTC.loc[t_start: t_end, :], # Only generation from nonthermal gens, first dim time starts from 1
            nI_RTC, # Number of intervals in an hour, typically DAUC: 1, RTUC: 4
            dict_UnitOnT0State=dict_UnitOnT0State, # How many time periods the units have been on at T0 from last RTUC model
            dict_PowerGeneratedT0=dict_PowerGeneratedT0, # Initial power generation level at T0 from last RTUC model
            ##############################
            dict_UnitOn=dict_uniton_slow, # Committment statuses of committed units
            dict_UnitStartUp=dict_UnitStartUp_slow, # Startup indicator, keys should be the same as dict_UnitOn
            dict_UnitShutDn=dict_UnitShutDn_slow, # Shutdown indicator, keys should be the same as dict_UnitOn
            dict_DispatchLimitsLower=dict_DispacthLimitsLower_slow, # Only apply for committed units, keys should be the same as dict_UnitOn
            dict_DispatchLimitsUpper=dict_DispacthLimitsUpper_slow, # Only apply for committed units, keys should be the same as dict_UnitOn
        )
        msg = "RTUC Model {} created!".format(i_rtuc)
        print msg
        content += msg
        content += '\n'

        # Solve RTUC model
        try:
            results = optimizer.solve(ins_ha)
        except:
            print 'Cannot solve RTUC model!'
            IP()

        if results.solver.termination_condition == TerminationCondition.infeasible:
            print 'Infeasibility detected in the RTUC model!'
            IP()


if __name__ == '__main__':
    test_dauc('118')

