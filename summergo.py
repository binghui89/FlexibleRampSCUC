import os, sys, platform, datetime, smtplib, multiprocessing, pandas as pd, numpy as np, matplotlib
from time import time
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from postprocessing import store_csvs
from pyomo.environ import *
from helper import import_scenario_data, extract_uniton
from email.mime.text import MIMEText
from numpy import sign
from matplotlib import pyplot as plt
from unit_commitment import GroupDataFrame, MyDataFrame, NewNetwork, create_model
from IPython import embed as IP


def return_unitont0state(instance, t=None):
    '''
    Find number of online/offline time intervals of thermal gens at the end of period t
    Copied from sequential_model.py
    '''

    if not t:
        t = instance.TimePeriods.last()
    elif t not in instance.TimePeriods:
        print "WARNING: NO UNIT_ON CREATED."
        return None
    dict_results = dict()
    for g in instance.ThermalGenerators.iterkeys():
        t_on  = max(0, value(instance.UnitOnT0State[g]))
        t_off = max(0, -value(instance.UnitOnT0State[g]))
        for tau in instance.TimePeriods.iterkeys():
            if instance.TimePeriods.value.index(tau) <= instance.TimePeriods.value.index(t):
                b = value(instance.UnitOn[g, tau])
                b = int(round(b))
                t_on  = b*(t_on + b) # Number of the last consecutive online intervals
                t_off = (1-b)*(t_off + 1 - b) # Number of the last consecutive offline intervals
        dict_results[g] = int(round(sign(t_on)*t_on - sign(t_off)*t_off)) # This is an integer?
    return dict_results

def return_powergenerated_t(instance, t=None):
    '''
    Find power generation levels after period t
    Copied from sequential_model.py
    '''

    if not t:
        t = instance.TimePeriods.last()
    elif t not in instance.TimePeriods:
        print "WARNING: NO POWER_GENERATED CREATED."
        return None
    dict_results = dict()
    for g in instance.AllGenerators.iterkeys():
        v = value(instance.PowerGenerated[g, t])
        dict_results[g] = max(0, v) # Sometimes it returns negative values, dunno why.
    return dict_results

def return_downscaled_initial_condition(instance, nI_L, nI_S):
    '''
    Calculate upper and lower dispatch limits based on the given solved Pyomo instance.
    '''

    nI_SperL = nI_S/nI_L # Number of short (S) intervals per long (L) interval, must be integer

    # Time index of the longer and shorter time scales
    tindex_L = np.array(instance.TimePeriods.value)
    tindex_S = np.arange(
        (tindex_L[0]-1)*nI_SperL+1, # Index of the starting interval of shorter time scale
        tindex_L[-1]*nI_SperL+1,  # Index of the ending interval of shorter time scale, plus 1 because numpy excludes the last number.
        1
    )
    t0_L = tindex_L[0] - 1
    t0_S = tindex_S[0] - 1
    ls_gen_therm = list(instance.ThermalGenerators.value)
    nG_therm = len(ls_gen_therm)

    # Results container
    df_POWER_START_L   = pd.DataFrame(np.nan, index=tindex_L, columns=instance.AllGenerators)
    df_POWER_END_L     = pd.DataFrame(np.nan, index=tindex_L, columns=instance.AllGenerators)
    df_UNITON_L        = pd.DataFrame(np.nan, index=tindex_L, columns=ls_gen_therm)
    df_UNITSTUP_L      = pd.DataFrame(np.nan, index=tindex_L, columns=ls_gen_therm)
    df_UNITSTDN_L      = pd.DataFrame(np.nan, index=tindex_L, columns=ls_gen_therm)
    df_SIGMAUP_L       = pd.DataFrame(np.nan, index=tindex_L, columns=ls_gen_therm)
    df_SIGMADN_L       = pd.DataFrame(np.nan, index=tindex_L, columns=ls_gen_therm)
    df_SIGMAPOWERUP_L  = pd.DataFrame(np.nan, index=tindex_L, columns=ls_gen_therm)
    df_SIGMAPOWERDN_L  = pd.DataFrame(np.nan, index=tindex_L, columns=ls_gen_therm)
    df_REGUP_L         = pd.DataFrame(np.nan, index=tindex_L, columns=ls_gen_therm)
    df_REGDN_L         = pd.DataFrame(np.nan, index=tindex_L, columns=ls_gen_therm)
    df_SPNUP_L         = pd.DataFrame(np.nan, index=tindex_L, columns=ls_gen_therm)
    df_UNITONT0_L      = pd.DataFrame(np.nan, index=[t0_L], columns=ls_gen_therm)
    df_SIGMADNT0_L     = pd.DataFrame(0,      index=[t0_L], columns=ls_gen_therm)
    for g in instance.AllGenerators:
        for t in tindex_L:
            df_POWER_END_L.at[t, g] = value(instance.PowerGenerated[g, t])
            if t == tindex_L[0]:
                df_POWER_START_L.at[t, g] = value(instance.PowerGeneratedT0[g])
            else:
                df_POWER_START_L.at[t, g] = value(instance.PowerGenerated[g, t-1])
    for g in ls_gen_therm:
        df_UNITONT0_L.at[t0_L, g] = value(instance.UnitOnT0[g])
        if g in instance.SigmaDnT0:
            df_SIGMADNT0_L.at[t0_L, g] = value(instance.SigmaDnT0[g])
        else:
            # This is a quick fix, since for committed unit, SigmaDnT0 is not 
            # included by instance, this is related to my biggest headache, 
            # i.e., what if a unit is turned off completedly after the model horizon?
            # Basically, SigmaDnT0 is not included for committed unit because
            # sigma_power_times_dn_initial_rule is not able to handle the case
            # when a unit is turned off beyond the end of model horizon.
            # And it will lead to infeasibility of 
            # thermal_gen_output_limits_shutdown_lower_initial_rule
            # and thermal_gen_output_limits_shutdown_upper_initial_rule.

            # This fix may affect time performance, besides, I think maybe we 
            # should use abs() function to compare float numbers.
            df_SIGMADNT0_L.at[t0_L, g] = value(instance.SigmaDn[g, tindex_L[0]]) *(value(instance.PowerGeneratedT0[g])<=value(instance.MinimumPowerOutput[g]))
        for t in tindex_L:
            df_UNITON_L.at[t, g]       = value(instance.UnitOn[g, t])
            df_UNITSTUP_L.at[t, g]     = value(instance.UnitStartUp[g, t])
            df_UNITSTDN_L.at[t, g]     = value(instance.UnitShutDn[g, t])
            df_SIGMAUP_L.at[t, g]      = value(instance.SigmaUp[g, t])
            df_SIGMADN_L.at[t, g]      = value(instance.SigmaDn[g, t])
            df_SIGMAPOWERUP_L.at[t, g] = value(instance.SigmaPowerTimesUp[g, t])
            df_SIGMAPOWERDN_L.at[t, g] = value(instance.SigmaPowerTimesDn[g, t])
            df_REGUP_L.at[t, g]        = value(instance.RegulatingReserveUpAvailable[g, t])
            df_REGDN_L.at[t, g]        = value(instance.RegulatingReserveDnAvailable[g, t])
            df_SPNUP_L.at[t, g]        = value(instance.SpinningReserveUpAvailable[g, t])

    # Calculate unit start-up/shut-down binary indicator
    # df_unitonaug = pd.concat(
    #     [df_UNITONT0_L[df_UNITON_L.columns], df_UNITON_L[df_UNITON_L.columns]], 
    #     axis=0
    # ) # Augmented df_uniton
    # df_UNITSTUP_L = pd.DataFrame(
    #     np.maximum(df_unitonaug.loc[1:, :].values - df_unitonaug.iloc[0:-1, :].values, 0),
    #     index=df_UNITON_L.index,
    #     columns=df_unitonaug.columns,
    # ).astype(int)
    # df_UNITSTDN_L = pd.DataFrame(
    #     np.maximum(df_unitonaug.iloc[0:-1, :].values - df_unitonaug.loc[1:, :].values, 0),
    #     index=df_UNITON_L.index,
    #     columns=df_unitonaug.columns,
    # ).astype(int)

    # Note: np array's index starts from 0
    ar_UNITSTUP_L2S = np.zeros((tindex_S.size, nG_therm), dtype=int)
    ar_UNITSTDN_L2S = np.zeros((tindex_S.size, nG_therm), dtype=int)
    df_UNITSTUP_L2S = pd.DataFrame(
        ar_UNITSTUP_L2S,
        index=tindex_S, 
        columns=ls_gen_therm,
    )
    df_UNITSTDN_L2S = pd.DataFrame(
        ar_UNITSTDN_L2S,
        index=tindex_S, 
        columns=ls_gen_therm,
    )
    df_UNITSTUP_L2S.loc[(tindex_L-1)*nI_SperL+1,:] = df_UNITSTUP_L.values # ar_UNITSTUP_L2S[(tindex_L-1)*nI_SperL, :] = df_UNITSTUP_L.values
    df_UNITSTDN_L2S.loc[tindex_L*nI_SperL, :] = df_UNITSTDN_L.values # ar_UNITSTDN_L2S[tindex_L*nI_SperL-1, :]   = df_UNITSTDN_L.values

    ar_UNITON_L2S = (df_UNITON_L+df_UNITSTDN_L).values.astype(int).repeat(nI_SperL, axis=0) - df_UNITSTDN_L2S.values.astype(int) # ar_UNITON_L2S = (df_UNITON_L+df_UNITSTDN_L).values.astype(int).repeat(nI_SperL, axis=0) - ar_UNITSTDN_L2S

    ar_SIGMAUP_L2S = df_SIGMAUP_L.astype(int).values.repeat(nI_SperL, axis=0)
    ar_SIGMADN_L2S = np.concatenate(
        [df_SIGMADNT0_L.values, df_SIGMADN_L.values], 
        axis=0
    ).repeat(nI_SperL, axis=0)[1:tindex_S.size+1, :]

    # Binary indicator of a unit is up, starting-up, or  shuttind-down. 
    # A unit is up if power >= Pmin, this means nominal ramp rate applies in this interval.
    # A unit is starting up if power < Pmin and it is ramping up at its start-up ramp rate.
    # A unit is shutting down if power < Pmin and it is ramping down at its shut-down ramp rate.
    # Note the differences: 
    # ar_is_up_L2S           <--> ar_UNITON_L2S
    # ar_is_startingup_L2S   <--> ar_SIGMAUP_L2S
    # ar_is_shuttingdown_L2S <--> ar_SIGMADN_L2S
    # A unit's status can only be one of the following exclusively:
    # (1) up, (2) starting-up, (3) shutting-down or (4) offline, so the 
    # following equation always holds: 
    # ar_is_up_L2S + ar_is_startingup_L2S + ar_is_shuttingdown_L2S <= 1
    ar_is_startingup_L2S   = ar_SIGMAUP_L2S
    ar_is_shuttingdown_L2S = np.concatenate([df_SIGMADNT0_L.values, ar_SIGMADN_L2S], axis=0)[0:-1]
    ar_is_up_L2S = np.maximum(
        0,
        ar_UNITON_L2S + df_UNITSTDN_L2S.values # ar_UNITSTDN_L2S 
        - ar_is_startingup_L2S 
        - ar_is_shuttingdown_L2S
    )

    # All ramp rate units are in MW/interval
    ar_ramp_dn_S = np.array([value(instance.RampDownLimitPerHour[g]) for g in ls_gen_therm])/nI_S
    ar_pmin        = np.array([value(instance.MinimumPowerOutput[g]) for g in ls_gen_therm])
    ar_pmax        = np.array([value(instance.MaximumPowerOutput[g]) for g in ls_gen_therm])
    ar_p_aug = np.concatenate(
        [
            df_POWER_START_L[ls_gen_therm].values[0,:].reshape(1, nG_therm),
            df_POWER_END_L[ls_gen_therm].values
        ],
        axis=0
    ) # Arrary of power generation from DAUC, including initial power level P0 in the first row

    # Power generation levels at the end of all time intervals are interpolated 
    # from longer time scale into shorter time scale.
    # We need to normlize the time index of both the longer and shorter time 
    # scales such that both start from 0 (time index 0, or the initial interval) 
    # and end at 1 (the last interval).
    ar_POWER_END_L = np.empty((tindex_S.size, nG_therm))
    tindex_S_normalized = (tindex_S.astype(float) - t0_S)/(tindex_S[-1] - t0_S)
    tindex_L_full_normalized = (np.concatenate([[t0_L], tindex_L]).astype(float) - t0_L)/(tindex_L[-1] - t0_L)
    for i in np.arange(ar_p_aug.shape[1]):
        ar_POWER_END_L[:, i] = np.interp(
            tindex_S_normalized,
            tindex_L_full_normalized, # This is a sequence from 0 to 1.
            ar_p_aug[:, i]
        )

    # Upper and lower dispatch limits
    ar_dispatch_max_S = np.empty((df_UNITSTUP_L.shape[0]*nI_SperL, df_UNITSTUP_L.shape[1]))
    ar_dispatch_min_S = np.empty((df_UNITSTUP_L.shape[0]*nI_SperL, df_UNITSTUP_L.shape[1]))

    for i in np.arange(df_UNITSTUP_L.shape[0]*nI_SperL-1, -1, -1):
        if i == df_UNITSTUP_L.shape[0]*nI_SperL-1:
            ar_dispatch_max_S[i, :] = (
                ar_pmax*ar_is_up_L2S[-1, :] 
                + (
                    ar_is_startingup_L2S[-1, :] 
                    + ar_is_shuttingdown_L2S[-1, :]
                )*ar_POWER_END_L[-1, :]
            )
            ar_dispatch_min_S[i, :] = (
                ar_pmin*ar_is_up_L2S[-1, :] 
                + (
                    ar_is_startingup_L2S[-1, :] 
                    + ar_is_shuttingdown_L2S[-1, :]
                )*ar_POWER_END_L[-1, :]
            )
        else:
            i_next = i+1
            tmp = ar_is_up_L2S[i, :]*ar_is_up_L2S[i_next, :]
            ar_dispatch_max_S[i, :] = np.minimum(
                ar_pmax, 
                tmp*(ar_dispatch_max_S[i_next, :] + ar_ramp_dn_S) +
                (1-tmp)*ar_POWER_END_L[i, :]
            )
            ar_dispatch_min_S[i, :] = np.minimum(
                ar_pmin, 
                ar_dispatch_max_S[i, :]
            )

    df_UNITON_L2S = pd.DataFrame(
        ar_UNITON_L2S,
        index=tindex_S, 
        columns=ls_gen_therm,
    )
    # df_UNITSTUP_L2S = pd.DataFrame(
    #     ar_UNITSTUP_L2S,
    #     index=tindex_S, 
    #     columns=ls_gen_therm,
    # )
    # df_UNITSTDN_L2S = pd.DataFrame(
    #     ar_UNITSTDN_L2S,
    #     index=tindex_S, 
    #     columns=ls_gen_therm,
    # )
    df_dispatch_max_L2S = pd.DataFrame(
        ar_dispatch_max_S,
        index=tindex_S, 
        columns=ls_gen_therm,
    )
    df_dispatch_min_L2S = pd.DataFrame(
        ar_dispatch_min_S,
        index=tindex_S, 
        columns=ls_gen_therm,
    )
    df_SIGMAUP_L2S = pd.DataFrame(
        ar_SIGMAUP_L2S,
        index=tindex_S, 
        columns=ls_gen_therm,
    )
    df_SIGMADN_L2S = pd.DataFrame(
        ar_SIGMADN_L2S,
        index=tindex_S, 
        columns=ls_gen_therm,
    )

    df_result = GroupDataFrame()
    df_result.df_uniton = df_UNITON_L2S
    df_result.df_unitstup = df_UNITSTUP_L2S
    df_result.df_unitstdn = df_UNITSTDN_L2S
    df_result.df_dispatch_min = df_dispatch_min_L2S
    df_result.df_dispatch_max = df_dispatch_max_L2S
    df_result.df_sigmaup = df_SIGMAUP_L2S
    df_result.df_sigmadn = df_SIGMADN_L2S
    return df_result

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

def summergo_uced(casename):
    t0 = time()
    content = ''

    nI_DAC = 1  # Number of DRUC intervals in an hour, DRUC runs per day, covers 24 1-h intervals
    nI_RTC = 12 # Number of HRUC intervals in an hour, HRUC runs per hour, covers 12 5-min intervals
    nI_RTD = 12 # Number of RTED intervals in an hour, RTED runs per 5-min, covers 1 5-min intervals
    nI_AGC = 3600/6 # Number of AGC intervals in an hour
    nI_RTCperDAC = nI_RTC/nI_DAC
    nI_RTDperRTC = nI_RTD/nI_RTC
    nI_AGCperRTD = nI_AGC/nI_RTD

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

        # In the SUMMER-GO 118 bus system, we need to run 5-min RTUC runs so use ED data temporarily
        csv_busload_RTC = csv_busload_RTD
        csv_genfor_RTC  = csv_genfor_RTD

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
        df_busload_AGC = pd.read_csv(csv_busload_AGC, index_col='Slot')
        df_genfor_AGC  = pd.read_csv(csv_genfor_AGC, index_col='Slot')
        if 'LOAD' in df_busload_AGC.columns.difference(['LOAD']):
            df_busload_AGC = df_busload_AGC.loc[:, df_busload_AGC.columns.difference(['LOAD'])]
        df_busload_AGC = MyDataFrame(df_busload_AGC.loc[:, df_busload_AGC.columns.difference(['LOAD'])])
        df_busload_AGC.fillna(0, inplace=True)
        df_busload_full_AGC = pd.DataFrame(0, index=df_busload_AGC.index, columns=network.set_bus) # For ACE calculation, include all buses
        df_busload_full_AGC.loc[:, df_busload_AGC.columns] = df_busload_AGC
        df_genfor_AGC  = MyDataFrame(df_genfor_AGC)
        df_genfor_AGC.fillna(0, inplace=True)

        df_agc_param = pd.DataFrame(0, index=network.dict_set_gens['ALL'], columns=['ACE_TARGET'])
        df_agc_param.loc[:, 'DEAD_BAND'] = dfs.df_gen.loc[:, 'DEAD_BAND']
        df_agc_param.loc[:, 'AGC_MODE']  = dfs.df_gen.loc[:, 'AGC_MODE']


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

    # RTED results
    df_POWER_RTD_BINDING = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])
    df_POWER_RTD_ADVISRY = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])
    df_REGUP_RTD_BINDING = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])
    df_REGDN_RTD_BINDING = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])
    df_SPNUP_RTD_BINDING = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])

    # AGC results
    df_ACTUAL_GENERATION = MyDataFrame(index=df_genfor_AGC.index, columns=network.dict_set_gens['ALL'], dtype='float')
    df_AGC_SCHEDULE      = MyDataFrame(index=df_genfor_AGC.index, columns=network.dict_set_gens['ALL'], dtype='float')
    df_ACE_TARGET        = MyDataFrame(index=df_genfor_AGC.index, columns=network.dict_set_gens['ALL'], dtype='float')
    df_AGC_MOVE          = MyDataFrame(index=df_genfor_AGC.index, columns=network.dict_set_gens['ALL'], dtype='float')
    df_ACE               = MyDataFrame(index=df_genfor_AGC.index, columns=['RAW', 'CPS2', 'SACE', 'ABS', 'INT'])

    ls_rtuc = list()
    ls_rted = list()

    # Start DAUC
    ############################################################################
    model = create_model(
        network,
        df_busload,
        df_genfor_nonthermal,
        nI=nI_DAC,
        dict_UnitOnT0State=dict_UnitOnT0State,
        dict_PowerGeneratedT0=dict_PowerGeneratedT0,
        flow_limits=False,
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
    # IP()

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

    dfs_DAC2RTC = return_downscaled_initial_condition(instance, nI_DAC, nI_RTC)
    df_UNITON_DAC2RTC       = dfs_DAC2RTC.df_uniton
    df_UNITSTUP_DAC2RTC     = dfs_DAC2RTC.df_unitstup
    df_UNITSTDN_DAC2RTC     = dfs_DAC2RTC.df_unitstdn
    df_dispatch_min_DAC2RTC = dfs_DAC2RTC.df_dispatch_min
    df_dispatch_max_DAC2RTC = dfs_DAC2RTC.df_dispatch_max
    df_SIGMAUP_DAC2RTC      = dfs_DAC2RTC.df_sigmaup
    df_SIGMADN_DAC2RTC      = dfs_DAC2RTC.df_sigmadn

    # Convert DAC initial conditions into RTC initial conditions
    dict_UnitOnT0State_RTC = (pd.Series(dict_UnitOnT0State)*nI_RTCperDAC).to_dict()
    dict_PowerGeneratedT0_RTC = dict_PowerGeneratedT0

    # IP()
    for i_rtuc in np.arange(1, 25):
        t_s_RTC = (i_rtuc-1)*12+1 # 4*(i_rtuc-1) + 1
        t_e_RTC = i_rtuc*12 # 4*i_rtuc

        dict_uniton_slow = MyDataFrame(
            df_UNITON_DAC2RTC.loc[t_s_RTC: t_e_RTC, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()
        dict_UnitStartUp_slow = MyDataFrame(
            df_UNITSTUP_DAC2RTC.loc[t_s_RTC: t_e_RTC, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()
        dict_UnitShutDn_slow = MyDataFrame(
            df_UNITSTDN_DAC2RTC.loc[t_s_RTC: t_e_RTC, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()
        dict_DispacthLimitsUpper_slow = MyDataFrame(
            df_dispatch_max_DAC2RTC.loc[t_s_RTC: t_e_RTC, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()
        dict_DispacthLimitsLower_slow = MyDataFrame(
            df_dispatch_min_DAC2RTC.loc[t_s_RTC: t_e_RTC, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()
        dict_SigmaUp_slow = MyDataFrame(
            df_SIGMAUP_DAC2RTC.loc[t_s_RTC: t_e_RTC, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()
        dict_SigmaDn_slow = MyDataFrame(
            df_SIGMADN_DAC2RTC.loc[t_s_RTC: t_e_RTC, network.dict_set_gens['THERMAL_slow']].T
        ).to_dict_2d()

        # Create RTUC model
        ins_RTC = create_model(
            network,
            df_busload_RTC.loc[t_s_RTC: t_e_RTC, :], # Only bus load, first dimension time starts from 1, no total load
            df_genfor_RTC.loc[t_s_RTC: t_e_RTC, :], # Only generation from nonthermal gens, first dim time starts from 1
            nI_RTC, # Number of intervals in an hour, typically DAUC: 1, RTUC: 4
            dict_UnitOnT0State=dict_UnitOnT0State_RTC, # How many time periods the units have been on at T0 from last RTUC model
            dict_PowerGeneratedT0=dict_PowerGeneratedT0_RTC, # Initial power generation level at T0 from last RTUC model
            ##############################
            dict_UnitOn=dict_uniton_slow, # Committment statuses of committed units
            dict_UnitStartUp=dict_UnitStartUp_slow, # Startup indicator, keys should be the same as dict_UnitOn
            dict_UnitShutDn=dict_UnitShutDn_slow, # Shutdown indicator, keys should be the same as dict_UnitOn
            dict_DispatchLimitsLower=dict_DispacthLimitsLower_slow, # Only apply for committed units, keys should be the same as dict_UnitOn
            dict_DispatchLimitsUpper=dict_DispacthLimitsUpper_slow, # Only apply for committed units, keys should be the same as dict_UnitOn
            dict_SigmaUp=dict_SigmaUp_slow,
            dict_SigmaDn=dict_SigmaDn_slow,
            flow_limits=False,
        )
        msg = "RTUC Model {} created!".format(i_rtuc)
        print msg
        content += msg
        content += '\n'

        # Solve RTUC model
        try:
            results_RTC = optimizer.solve(ins_RTC)
        except:
            print 'Cannot solve RTUC model!'
            IP()

        if results_RTC.solver.termination_condition == TerminationCondition.infeasible:
            print 'Infeasibility detected in the RTUC model!'
            IP()
        elif value(ins_RTC.SlackPenalty) > 1E-3:
            print 'Infeasibility in the RTUC model, penalty: {}'.format(value(ins_RTC.SlackPenalty))
            ls_slackvar_name = [
                'Slack_startup_lower',   'Slack_startup_upper', 
                'Slack_shutdown_lower',  'Slack_shutdown_upper', 
                'Slack_overlap_startup', 'Slack_overlap_shutdown', 
                'Slack_rampup',          'Slack_rampdn',
            ]
            for slackvar_name in ls_slackvar_name:
                slackvar = getattr(ins_RTC, slackvar_name)
                for k in slackvar.iterkeys():
                    if value(slackvar[k]) > 0:
                        print slackvar_name, k, value(slackvar[k])
            IP()
        msg = (
            'RTUC Model {} '
            'solved at: {:>.2f} s, '
            'objective: {:>.2f}, '
            # 'penalty: {:s}'.format(
            'penalty: {:>.2f}'.format(
                i_rtuc, 
                time() - t0,
                value(ins_RTC.TotalCostObjective),
                # 'N/A',
                value(ins_RTC.SlackPenalty),
            )
        )
        print(msg)
        # IP()
        ins_RTC.solutions.load_from(results_RTC)
        ls_rtuc.append(ins_RTC)

        # If there is no RTED run.
        # dict_UnitOnT0State_RTC = return_unitont0state(ins_RTC, ins_RTC.TimePeriods.last())
        # dict_PowerGeneratedT0_RTC = return_powergenerated_t(ins_RTC, ins_RTC.TimePeriods.last())

        dfs_RTC2RTD = return_downscaled_initial_condition(ins_RTC, nI_RTC, nI_RTD)
        df_UNITON_RTC2RTD       = dfs_RTC2RTD.df_uniton
        df_UNITSTUP_RTC2RTD     = dfs_RTC2RTD.df_unitstup
        df_UNITSTDN_RTC2RTD     = dfs_RTC2RTD.df_unitstdn
        df_dispatch_min_RTC2RTD = dfs_RTC2RTD.df_dispatch_min
        df_dispatch_max_RTC2RTD = dfs_RTC2RTD.df_dispatch_max
        df_SIGMAUP_RTC2RTD      = dfs_RTC2RTD.df_sigmaup
        df_SIGMADN_RTC2RTD      = dfs_RTC2RTD.df_sigmadn

        ls_t_ed = np.arange((i_rtuc-1)*12+1, i_rtuc*12 +1)

        # Convert RTC initial conditions into RTD initial conditions
        dict_UnitOnT0State_RTD = (pd.Series(dict_UnitOnT0State_RTC)*nI_RTDperRTC).to_dict()
        dict_PowerGeneratedT0_RTD = dict_PowerGeneratedT0_RTC

        for t_s_RTD in ls_t_ed:
            t_e_RTD = t_s_RTD  # Total 1 ED intervals in SUMMER-GO

            dict_uniton_all = MyDataFrame(
                df_UNITON_RTC2RTD.loc[t_s_RTD: t_e_RTD, network.dict_set_gens['THERMAL']].T
            ).to_dict_2d()
            dict_UnitStartUp_all = MyDataFrame(
                df_UNITSTUP_RTC2RTD.loc[t_s_RTD: t_e_RTD, network.dict_set_gens['THERMAL']].T
            ).to_dict_2d()
            dict_UnitShutDn_all = MyDataFrame(
                df_UNITSTDN_RTC2RTD.loc[t_s_RTD: t_e_RTD, network.dict_set_gens['THERMAL']].T
            ).to_dict_2d()
            dict_DispacthLimitsUpper_all = MyDataFrame(
                df_dispatch_max_RTC2RTD.loc[t_s_RTD: t_e_RTD, network.dict_set_gens['THERMAL']].T
            ).to_dict_2d()
            dict_DispacthLimitsLower_all = MyDataFrame(
                df_dispatch_min_RTC2RTD.loc[t_s_RTD: t_e_RTD, network.dict_set_gens['THERMAL']].T
            ).to_dict_2d()
            dict_SigmaUp_all = MyDataFrame(
                df_SIGMAUP_RTC2RTD.loc[t_s_RTD: t_e_RTD, network.dict_set_gens['THERMAL']].T
            ).to_dict_2d()
            dict_SigmaDn_all = MyDataFrame(
                df_SIGMADN_RTC2RTD.loc[t_s_RTD: t_e_RTD, network.dict_set_gens['THERMAL']].T
            ).to_dict_2d()

            # Create RTED model
            ins_RTD = create_model(
                network,
                df_busload_RTD.loc[t_s_RTD: t_e_RTD, :], # Only bus load, first dimension time starts from 1, no total load
                df_genfor_RTD.loc[t_s_RTD: t_e_RTD, :], # Only generation from nonthermal gens, first dim time starts from 1
                nI_RTD, # Number of intervals in an hour, typically DAUC: 1, RTUC: 4
                dict_UnitOnT0State=dict_UnitOnT0State_RTD, # How many time periods the units have been on at T0 from last RTUC model
                dict_PowerGeneratedT0=dict_PowerGeneratedT0_RTD, # Initial power generation level at T0 from last RTUC model
                ##############################
                dict_UnitOn=dict_uniton_all, # Committment statuses of committed units
                dict_UnitStartUp=dict_UnitStartUp_all, # Startup indicator, keys should be the same as dict_UnitOn
                dict_UnitShutDn=dict_UnitShutDn_all, # Shutdown indicator, keys should be the same as dict_UnitOn
                dict_DispatchLimitsLower=dict_DispacthLimitsLower_all, # Only apply for committed units, keys should be the same as dict_UnitOn
                dict_DispatchLimitsUpper=dict_DispacthLimitsUpper_all, # Only apply for committed units, keys should be the same as dict_UnitOn
                dict_SigmaUp=dict_SigmaUp_all,
                dict_SigmaDn=dict_SigmaDn_all,
                flow_limits=False,
            )
            # msg = "RTED Model {} created!".format(t_s_RTD)
            # print msg
            # content += msg
            # content += '\n'

            # Solve RTUC model
            try:
                results_RTD = optimizer.solve(ins_RTD)
            except:
                print 'Cannot solve RTED model!'
                IP()

            if results_RTD.solver.termination_condition == TerminationCondition.infeasible:
                print 'Infeasibility detected in the RTED model!'
                IP()
            elif value(ins_RTD.SlackPenalty) > 1E-3:
                print 'Infeasibility in the RTED model, penalty: {}'.format(value(ins_RTD.SlackPenalty))
                ls_slackvar_name = [
                    'Slack_startup_lower',   'Slack_startup_upper', 
                    'Slack_shutdown_lower',  'Slack_shutdown_upper', 
                    'Slack_overlap_startup', 'Slack_overlap_shutdown', 
                    'Slack_rampup',          'Slack_rampdn',
                ]
                for slackvar_name in ls_slackvar_name:
                    slackvar = getattr(ins_RTD, slackvar_name)
                    for k in slackvar.iterkeys():
                        if value(slackvar[k]) > 0:
                            print attr, k, value(slackvar[k])
                IP()

            msg = (
                '    '
                'RTED Model {} '
                'solved at: {:>.2f} s, '
                'objective: {:>.2f}, '
                'penalty: {:>.2f}'.format(
                    t_s_RTD, 
                    time() - t0,
                    value(ins_RTD.TotalCostObjective),
                    value(ins_RTD.SlackPenalty),
                )
            )
            print msg

            # Extract initial parameters from the binding interval for the next RTED run
            dict_UnitOnT0State_RTD = return_unitont0state(
                ins_RTD, ins_RTD.TimePeriods.first()
            )
            dict_PowerGeneratedT0_RTD = return_powergenerated_t(
                ins_RTD, ins_RTD.TimePeriods.first()
            )
            # IP()
            # dict_PowerGeneratedT0_RTD = df_AGC_SCHEDULE.loc[t_AGC, network.dict_set_gens['THERMAL']].to_dict()

            ins_RTD.solutions.load_from(results_RTD)
            ls_rted.append(ins_RTD)


        # IP()
        # Extract initial parameters from the binding interval of the last ED run for the next RTUC run
        # dict_UnitOnT0State_RTC = (pd.Series(dict_UnitOnT0State_RTD)/nI_RTDperRTC).to_dict()
        dict_UnitOnT0State_RTC = return_unitont0state(ins_RTC, ins_RTC.TimePeriods.last()) # Because in SUMMER-GO the first time interval of a RTUC run is the last time interval of the previous RTUC run.
        dict_PowerGeneratedT0_RTC = dict_PowerGeneratedT0_RTD
    
    # IP()
    df_power_start = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])
    df_power_end   = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])
    df_uniton      = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])
    df_regup       = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])
    df_regdn       = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])
    df_spnup       = MyDataFrame(0.0, index=df_genfor_RTD.index, columns=network.dict_set_gens['ALL'])
    sr_curtailment = pd.Series(0, index=df_genfor_RTD.index)

    for i in range(len(ls_rted)):
        ins_RTD = ls_rted[i]
        t_s_RTD = i + 1
        sr_curtailment.at[t_s_RTD] = value(ins_RTD.Curtailment[t_s_RTD])
        for g in ins_RTD.AllGenerators:
            df_power_end.at[t_s_RTD, g]   = value(ins_RTD.PowerGenerated[g, t_s_RTD])
            df_power_start.at[t_s_RTD, g] = value(ins_RTD.PowerGeneratedT0[g])

        for g in ins_RTD.ThermalGenerators:
            df_uniton.at[t_s_RTD, g] = value(ins_RTD.UnitOn[g, t_s_RTD])
            df_regup.at[t_s_RTD, g]  = value(ins_RTD.RegulatingReserveUpAvailable[g, t_s_RTD])
            df_regdn.at[t_s_RTD, g]  = value(ins_RTD.RegulatingReserveDnAvailable[g, t_s_RTD])
            df_spnup.at[t_s_RTD, g]  = value(ins_RTD.SpinningReserveUpAvailable[g, t_s_RTD])
    df_power_mean = (df_power_start + df_power_end)/2

    # Compare load and supply
    ax1 = plt.subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax1.step(
        df_power_mean.index,
        df_power_mean.values.sum(axis=1),
        where='post',
        color='b',
        label='Generation',
    )
    ax1.step(
        df_busload_RTD.index,
        df_busload_RTD.values.sum(axis=1),
        where='post',
        color='k',
        label='Total load',
    )
    ax1.step(
        sr_curtailment.index,
        sr_curtailment,
        color='g',
        where='post',
        label='Shedded load'
    )
    ax1.fill_between(
        df_power_mean.index,
        df_power_mean.values.sum(axis=1)-df_regdn.values.sum(axis=1),
        df_power_mean.values.sum(axis=1),
        color='b',
        step='post',
        alpha=0.2,
    )
    ax1.fill_between(
        df_power_mean.index,
        df_power_mean.values.sum(axis=1),
        df_power_mean.values.sum(axis=1)+df_regup.values.sum(axis=1),
        color='r',
        step='post',
        alpha=0.2,
    )
    ax1.set_ylabel('MW')
    ax1.legend()
    ax2.step(
        df_uniton.index,
        df_uniton.values.sum(axis=1),
        where='post',
        color='r',
        label='Committed units'
    )
    ax2.set_ylabel('# of units')
    ax2.legend()
    plt.title('RTED results')
    plt.show()

if __name__ == '__main__':
    # test_dauc('118')
    summergo_uced('118')
