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

def send_mail(to_list, sub, content, mail_pass):  
    # For help go to the following link:
    # https://www.digitalocean.com/community/questions/unable-to-send-mail-through-smtp-gmail-com
    mail_host="smtp.gmail.com:587"  # SMTP server
    mail_user="reagan.fruit"    # Username
    mail_postfix="gmail.com"

    me="Binghui Li"+"<"+mail_user+"@"+mail_postfix+">"  
    msg = MIMEText(content,_subtype='plain',_charset='gb2312')  
    msg['Subject'] = sub  
    msg['From'] = me  
    msg['To'] = ";".join(to_list)  
    try:  
        server = smtplib.SMTP(mail_host)  
        server.ehlo()
        server.starttls()
        server.login(mail_user,mail_pass)  
        server.sendmail(me, to_list, msg.as_string())  
        server.close()  
        return True  
    except Exception, e:  
        print str(e)  
        return False  

def return_unitont0state(instance, t=None):
    # Find number of online/offline time intervals of thermal gens after period t
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
                t_on  = b*(t_on + b) # Number of the last consecutive online hours
                t_off = (1-b)*(t_off + 1 - b) # Number of the last consecutive offline hours
        dict_results[g] = int(round(sign(t_on)*t_on - sign(t_off)*t_off)) # This is an integer?
    return dict_results

def return_powergenerated_t(instance, t=None):
    # Find power generation levels after period t
    if not t:
        t = instance.TimePeriods.last()
    elif t not in instance.TimePeriods:
        print "WARNING: NO POWER_GENERATED CREATED."
        return None
    dict_results = dict()
    for g in instance.ThermalGenerators.iterkeys():
        v = value(instance.PowerGenerated[g, t])
        dict_results[g] = max(0, v) # Sometimes it returns negative values, dunno why.
    return dict_results

def return_dispatchupperlimits_from_uniton(df_uniton, ar_pmax, ar_ramp, ar_rsd):
    '''
    Calculate the upper dispatch limits based solely on commitment statuses, 
    max capacity, ramp rate and shut-down ramp rates.
    nT: Number of intervals, nG: number of generators
    df_uniton: nT by nG dataframe of commitment statuses
    ar_pmax: 1-dim np array of max capacity, unit: MW.
    ar_ramp: 1-dim np array of ramp rates, unit: MW/interval.
    ar_rsd:  1_dim np array of shut-down ramp rates: unit: MW/interval, 
    Returns a dataframe of the size nT by nG
    '''

    mat_uniton = df_uniton.values
    mat_dispatchlimits = np.zeros(mat_uniton.shape)
    for i in range(mat_dispatchlimits.shape[0]-1, -1, -1):
        if i == mat_dispatchlimits.shape[0]-1: # The last row
            mat_dispatchlimits[i, :] = mat_uniton[i, :]*ar_pmax
        else:
            mat_dispatchlimits[i, :] = mat_uniton[i, :]*(
                (mat_uniton[i, :]-mat_uniton[i+1, :])*ar_rsd 
                + 
                mat_uniton[i+1,:]*(mat_dispatchlimits[i+1, :]+ar_ramp)
            )

    mat_dispatchlimits = np.minimum(
        mat_dispatchlimits, 
        np.matlib.repmat(
            ar_pmax,
            mat_uniton.shape[0], 
            1
        )
    )

    return pd.DataFrame(
        mat_dispatchlimits, 
        index=df_uniton.index,
        columns=df_uniton.columns,
    )

def examine_load():
    df_load_da = pd.read_csv('/home/bxl180002/git/FlexibleRampSCUC/118bus/loads.csv', index_col=0)
    df_load_ha = pd.read_csv('/home/bxl180002/git/FlexibleRampSCUC/118bus/ha_loads.csv', index_col=['Slot'])

    df_load = pd.DataFrame(index=df_load_ha.index)
    for i in df_load.index:
        h = (i-1)/4+1
        df_load.loc[i, 'DA'] = df_load_da.loc[h, df_load_da.columns.difference(['LOAD'])].sum()
        df_load.loc[i, 'HA'] = df_load_ha.loc[i, df_load_ha.columns.difference(['LOAD'])].sum()

    IP()

def sequential_run_old():
    from SCUC_RampConstraint_3 import create_model, da_input, Network, MyDataFrame
    from SCED_RampConstraint_3.py import *
    t0 = time()
    content = ''

    nI_da = 1 # Number of DAUC intervals in an hour
    nI_ha = 4 # Number of RTUC intervals in an hour
    nI_ed = 12 # Number of RTED intervals in an hour
    nI_agc = 3600/6 # Number of AGC intervals in an hour

    # Time table, should we add AGC time as well?
    df_timesequence = pd.DataFrame(columns=['tH', 'tQ', 't5'], dtype='int')
    for i in range(1, 25):
        for j in range(1, 5):
            for k in range(1,4):
                index = (i-1)*12+(j-1)*3 + k
                df_timesequence.loc[index, 'tH'] = i
                df_timesequence.loc[index, 'tQ'] = (i-1)*4 + j
                df_timesequence.loc[index, 't5'] = index
    df_timesequence = df_timesequence.astype('int')

    # TX 2000 bus system
    # csv_bus               = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/bus.csv'
    # csv_branch            = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/branch.csv'
    # csv_ptdf              = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ptdf.csv'
    # csv_gen               = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/generator_data_plexos_withRT.csv'
    # csv_marginalcost      = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/marginalcost.csv'
    # csv_blockmarginalcost = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/blockmarginalcost.csv'
    # csv_blockoutputlimit  = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/blockoutputlimit.csv'
    # csv_busload           = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/loads.csv'
    # csv_genfor            = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/generator.csv'
    # csv_busload_ha        = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ha_load.csv'
    # csv_genfor_ha         = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ha_generator.csv'
    # csv_busload_ed        = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ed_load.csv'
    # csv_genfor_ed         = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/ed_generator.csv'

    # 118 bus system
    csv_bus               = '/home/bxl180002/git/FlexibleRampSCUC/118bus/bus.csv'
    csv_branch            = '/home/bxl180002/git/FlexibleRampSCUC/118bus/branch.csv'
    csv_ptdf              = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ptdf.csv'
    csv_gen               = '/home/bxl180002/git/FlexibleRampSCUC/118bus/generator_data_plexos_withRT.csv'
    csv_marginalcost      = '/home/bxl180002/git/FlexibleRampSCUC/118bus/marginalcost.csv'
    csv_blockmarginalcost = '/home/bxl180002/git/FlexibleRampSCUC/118bus/blockmarginalcost.csv'
    csv_blockoutputlimit  = '/home/bxl180002/git/FlexibleRampSCUC/118bus/blockoutputlimit.csv'
    csv_busload           = '/home/bxl180002/git/FlexibleRampSCUC/118bus/loads.csv'
    csv_genfor            = '/home/bxl180002/git/FlexibleRampSCUC/118bus/generator.csv'
    csv_busload_ha        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ha_loads.csv'
    csv_genfor_ha         = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ha_generator.csv'
    csv_busload_ed        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ed_loads.csv'
    csv_genfor_ed         = '/home/bxl180002/git/FlexibleRampSCUC/118bus/ed_generator.csv'
    csv_busload_agc       = '/home/bxl180002/git/FlexibleRampSCUC/118bus/agc_loads.csv'
    csv_genfor_agc        = '/home/bxl180002/git/FlexibleRampSCUC/118bus/agc_generator.csv'

    # Build network object, will moved to case specific initiation functions, 
    # we need to make sure that the network object contains exactly what we need
    # in the model, no more, no less
    network = Network(csv_bus, csv_branch, csv_ptdf, csv_gen, csv_marginalcost, csv_blockmarginalcost, csv_blockoutputlimit)
    network.df_bus['VOLL'] = 9000
    network.baseMVA = 100

    # This is to fix a bug in the 118 bus system
    if 'Hydro 31' in network.df_gen.index:
        network.df_gen.drop('Hydro 31', inplace=True)

    # Build network object, will moved to case specific initiation functions
    network.dict_gens = dict()
    network.dict_gens['ALL'] = network.df_gen.index.tolist()
    for t in network.df_gen['GEN_TYPE'].unique():
        network.dict_gens[t] = network.df_gen.loc[network.df_gen['GEN_TYPE']==t, 'GEN_TYPE'].index.tolist()
    network.dict_gens['Non thermal'] = network.df_gen.index.difference(
        network.df_gen.loc[network.df_gen['GEN_TYPE']=='Thermal', 'GEN_TYPE'].index
    ).tolist()
    network.dict_gens['Thermal_slow'] = network.df_gen[
        (network.df_gen['MINIMUM_UP_TIME'] > 1) 
        & 
        (network.df_gen['GEN_TYPE']=='Thermal')
    ].index
    network.dict_gens['Thermal_fast'] = network.df_gen[
        (network.df_gen['MINIMUM_UP_TIME'] <= 1) 
        & 
        (network.df_gen['GEN_TYPE']=='Thermal')
    ].index
    # Bus-generator matrix
    network.ls_bus = network.df_ptdf.columns.tolist()
    network.mat_busgen=np.zeros([len(network.ls_bus), len(network.dict_gens['ALL'])])
    dict_genbus = network.df_gen['GEN_BUS'].to_dict()
    for g, b in dict_genbus.iteritems():
        i = network.ls_bus.index(b)
        j = network.dict_gens['ALL'].index(g)
        network.mat_busgen[i, j] = 1
    df_tmp = pd.DataFrame(0, index=network.ls_bus, columns=network.dict_gens['ALL'])
    for g, b in dict_genbus.iteritems():
        df_tmp.loc[b, g] = 1 # This provides an alternative choice other than network.mat_busgen, should compare their performance later
    # AGC related parameters, need update
    network.df_gen['AGC_MODE'] = 'RAW' # Possible options: NA, RAW, SMOOTH, CPS2
    network.df_gen['DEAD_BAND'] = 5 # MW, 5 MW from FESTIV


    # Prepare day-ahead UC data
    df_busload = pd.read_csv(csv_busload, index_col=0)
    df_genfor  = pd.read_csv(csv_genfor, index_col=0)
    df_busload = MyDataFrame(df_busload.loc[:, df_busload.columns.difference(['LOAD'])])
    df_genfor  = MyDataFrame(df_genfor)
    df_genfor.index = range(1, 25) # Kwami's convention: time starts from 1...
    df_genfor_nonthermal = df_genfor.loc[:, network.df_gen[network.df_gen['GEN_TYPE']!='Thermal'].index]
    df_genfor_nonthermal.fillna(0, inplace=True)

    # Prepare real-time UC (hourly ahead) data
    df_busload_ha = pd.read_csv(csv_busload_ha, index_col=['Slot'])
    df_genfor_ha  = pd.read_csv(csv_genfor_ha, index_col=['Slot'])
    df_busload_ha = MyDataFrame(df_busload_ha.loc[:, df_busload_ha.columns.difference(['LOAD'])])
    df_genfor_ha  = MyDataFrame(df_genfor_ha)
    df_genfor_ha  = df_genfor_ha.loc[:, network.df_gen[network.df_gen['GEN_TYPE']!='Thermal'].index]
    df_genfor_ha.fillna(0, inplace=True)

    # Prepare economic dispatch data
    df_busload_ed = pd.read_csv(csv_busload_ed, index_col=['Slot'])
    df_genfor_ed  = pd.read_csv(csv_genfor_ed, index_col=['Slot'])
    df_busload_ed = MyDataFrame(df_busload_ed.loc[:, df_busload_ed.columns.difference(['LOAD'])])
    df_genfor_ed  = MyDataFrame(df_genfor_ed)
    df_genfor_ed  = df_genfor_ed.loc[:, network.df_gen[network.df_gen['GEN_TYPE']!='Thermal'].index]
    df_genfor_ed.fillna(0, inplace=True)

    # Prepare AGC data
    df_busload_agc = pd.read_csv(csv_busload_agc, index_col='Slot')
    df_genfor_agc  = pd.read_csv(csv_genfor_agc, index_col='Slot')
    if 'LOAD' in df_busload_agc.columns.difference(['LOAD']):
        df_busload_agc = df_busload_agc.loc[:, df_busload_agc.columns.difference(['LOAD'])]
    df_busload_agc = MyDataFrame(df_busload_agc.loc[:, df_busload_agc.columns.difference(['LOAD'])])
    df_busload_agc.fillna(0, inplace=True)
    df_busload_full_agc = pd.DataFrame(0, index=df_busload_agc.index, columns=network.ls_bus) # For ACE calculation, include all buses
    df_busload_full_agc.loc[:, df_busload_agc.columns] = df_busload_agc
    df_genfor_agc  = MyDataFrame(df_genfor_agc)
    df_genfor_agc.fillna(0, inplace=True)
    df_agc_param = pd.DataFrame(0, index=network.df_gen.index, columns=['ACE_TARGET'])
    df_agc_param.loc[:, 'DEAD_BAND'] = network.df_gen.loc[:, 'DEAD_BAND']
    df_agc_param.loc[:, 'AGC_MODE']  = network.df_gen.loc[:, 'AGC_MODE']

    # Result container

    # RTUC results
    df_UNITON_RTUC_BINDING = MyDataFrame(index=df_genfor_ha.index, columns=network.dict_gens['Thermal'])
    df_UNITON_RTUC_ADVISRY = MyDataFrame(index=df_genfor_ha.index, columns=network.dict_gens['Thermal'])

    # Economic dispatch results
    df_POWER_RTED_BINDING = MyDataFrame(index=df_genfor_ed.index, columns=network.df_gen.index)
    df_POWER_RTED_ADVISRY = MyDataFrame(index=df_genfor_ed.index, columns=network.df_gen.index)
    df_REGUP_RTED_BINDING = MyDataFrame(index=df_genfor_ed.index, columns=network.df_gen.index)
    df_REGDN_RTED_BINDING = MyDataFrame(index=df_genfor_ed.index, columns=network.df_gen.index)
    df_SPNUP_RTED_BINDING = MyDataFrame(index=df_genfor_ed.index, columns=network.df_gen.index)

    # AGC results
    df_ACTUAL_GENERATION = MyDataFrame(index=df_genfor_agc.index, columns=network.df_gen.index)
    df_AGC_SCHEDULE      = MyDataFrame(index=df_genfor_agc.index, columns=network.df_gen.index)
    df_ACE_TARGET        = MyDataFrame(index=df_genfor_agc.index, columns=network.df_gen.index)
    df_AGC_MOVE          = MyDataFrame(index=df_genfor_agc.index, columns=network.df_gen.index)
    df_ACE               = MyDataFrame(index=df_genfor_agc.index)
    dict_ACE = {
        'Slot': list(), # Consider using Numpy array? How's the time performance?
        'RAW':  list(),
        'CPS2': list(),
        'SACE': list(),
        'ABS':  list(),
        'INT':  list(),
    }

    # Reserve margins, will be move to case specific data
    ReserveFactor = 0.1
    RegulatingReserveFactor = 0.05
    # Parameters for AGC, will be moved to case specific data
    CPS2_interval_minute = 10 # minutes
    K1, K2 = 0.5, 0.5
    Type3_integral = 180 # s, Number of seconds integrated over in smoothed ACE mode
    L10 = 50 # MW

    ############################################################################
    # Start DAUC
    model = create_model(
        network,
        df_busload,
        df_genfor_nonthermal,
        ReserveFactor,
        RegulatingReserveFactor,
        nI=1,
    )
    msg = 'Model created at: {:>.2f} s'.format(time() - t0)
    print(msg)
    content += msg
    content += '\n'

    instance = model
    optimizer = SolverFactory('cplex')
    results = optimizer.solve(instance)
    instance.solutions.load_from(results)
    msg = 'Model solved at: {:>.2f} s, objective: {:>.2f}'.format(
            time() - t0,
            value(instance.TotalCostObjective)
        )
    print(msg)
    content += msg
    content += '\n'
    # End of DAUC
    ############################################################################

    ############################################################################
    # Start RTUC
    df_uniton_da = pd.DataFrame(index=df_genfor.index, columns=network.dict_gens['Thermal'])
    for g, t in instance.UnitOn.iterkeys():
        df_uniton_da.at[t, g] = value(instance.UnitOn[g, t])

    # Obtain commitment statuses of slow-starting units at RTUC time scale from 
    # DAUC model results
    df_uniton_slow_ha = MyDataFrame(
        data=df_uniton_da[network.dict_gens['Thermal_slow']].values.repeat(nI_ha/nI_da, axis=0),
        index=df_genfor_ha.index,
        columns=network.dict_gens['Thermal_slow'],
    )
    # df_uniton_slow_ha = MyDataFrame(index=df_genfor_ha.index)
    # for g in network.dict_gens['Thermal_slow']:
    #     for h in instance.TimePeriods:
    #         v = int(value(instance.UnitOn[g, h]))
    #         for i in range(4*(h-1)+1, 4*h+1):
    #             df_uniton_slow_ha.at[i, g] = v

    # Create dispatch upper limits for commited slow-ramping (<1 hr) units at 
    # RTUC time scale
    df_DispatchLimits_slow_ha = return_dispatchupperlimits_from_uniton(
        df_uniton_slow_ha,
        network.df_gen.loc[network.dict_gens['Thermal_slow'], 'PMAX'].values,
        network.df_gen.loc[network.dict_gens['Thermal_slow'], 'RAMP_10'].values/nI_ha,
        network.df_gen.loc[network.dict_gens['Thermal_slow'], 'SHUTDOWN_RAMP'].values,
    )

    # RTUC and RTED initial parameters
    dict_UnitOnT0State       = None
    dict_UnitOnT0State_ed    = None
    dict_PowerGeneratedT0    = None
    dict_PowerGeneratedT0_ed = None

    # Start real-time simulations, RTUC and RTED
    ls_ins_ha = list()
    ls_ins_ed   = list()
    for i_rtuc in range(1, 94):
        t_start = i_rtuc # 4*(i_rtuc-1) + 1
        t_end   = i_rtuc + 3 # 4*i_rtuc
    # for i_rtuc in range(1, 25):
    #     t_start = 4*(i_rtuc-1) + 1
    #     t_end   = 4*i_rtuc

        dict_uniton_ha = MyDataFrame(
            df_uniton_slow_ha.loc[t_start: t_end, :].T
        ).to_dict_2d()

        dict_DispacthLimitsUpper_slow = MyDataFrame(
            df_DispatchLimits_slow_ha.loc[t_start: t_end, :].T
        ).to_dict_2d()

        # Create RTUC model
        ins_ha = create_model(
            network,
            df_busload_ha.loc[t_start: t_end, :],
            df_genfor_ha.loc[t_start: t_end, :],
            ReserveFactor,
            RegulatingReserveFactor,
            nI_ha,
            dict_UnitOnT0State, 
            dict_PowerGeneratedT0,
            dict_uniton_ha,
            dict_DispacthLimitsUpper_slow
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

        ins_ha.solutions.load_from(results)
        if hasattr(ins_ha, 'SlackPenalty'):
            if value(ins_ha.SlackPenalty) > 1E-5:
                print 'Infeasibility detected in the RTUC model!'
                IP()

        msg = (
            'RTUC Model {} '
            'solved at: {:>.2f} s, '
            'objective: {:>.2f}, '
            # 'penalty: {:s}'.format(
            'penalty: {:>.2f}'.format(
                i_rtuc, 
                time() - t0,
                value(ins_ha.TotalCostObjective),
                # 'N/A',
                value(ins_ha.SlackPenalty),
            )
        )
        print(msg)
        content += msg
        content += '\n'
        ls_ins_ha.append(ins_ha)

        # Save RTUC results
        t_binding_ha = t_start
        t_advisry_ha = ins_ha.TimePeriods.next(t_binding_ha)
        for g in ins_ha.ThermalGenerators:
            df_UNITON_RTUC_BINDING.at[t_start, g] = value(ins_ha.UnitOn[g, t_binding_ha])
            df_UNITON_RTUC_ADVISRY.at[t_start, g] = value(ins_ha.UnitOn[g, t_advisry_ha])

        ########################################################################
        # Start RTED

        # Initial number of hours thermal units having been online from RTUC run
        dict_uniton_ed = dict_UnitOnT0State

        # List of time intervals in the ED run
        ls_t_ed = df_timesequence.loc[df_timesequence['tQ']==t_start, 't5'].tolist()

        # IP()
        # Gather unit commitment statuses from the RTUC model
        df_uniton_fast_ha = pd.DataFrame(
            index=ins_ha.TimePeriods.value,
            columns=network.dict_gens['Thermal_fast'],
        )
        for g in network.dict_gens['Thermal_fast']:
            for t in ins_ha.TimePeriods.value:
                df_uniton_fast_ha.at[t, g] = value(ins_ha.UnitOn[g, t])
        df_uniton_all_ha = pd.concat(
            [
                df_uniton_slow_ha.loc[ins_ha.TimePeriods.value, :],
                df_uniton_fast_ha,
            ],
            axis=1,
        )

        # Conver the RTUC unit commitment statuses into the RTED time scale
        df_uniton_all_ed = pd.DataFrame(
            data=df_uniton_all_ha.values.repeat(nI_ed/nI_ha, axis=0),
            index=np.concatenate(
                [
                    np.arange(nI_ed/nI_ha*(i-1)+1, nI_ed/nI_ha*i+1) 
                    for i in df_uniton_all_ha.index
                ]
            ),
            columns=df_uniton_all_ha.columns,
        )

        # Gather upper dispatch limits for the RTED model
        df_DispatchLimits_ed = return_dispatchupperlimits_from_uniton(
            df_uniton_all_ed[network.dict_gens['Thermal']],
            network.df_gen.loc[network.dict_gens['Thermal'], 'PMAX'].values,
            network.df_gen.loc[network.dict_gens['Thermal'], 'RAMP_10'].values/nI_ed,
            network.df_gen.loc[network.dict_gens['Thermal'], 'SHUTDOWN_RAMP'].values,
        )

        for t_start_ed in ls_t_ed:
            t_end_ed = t_start_ed + 5 # Total 6 ED intervals, 1 binding interval, 5 look-ahead interval, 30 min in total

            # Obtain commitment statuses of slow-starting units from DAUC 
            # model, can we do this out of the for-loop?
            dict_uniton_ed = dict()
            for t5 in range(t_start_ed, t_end_ed+1):
                tQ = df_timesequence.loc[df_timesequence['t5'] == t5, 'tQ'].tolist()[0]
                for g in ins_ha.ThermalGenerators.iterkeys():
                    v = value(ins_ha.UnitOn[g, tQ])
                    dict_uniton_ed[g, t5] = int(round(v))
            
            dict_DispatchLimitsUpper_ed = MyDataFrame(
                df_DispatchLimits_ed.loc[t_start_ed: t_end_ed, :].T
            ).to_dict_2d()

            # Create ED model
            ins_ed = build_sced_model(
                network,
                df_busload_ed.loc[t_start_ed: t_end_ed, :], # Only bus load, first dimension time starts from 1, no total load. For ED model, there should be only 1 row
                df_genfor_ed.loc[t_start_ed: t_end_ed, :], # Only generation from nonthermal gens, first dim time starts from 1
                ReserveFactor,
                RegulatingReserveFactor,
                nI_ed, # Number of intervals in an hour, typically DAUC: 1, RTUC: 4
                dict_UnitOnT0State_ed, # Number of online hours of all gens
                dict_PowerGeneratedT0_ed, # Initial generation levels of all thermal gens at T0
                dict_uniton_ed, # Commitment statuses of ALL units from RTUC model
                dict_DispatchLimitsUpper_ed, # Upper dispatch limits for *all* thermal units
            )

            # Solve the model
            try:
                results_ed = optimizer.solve(ins_ed)
            except:
                print 'Cannot solve RTED model!'
                IP()

            if (results_ed.solver.termination_condition == TerminationCondition.infeasible):
                print 'Infeasibility detected in the RTED model!'
                IP()

            ins_ed.solutions.load_from(results)
            if hasattr(ins_ed, 'SlackPenalty'):
                if value(ins_ed.SlackPenalty) > 1E-3:
                    print 'Infeasibility detected in the RTED model!'
                    IP()
            msg = (
                '    RTED Model {} '
                'solved at: {:>.2f} s, '
                'objective: {:>.2f}, '
                'penalty: {:>.2f}'.format(
                    t_start_ed, 
                    time() - t0,
                    value(ins_ed.TotalCostObjective),
                    value(ins_ed.SlackPenalty),
                )
            )
            print msg
            content += msg
            content += '\n'
            ls_ins_ed.append(ins_ed)

            # End of RTED
            ####################################################################
            # Use df_timesequence in the future
            t_start_agc = int( (t_start_ed-1)*nI_agc/nI_ed+1 )
            t_end_agc   = int( nI_agc/nI_ed*t_start_ed ) 

            # Gather information for the AGC run
            ls_curtail_binding   = list()
            ls_nocurtail_binding = list()
            # for g in network.dict_gens['Non thermal']:
            #     if value(ins_ed.PowerGenerated[g, t_start_ed]) < value(ins_ed.PowerForecast[g, t_start_ed]):
            #         ls_curtail_binding.append(g)
            #     else:
            #         ls_nocurtail_binding.append(g)

            # Gather dispatch setting point, reserve and ramp available for AGC
            # Can we use a numpy array or a pandas data frame?
            dict_dispatch_binding  = dict() # Dispatch setting point at this interval, MW
            dict_dispatch_advisory = dict() # Dispatch setting point at the next interval, MW
            dict_reg_up_agc   = dict() # Reg-up for thermal gens, MW, only non-zeros
            dict_reg_dn_agc   = dict() # Reg-down for thermal gens, MW, only non-zeros
            dict_spn_up_agc   = dict() # Spinning reserve, MW, only up, only non-zeros
            dict_ramp_up_agc  = dict() # Ramp up capacity across one AGC interval, MW/AGC interval, only online units
            dict_ramp_dn_agc  = dict() # Ramp down capacity across one AGC interval, MW/AGC interval, only online units
            dict_start_up_agc = dict()
            dict_shut_dn_agc  = dict()

            # Start-up and shut-down status
            # df_uniton_agc = MyDataFrame(
            #     index = range(t_start_agc, t_end_agc+1),
            #     columns=network.dict_gens['Thermal'],
            # )
            for g in ins_ed.AllGenerators.iterkeys():
                dict_dispatch_binding[g]  = value(ins_ed.PowerGenerated[g, t_start_ed])
                dict_dispatch_advisory[g] = value(ins_ed.PowerGenerated[g, ins_ed.TimePeriods.next(t_start_ed)]) # What if there is no next?
                if g in ins_ed.ThermalGenerators: # In our case, only thermal gens provide regulation
                    reg_up = value(ins_ed.RegulatingReserveUpAvailable[g, t_start_ed])
                    reg_dn = value(ins_ed.RegulatingReserveDnAvailable[g, t_start_ed])
                    spn_up = value(ins_ed.SpinningReserveUpAvailable[g, t_start_ed])
                    dict_reg_up_agc[g] = reg_up
                    dict_reg_dn_agc[g] = reg_dn
                    dict_spn_up_agc[g] = spn_up
                    # if abs(value(ins_ed.UnitOn[g, t_start_ed])-1) < 1e-3:
                    #     dict_ramp_up_agc[g] = value(ins_ed.NominalRampUpLimit[g])/(nI_agc/nI_ed)
                    #     dict_ramp_dn_agc[g] = value(ins_ed.NominalRampDownLimit[g])/(nI_agc/nI_ed)
                    if abs(
                        value(ins_ed.UnitOn[g, t_start_ed]) 
                        -
                        value(ins_ed.UnitOnT0[g]) 
                        - 
                        1
                    ) < 1e-3: # Unit is starting up
                        dict_ramp_up_agc[g] = value(ins_ed.StartupRampLimit[g])/(nI_agc/nI_ed)
                        dict_ramp_dn_agc[g] = 0
                    elif abs(
                        value(ins_ed.UnitOn[g, t_start_ed]) 
                        -
                        value(ins_ed.UnitOnT0[g]) 
                        + 
                        1
                    ) < 1e-3: # Unit is shutting down
                        dict_ramp_up_agc[g] = 0
                        dict_ramp_dn_agc[g] = value(ins_ed.ShutdownRampLimit[g])/(nI_agc/nI_ed)
                    elif abs(
                        value(ins_ed.UnitOn[g, t_start_ed])
                        -
                        1
                    ) < 1e-3: # Online units going through normal ramps
                        dict_ramp_up_agc[g] = value(ins_ed.NominalRampUpLimit[g])/(nI_agc/nI_ed)
                        dict_ramp_dn_agc[g] = value(ins_ed.NominalRampDownLimit[g])/(nI_agc/nI_ed)
                    else: # Units are turned off in both time intervals
                        dict_ramp_up_agc[g] = 0
                        dict_ramp_dn_agc[g] = 0
                    # for t in ins_ed.TimePeriods:
                    #     df_uniton_agc.at[t_start_agc: t_end_agc, g] = value(ins_ed.UnitOn[g, t])
                else: # Non-thermal genes, gather curtailment status
                    if value(ins_ed.PowerGenerated[g, t_start_ed]) < value(ins_ed.PowerForecast[g, t_start_ed]):
                        ls_curtail_binding.append(g)
                    else:
                        ls_nocurtail_binding.append(g)

            df_POWER_RTED_BINDING.loc[t_start_ed, :] = pd.Series(dict_dispatch_binding)
            df_POWER_RTED_ADVISRY.loc[t_start_ed, :] = pd.Series(dict_dispatch_advisory)
            df_REGUP_RTED_BINDING.loc[t_start_ed, :] = pd.Series(dict_reg_up_agc)
            df_REGDN_RTED_BINDING.loc[t_start_ed, :] = pd.Series(dict_reg_dn_agc)
            df_SPNUP_RTED_BINDING.loc[t_start_ed, :] = pd.Series(dict_spn_up_agc)
            df_REGUP_RTED_BINDING.loc[t_start_ed, :].fillna(0, inplace=True)
            df_REGDN_RTED_BINDING.loc[t_start_ed, :].fillna(0, inplace=True)
            df_SPNUP_RTED_BINDING.loc[t_start_ed, :].fillna(0, inplace=True)

            df_agc_tmp = pd.DataFrame.from_dict(
                {
                    'REG_UP_AGC':       dict_reg_up_agc,
                    'REG_DN_AGC':       dict_reg_dn_agc,
                    'ED_DISPATCH':      dict_dispatch_binding,
                    'ED_DISPATCH_NEXT': dict_dispatch_advisory,
                    'RAMP_UP_AGC':      dict_ramp_up_agc,
                    'RAMP_DN_AGC':      dict_ramp_dn_agc,
                }
            )
            # Non-thermal units do not provide regulations
            df_agc_tmp.loc[:, 'REG_UP_AGC'].fillna(0, inplace=True)
            df_agc_tmp.loc[:, 'REG_DN_AGC'].fillna(0, inplace=True)
            # The following is a kluge, we use rated capacity of non-thermal 
            # units as their AGC ramp limits, basically the AGC movement will 
            # also be constrained by the RTED schedule, so don't worry about 
            # the data accuracy. We just need "something" there rather than NA.
            i_na = pd.isna(df_agc_tmp['RAMP_UP_AGC'])
            df_agc_tmp.loc[i_na, 'RAMP_UP_AGC'] = network.df_gen.loc[df_agc_tmp.index[i_na],'PMAX']
            i_na = pd.isna(df_agc_tmp['RAMP_DN_AGC'])
            df_agc_tmp.loc[i_na, 'RAMP_DN_AGC'] = network.df_gen.loc[df_agc_tmp.index[i_na],'PMAX']

            # Copy AGC mode and dead band
            df_agc_tmp.loc[:, 'AGC_MODE']  = df_agc_param.loc[:, 'AGC_MODE']
            df_agc_tmp.loc[:, 'DEAD_BAND'] = df_agc_param.loc[:, 'DEAD_BAND']

            # Copy PMIN, this part out of the loop?
            df_agc_tmp.loc[:, 'PMIN'] = network.df_gen['PMIN']
            df_agc_tmp.loc[:, 'PMAX'] = network.df_gen['PMAX']

            sum_reg_up_agc  = df_agc_tmp['REG_UP_AGC'].sum()
            sum_reg_dn_agc  = df_agc_tmp['REG_DN_AGC'].sum()
            # sum_ramp_up_agc = sum(dict_ramp_up_agc[g] for g in dict_ramp_up_agc.iterkeys())
            # sum_ramp_dn_agc = sum(dict_ramp_dn_agc[g] for g in dict_ramp_dn_agc.iterkeys())

            # Start of the AGC loop
            ####################################################################
            for t_AGC in range(t_start_agc, t_end_agc+1):
                # Calculate actual generation
                if t_AGC == 1:
                    dict_tmp = dict()
                    for g in network.dict_gens['ALL']:
                        dict_tmp[g] = [value(ins_ed.PowerGenerated[g, t_start_ed])]
                    df_tmp = pd.DataFrame(dict_tmp)
                    df_ACTUAL_GENERATION.loc[t_AGC, :] = df_tmp[df_ACTUAL_GENERATION.columns].values # Make sure columns are aligned
                else:
                    df_ACTUAL_GENERATION.loc[t_AGC, network.dict_gens['Thermal']] = df_AGC_SCHEDULE.loc[t_AGC-1, network.dict_gens['Thermal']]
                    df_ACTUAL_GENERATION.loc[t_AGC, ls_curtail_binding] = pd.concat(
                        [
                            df_AGC_SCHEDULE.loc[t_AGC-1, ls_curtail_binding],
                            df_genfor_agc.loc[t_AGC, ls_curtail_binding],
                        ], 
                        axis=1,
                    ).min(axis=1)
                    df_ACTUAL_GENERATION.loc[t_AGC, ls_nocurtail_binding] = df_genfor_agc.loc[t_AGC, ls_nocurtail_binding]
                df_agc_tmp.loc[:, 'ACTUAL_GENERATION'] = df_ACTUAL_GENERATION.loc[t_AGC, :]

                # Calculate ACE
                # First, calculate line losses, bus order is the same as in network.df_ptdf column direction
                # sparse matrix operation?
                bus_inject = np.matmul(
                    network.mat_busgen, 
                    df_ACTUAL_GENERATION.loc[t_AGC, network.dict_gens['ALL']].tolist()
                ) - df_busload_full_agc.loc[t_AGC, network.df_ptdf.columns]
                flow_br = np.matmul(network.df_ptdf.values, bus_inject)
                total_loss = sum(
                    (
                        (flow_br/network.baseMVA)**2
                        *
                        network.df_branch.loc[network.df_ptdf.index, 'BR_R'].values
                    )
                    *
                    network.baseMVA
                ) # This is not correct by following FESTIV, how should we improve it?
                total_gen  = df_ACTUAL_GENERATION.loc[t_AGC, network.dict_gens['ALL']].sum()
                total_load = df_busload_agc.loc[t_AGC, ins_ed.LoadBuses.value].sum()
                ace_raw = total_gen - total_load # - total_loss

                if t_AGC == 1:
                    previous_ACE_int  = 0
                    previous_CPS2_ACE = 0
                    previous_SACE     = 0
                    previous_ACE_ABS  = 0
                else:
                    previous_ACE_int  = dict_ACE['INT'][-1]
                    previous_ACE_ABS  = dict_ACE['ABS'][-1]
                    if t_AGC%(CPS2_interval_minute*60/(3600/nI_agc))==1: # This indicates the start of a new CPS2 interval
                        previous_CPS2_ACE = 0
                    else:
                        previous_CPS2_ACE = dict_ACE['CPS2'][-1]
                    i_s = dict_ACE['Slot'].index(
                        max( 1, round(t_AGC - Type3_integral/(3600/nI_agc)) )
                    )
                    i_e = dict_ACE['Slot'].index(
                        t_AGC-1
                    )
                    previous_SACE = dict_ACE['SACE'][i_s: i_e+1]

                dict_ACE['Slot'].append(t_AGC)
                dict_ACE['RAW' ].append(ace_raw)
                dict_ACE['INT' ].append(ace_raw*float(t_AGC/nI_agc) + previous_ACE_int)
                dict_ACE['ABS' ].append(abs(ace_raw*float(t_AGC/nI_agc)) + previous_ACE_ABS)
                dict_ACE['CPS2'].append(ace_raw*(t_AGC*3600/nI_agc/(CPS2_interval_minute*60.0)) + previous_CPS2_ACE)
                dict_ACE['SACE'].append(K1*ace_raw + K2*np.mean(previous_SACE))

                # Set ACE target for AGC
                seconds_Left_in_CPS2_interval = (
                    CPS2_interval_minute*60.0 
                    - 
                    (t_AGC*3600/nI_agc) % (CPS2_interval_minute*60)
                ) # In second
                df_agc_tmp.loc[df_agc_tmp['AGC_MODE']=='RAW', 'ACE_TARGET']    = dict_ACE['RAW'][-1]
                df_agc_tmp.loc[df_agc_tmp['AGC_MODE']=='SMOOTH', 'ACE_TARGET'] = dict_ACE['SACE'][-1]
                df_agc_tmp.loc[df_agc_tmp['AGC_MODE']=='CPS2', 'ACE_TARGET']   = (
                    dict_ACE['CPS2'][-1] 
                    + 
                    dict_ACE['RAW'][-1]*(
                        seconds_Left_in_CPS2_interval/(CPS2_interval_minute*60.0)
                    )
                )
                df_agc_tmp.loc[
                    (df_agc_tmp['AGC_MODE'] == 'CPS2')
                    & (df_agc_tmp['DEAD_BAND'] <= L10),
                    'DEAD_BAND'
                ] = L10

                # Finally, the AGC module
                # Determine responding and non-responding units to ACE signals
                i_reg_up_units = (
                    ( df_agc_tmp.loc[:, 'ACE_TARGET'] <= -df_agc_tmp.loc[:, 'DEAD_BAND'] ) &
                    ( df_agc_tmp.loc[:, 'REG_UP_AGC'] >= 1E-3 )
                ) # Units that deploy reg-up reserves
                i_reg_dn_units = (
                    ( df_agc_tmp.loc[:, 'ACE_TARGET'] >= df_agc_tmp.loc[:, 'DEAD_BAND'] ) &
                    ( df_agc_tmp.loc[:, 'REG_DN_AGC'] >= 1E-3 )
                ) # Units that deploy reg-down reserves

                # First, determine RTED scheduled movement during one AGC 
                # interval for reg non-responding units, note AGC_MOVE can be 
                # either positive or negative 

                # Use the initial actual generation and next adivsory dispatch 
                # level to determine the RTED scheduled movement, to avoid 
                # generation falling below minimum thermal power level
                df_agc_tmp.loc[~(i_reg_up_units | i_reg_dn_units), 'AGC_MOVE'] = (
                    df_agc_tmp[['PMIN', 'ED_DISPATCH_NEXT']].max(axis=1) # Bounded below by PMIN
                    - df_ACTUAL_GENERATION.loc[int( (t_start_ed-1)*nI_agc/nI_ed+1 ), :]
                )/(nI_agc/nI_ed) # Maybe the calculation of RTED step out of the AGC loop?

                # Calculate upper/lower power limit including regulation
                # Because in my RTED formulation, the max available power only 
                # considers a fixed dispatch setting point + reg-up + spin-up, 
                # while in AGC the dispatch setting point changes from this 
                # interval continuously into the next interval, and the sum may
                # exceed the Pmax or lower than Pmin, thus the sums must be 
                # capped by Pmax, or floored by Pmin.
                df_agc_tmp.loc[:, 'POWER+REGUP'] = pd.concat(
                    [
                        df_agc_tmp['PMAX'],
                        df_ACTUAL_GENERATION.loc[t_AGC, :] + df_agc_tmp.loc[:, 'REG_UP_AGC']
                    ],
                    axis=1
                ).min(axis=1)
                df_agc_tmp.loc[:, 'POWER-REGDN'] = pd.concat(
                    [
                        df_agc_tmp['PMIN'],
                        df_ACTUAL_GENERATION.loc[t_AGC, :] - df_agc_tmp.loc[:, 'REG_DN_AGC']
                    ],
                    axis=1
                ).max(axis=1)

                # Then, determine AGC movement for AGC responding units, we 
                # follow FESTIV's option 2, where each unit's deployed 
                # regulation is proportional to its regulation bid into the RTED market
                df_agc_tmp.loc[i_reg_up_units, 'AGC_MOVE'] = pd.concat(
                    [
                        -df_agc_tmp.loc[:, 'ACE_TARGET']*df_agc_tmp.loc[:, 'REG_UP_AGC']/sum_reg_up_agc,
                        df_agc_tmp.loc[:, 'POWER+REGUP'] - df_ACTUAL_GENERATION.loc[t_AGC, :],
                    ],
                    axis=1,
                ).min(axis=1)
                df_agc_tmp.loc[i_reg_dn_units, 'AGC_MOVE'] = pd.concat(
                    [
                        -df_agc_tmp.loc[:, 'ACE_TARGET']*df_agc_tmp.loc[:, 'REG_DN_AGC']/sum_reg_dn_agc,
                        df_agc_tmp.loc[:, 'POWER-REGDN'] - df_ACTUAL_GENERATION.loc[t_AGC, :],
                    ],
                    axis=1,
                ).max(axis=1)

                # Now, determine AGC basepoint, based on ramp rate limits, 
                # RTED movement (reg non-responding units) and AGC movement 
                # (reg responding units).
                # Basically, reg responding units are bounded by three terms: 
                # available regulation, ramp rate limits and its own ACE target, 
                # while non-responding units are constrained by RTED schedules 
                # and ramp rate limits.
                tmp = pd.concat(
                    [
                        df_agc_tmp.loc[:, 'AGC_MOVE'],
                        -df_agc_tmp.loc[:, 'RAMP_DN_AGC'] # ramp down is positive
                    ],
                    axis=1
                ).max(axis=1)
                df_agc_tmp.loc[:, 'AGC_BASEPOINT'] = pd.concat(
                    [
                        tmp,
                        df_agc_tmp.loc[:, 'RAMP_UP_AGC']
                    ],
                    axis=1
                ).min(axis=1) + df_agc_tmp.loc[:, 'ACTUAL_GENERATION']

                # Finally, let's set the actual generation of all down units to zero
                df_agc_tmp.loc[
                    df_uniton_all_ed.loc[t_start_ed, abs(df_uniton_all_ed.loc[t_start_ed, :])<1E-3].index, 
                    'AGC_BASEPOINT'
                ] = 0

                df_AGC_SCHEDULE.loc[t_AGC, :] = df_agc_tmp.loc[:, 'AGC_BASEPOINT']
                df_AGC_MOVE.loc[t_AGC, :]     = df_agc_tmp.loc[:, 'AGC_MOVE']
                df_ACE_TARGET.loc[t_AGC, :]   = df_agc_tmp.loc[:, 'ACE_TARGET']

                df_AGC_SCHEDULE[df_AGC_SCHEDULE < 0] = 0 # Fix numerical errors

                ################################################################
                # The following code is adapted from FESTIV's implementation
                ################################################################

                # # First calculate interpolated RTED schedules
                # df_agc_tmp.loc[:, 'RTED_INTERPOLATED'] = (
                #     df_agc_tmp.loc[:, 'ACTUAL_GENERATION'] 
                #     + ( t_AGC - (t_start_ed-1)*(nI_agc/nI_ed) )*(
                #         df_agc_tmp.loc[:, 'ED_DISPATCH_NEXT'] 
                #         - df_agc_tmp.loc[:, 'ED_DISPATCH']
                #     )
                # )
                # # Second calculate max and min AGC limits
                # df_agc_tmp.loc[:, 'AGC_LIMIT_UP'] = pd.concat(
                #     [
                #         df_agc_tmp.loc[:, 'ACTUAL_GENERATION'] + df_agc_tmp.loc[:, 'RAMP_UP_AGC'],
                #         df_agc_tmp.loc[:, 'RTED_INTERPOLATED'] + df_agc_tmp.loc[:, 'REG_UP_AGC']
                #     ],
                #     axis=1,
                # ).min(axis=1)
                # df_agc_tmp.loc[:, 'AGC_LIMIT_DN'] = pd.concat(
                #     [
                #         df_agc_tmp.loc[:, 'ACTUAL_GENERATION'] - df_agc_tmp.loc[:, 'RAMP_DN_AGC'],
                #         df_agc_tmp.loc[:, 'RTED_INTERPOLATED'] - df_agc_tmp.loc[:, 'REG_DN_AGC']
                #     ],
                #     axis=1,
                # ).max(axis=1)
                # # Third, determine AGC movement, proportional to reg available
                # i_reg_up_units = (
                #     ( df_agc_tmp.loc[:, 'ACE_TARGET'] <= -df_agc_tmp.loc[:, 'DEAD_BAND'] ) &
                #     ( df_agc_tmp.loc[:, 'REG_UP_AGC'] >= 1E-3 )
                # )
                # i_reg_dn_units = (
                #     ( df_agc_tmp.loc[:, 'ACE_TARGET'] >= df_agc_tmp.loc[:, 'DEAD_BAND'] ) &
                #     ( df_agc_tmp.loc[:, 'REG_DN_AGC'] >= 1E-3 )
                # )
                # df_agc_tmp.loc[i_reg_up_units, 'AGC_MOVE'] = pd.concat(
                #     [
                #         df_agc_tmp.loc[:, 'RAMP_UP_AGC'],
                #         -df_agc_tmp.loc[:, 'ACE_TARGET']*df_agc_tmp.loc[:, 'REG_UP_AGC']/sum_reg_up_agc
                #     ],
                #     axis=1,
                # ).min(axis=1)
                # df_agc_tmp.loc[i_reg_dn_units, 'AGC_MOVE'] = pd.concat(
                #     [
                #         -df_agc_tmp.loc[:, 'RAMP_DN_AGC'],
                #         -df_agc_tmp.loc[:, 'ACE_TARGET']*df_agc_tmp.loc[:, 'REG_DN_AGC']/sum_reg_dn_agc
                #     ],
                #     axis=1,
                # ).max(axis=1)
                # df_agc_tmp.loc[
                #     ~(i_reg_up_units | i_reg_dn_units),
                #     'AGC_MOVE'
                # ] = 0

                # # Finally, determine AGC basepoint
                # tmp = pd.concat(
                #     [
                #         df_agc_tmp.loc[(i_reg_up_units | i_reg_dn_units), 'AGC_LIMIT_DN'],
                #         df_agc_tmp.loc[(i_reg_up_units | i_reg_dn_units), 'ACTUAL_GENERATION'] + 
                #         df_agc_tmp.loc[(i_reg_up_units | i_reg_dn_units), 'AGC_MOVE']
                #     ],
                #     axis=1
                # ).max(axis=1)
                # df_agc_tmp.loc[(i_reg_up_units | i_reg_dn_units), 'AGC_BASEPOINT'] = pd.concat(
                #     [
                #         tmp,
                #         df_agc_tmp.loc[(i_reg_up_units | i_reg_dn_units), 'AGC_LIMIT_UP']
                #     ],
                #     axis=1
                # ).min(axis=1)
                # tmp = pd.concat(
                #     [
                #         df_agc_tmp.loc[~(i_reg_up_units | i_reg_dn_units), 'AGC_LIMIT_DN'],
                #         df_agc_tmp.loc[~(i_reg_up_units | i_reg_dn_units), 'RTED_INTERPOLATED']
                #     ],
                #     axis=1
                # ).max(axis=1)
                # df_agc_tmp.loc[~(i_reg_up_units | i_reg_dn_units), 'AGC_BASEPOINT'] = pd.concat(
                #     [
                #         tmp,
                #         df_agc_tmp.loc[(i_reg_up_units | i_reg_dn_units), 'AGC_LIMIT_UP']
                #     ],
                #     axis=1
                # ).min(axis=1)
                # df_AGC_SCHEDULE.loc[t_AGC, :] = df_agc_tmp.loc[:, 'AGC_BASEPOINT']

                # This is for debugging
                # if t_start_agc >= 501:
                #     IP()

            # End of the AGC loop
            ####################################################################

            # Extract initial parameters from the binding interval for the next RTED run
            dict_UnitOnT0State_ed = return_unitont0state(
                ins_ed, ins_ed.TimePeriods.first()
            )
            # dict_PowerGeneratedT0_ed = return_powergenerated_t(
            #     ins_ed, ins_ed.TimePeriods.first()
            # )
            dict_PowerGeneratedT0_ed = df_AGC_SCHEDULE.loc[t_AGC, network.dict_gens['Thermal']].to_dict()


        # Extract initial parameters from the binding interval of the last ED run for the next RTUC run
        dict_UnitOnT0State = return_unitont0state(ins_ha, ins_ha.TimePeriods.first())
        # dict_PowerGeneratedT0 = return_powergenerated_t(ins_ha, ins_ha.TimePeriods.first())
        dict_PowerGeneratedT0 = dict_PowerGeneratedT0_ed

    df_ACE = pd.DataFrame(dict_ACE)
    
    # Write all results
    # df_ACE.to_csv('ACE.csv', index=False)
    # df_ACTUAL_GENERATION.to_csv('ACTUAL_GENERATION.csv')
    # df_AGC_SCHEDULE.to_csv('AGC_SCHEDULE.csv')
    # df_ACE_TARGET.to_csv('ACE_TARGET.csv')
    # df_AGC_MOVE.to_csv('AGC_MOVE.csv')
    # df_POWER_RTED_BINDING.to_csv('POWER_RTED_BINDING.csv')
    # df_POWER_RTED_ADVISRY.to_csv('POWER_RTED_ADVISRY.csv')
    # df_REGUP_RTED_BINDING.to_csv('REGUP_RTED_BINDING.csv')
    # df_REGDN_RTED_BINDING.to_csv('REGDN_RTED_BINDING.csv')
    # df_SPNUP_RTED_BINDING.to_csv('SPNUP_RTED_BINDING.csv')
    # df_UNITON_RTUC_BINDING.to_csv('UNITON_RTUC_BINDING.csv')
    # df_UNITON_RTUC_ADVISRY.to_csv('UNITON_RTUC_ADVISRY.csv')

    if content:
        if len(sys.argv) > 1:
            mail_pass = sys.argv[1]
            mailto_list = ['reagan.fruit@gmail.com']
            # content = "The model run is completed!"
            # if os.path.isfile('lab.out'):
            #     with open('lab.out', 'r') as myfile:
            #         content = myfile.readlines()
            #         content = ''.join(content)
            if send_mail(
                mailto_list,
                "Model run completed.",
                content,
                mail_pass
            ):
                print "Email sent successfully!"  
            else:  
                print "Email sent failed."

    IP()

def sequential_run_new():
    pass


if __name__ == "__main__":
    # sequential_run_old()
    sequential_run_new()
