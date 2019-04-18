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
from SCUC_RampConstraint_3 import create_model, da_input, Network, MyDataFrame
from economic_dispatch import *

# model = create_model()

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

# The following is for Zhang's 10 independent runs in July
################################################################################

def return_wind_farms():
    data_path = os.path.sep.join(
        ['.', 'TEXAS2k_B']
    )
    gen_df = pd.read_csv(
        os.path.sep.join( [data_path, 'generator_data_plexos_withRT.csv'] ),
        index_col=0,
    )
    wind_generator_names  =  [ x for x in gen_df.index if x.startswith('wind') ]
    return wind_generator_names

def import_scenario_data_mucun():
    wind_generator_names = return_wind_farms()
    # data_path = os.path.sep.join(
    #     ['.', 'TEXAS2k_B']
    # )
    # gen_df = pd.read_csv(
    #     os.path.sep.join( [data_path, 'generator_data_plexos_withRT.csv'] ),
    #     index_col=0,
    # )
    # wind_generator_names  =  [ x for x in gen_df.index if x.startswith('wind') ]
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

def vary_wind_penetration(year, month, day, use_forecast=False):
    # This function is used to provide 10% to 90% wind penetrations, before use
    # this function, the wind scaling facotr should set to 1.0.
    # Although wind power capacity is also defined, it is strange that it is not
    # used in the model itself, only forecasted energy is used. I skim over the
    # code to check it, I might be wrong.

    def return_wind(sid_WIND, year, month, day, use_forecast=False):
        map_day2dir   = {
            (2012, 2, 10):  '/home/bxl180002/git/FlexibleRampSCUC/10scenario_independent/20120210',
            (2012, 5, 10):  '/home/bxl180002/git/FlexibleRampSCUC/10scenario_independent/20120510',
            (2012, 8, 10):  '/home/bxl180002/git/FlexibleRampSCUC/10scenario_independent/20120810',
            (2012, 11, 10): '/home/bxl180002/git/FlexibleRampSCUC/10scenario_independent/20121110',
        }
        dirwinddata = map_day2dir[year, month, day]
        p_csf = os.path.sep.join([dirwinddata, str(sid_WIND)+'.csv'])
        df = pd.read_csv(p_csf)
        i_selected = (df['year'] == year)&(df['month'] == month)&(df['day'] == day)
        if use_forecast is True:
            print 'Use forecasted data!'
            return df.loc[i_selected, 'day-ahead']
        else:
            print 'Use real data!'
            return df.loc[i_selected, 'actual']
    wind_generator_names = return_wind_farms()
    data_path = os.path.sep.join(
        ['.', 'TEXAS2k_B']
    )
    load_df = pd.read_csv(
        os.path.sep.join( [data_path, 'loads.csv'] ),
        index_col=0,
    )
    df_wind_mapping   = pd.DataFrame.from_csv(
        os.path.sep.join(
            [data_path, 'wind_generator_data.csv']
        )
    )
    map_wind2site = df_wind_mapping['SITE_ID'].to_dict()

    df_wind = dict()
    for w in wind_generator_names:
        sid_WIND = map_wind2site[w]
        pwind_thatday = return_wind(sid_WIND, year, month, day, use_forecast)
        df_wind[w] = pwind_thatday.tolist()

    total_wind = sum(sum(df_wind[w]) for w in df_wind)
    total_load = sum(load_df['LOAD'])
    x0 = total_wind/total_load # Current wind penetration

    WindPowerForecast = dict()
    penetrations = range(10, 100, 10)
    for x in penetrations:
        xname = 'S' + str(x)
        scaling = x*0.01/x0
        WindPowerForecast[xname] = dict()
        for w in wind_generator_names:
            for h in range(1, 25):
                # Because in their generator.csv file, time starts from 0
                # WindPowerForecast[xname][w, h] = genfor_df.loc[h-1, w]*scaling
                WindPowerForecast[xname][w, h] = df_wind[w][h-1]*scaling
    print "Scenario data created!"
    return WindPowerForecast

def scenario_data_118():
    PowerForecastWind_w = import_scenario_data_mucun()
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
    for W in W2w:
        w = W2w[W]
        for s in PowerForecastWind_w:
            if s not in PowerForecastWind_W:
                PowerForecastWind_W[s] = dict()
            for h in range(1, 25):
                PowerForecastWind_W[s][W, h] = PowerForecastWind_w[s][w, h]
    return PowerForecastWind_W

def independent_run_10_case(year, month, day, use_forecast):
    t0 = time()
    # WindPowerForecast = import_scenario_data_mucun()
    WindPowerForecast = vary_wind_penetration(year, month, day, use_forecast)
    # WindPowerForecast = scenario_data_118()
    dirhome = os.getcwd()
    dirwork = 'results_10scenario_independent_20120510'
    if not os.path.isdir(dirwork):
        os.mkdir(dirwork)
    os.chdir(dirwork)
    instance = model.create_instance()
    for s in WindPowerForecast:
        instance.PowerForecast.store_values( WindPowerForecast[s] )
        instance.preprocess()
        optimizer = SolverFactory('cplex')
        results = optimizer.solve(instance)
        instance.solutions.load_from(results)
        print "Scenario {:s} solved at {:>7.2f} s, value: {:>.2f}.".format(
            s,
            time() - t0,
            value( instance.TotalCostObjective ),
        )
        store_csvs(instance, s)
        print "Results stored!"
    os.chdir(dirhome)

################################################################################
# End of Zhang's 10 independent runs in July

def solve_single_case(scenario_name=None):
    snames, sdata = import_scenario_data()

    print 'Case starts at {}'.format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    )
    dirhome = os.getcwd()
    if not scenario_name:
        scenario_name = 'xa'
    print 'Scenario: ', scenario_name, 'PID: ', os.getpid()

    if not os.path.isdir(scenario_name):
        os.mkdir(scenario_name)
    os.chdir(scenario_name)

    model = create_model()

    instance = model
    instance.PowerForecast.store_values(sdata[scenario_name]) # Use scenario data
    optimizer = SolverFactory('cplex')
    results = optimizer.solve(instance)
    instance.solutions.load_from(results)
    store_csvs(instance, 'UC')
    print "Single case {} solved at {}, value: {:>.2f}.".format(
        scenario_name,
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
        value( instance.TotalCostObjective ),
    )

    # Solve dispatch model
    print 'Solving dispatch model...'
    print 'Dispatch model starts at {}'.format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    )
    instance = solve_dispatch_model(instance)
    store_csvs(instance, 'ED')
    print "Dispatch model solved at {}, value: {:>.2f}.".format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
        value( instance.TotalCostObjective ),
    )
    os.chdir(dirhome)

def solve_dispatch_model(instance, dict_uniton=None):
    snames, sdata = import_scenario_data()

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

    # Now we solve it...
    # instance.preprocess() # Do we really need to do this?
    optimizer = SolverFactory('cplex')
    results = optimizer.solve(instance)
    instance.solutions.load_from(results)
    return instance

def solve_after_stochastic():
    print 'Case starts at {}'.format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    )
    csvef = '/home/bxl180002/git/FlexibleRampSCUC/results_runef_reduced_reserve/ef.csv'
    snames, sdata = import_scenario_data()
    dirwork = os.path.dirname(csvef)
    dirhome = os.getcwd()

    os.chdir(dirwork)
    dict_uniton = extract_uniton(csvef)
    print 'Creating model...'
    instance = create_model()
    print 'Model created at {}'.format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    )
    print 'Solving dispatch model...'
    print 'Dispatch model starts at {}'.format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S')
    )
    instance = solve_dispatch_model(instance, dict_uniton=dict_uniton)
    store_csvs(instance, 'ED')
    print "Dispatch model solved at {}, value: {:>.2f}.".format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
        value( instance.TotalCostObjective ),
    )
    os.chdir(dirhome)

def par_solve():
    ncores = multiprocessing.cpu_count()
    sname = range(10)
    sdata = range(10)
    for s in sname:
        multiprocessing.Process(
            target=solve_single_case,
            args=(s, sdata[sname.index(s)])
        ).start()
    print 'Multiprocessing done!'

def test_par():
    def f(a, q):
        print 'PID: ', os.getpid()
        q.put(a)

    q = multiprocessing.Queue()
    index = range(0, 10)
    for i in index:
        p = multiprocessing.Process(target=f, args=(i, q))
        p.start()

    IP()

def extract_all_scenarios():
    dir_sto = '/home/bxl180002/git/FlexibleRampSCUC/results_runef_reduced_reserve'
    dir_det = '/home/bxl180002/git/FlexibleRampSCUC/results_determin_reduced_reserve'
    dict_labels = {
        'N1Scenario':  'N1',
        'N2Scenario':  'N2',
        'N3Scenario':  'N3',
        'N4Scenario':  'N4',
        'N5Scenario':  'N5',
        'N6Scenario':  'N6',
        'N7Scenario':  'N7',
        'N8Scenario':  'N8',
        'N9Scenario':  'N9',
        'N10Scenario': 'N10',
        'xa':          'A',
        'SP':          'S',
        'xf':          'D',
    }
    # plt.style.use('ggplot')

    # When dim = 0
    dict_dim0 = dict()
    sname = ['N' + str(i) +'Scenario' for i in  range(1, 11)] + ['xf', 'xa']
    list_csv = [
        os.path.sep.join(
            [dir_det, s, 'ED', '0dim_var.csv']
        )
        for s in sname
    ]
    sname = sname + ['SP']
    list_csv.append(os.path.sep.join([dir_sto, 'ED', '0dim_var.csv']))
    for i in range(0, len(sname)):
        s = sname[i]
        fcsv = list_csv[i]
        df_tmp = pd.read_csv(fcsv, index_col=0)
        dict_dim0[s] = df_tmp['Values']
    df_dim0 = pd.DataFrame(dict_dim0, columns=sname)
    df_dim0.index = df_tmp.index
    df_dim0.loc['TotalCost', :] = df_dim0.sum(axis=0)

    # When dim = 1
    list_csv = [
        os.path.sep.join([os.path.dirname(f), '1dim_var.csv'])
        for f in list_csv
    ]
    varname_dim1 = [
        'RegulatingReserveUpShortage',
        'RegulatingReserveDnShortage',
        'SpinningReserveUpShortage',
        'Curtailment',
    ]
    dict_dict_dim1 = dict()
    dict_df_dim1   = dict()
    for i in range(0, len(sname)):
        s = sname[i]
        fcsv = list_csv[i]
        df_tmp = pd.read_csv(fcsv, index_col=0)
        for v in varname_dim1:
            if v not in dict_dict_dim1:
                dict_dict_dim1[v] = dict()
            dict_dict_dim1[v][s] = df_tmp[v].tolist()
    for v in varname_dim1:
        dict_df_dim1[v] = pd.DataFrame(dict_dict_dim1[v], columns=sname)

    df_y = dict_df_dim1['SpinningReserveUpShortage'].sum(axis=0)
    plt.figure()
    ax = plt.subplot()
    x = range(len(sname))
    y = [df_y[k]/1E3 for k in sname]
    xname = [dict_labels[s] for s in sname]
    ax.bar(x, y, color='k',alpha=0.4,)
    ax.set_xticks(x)
    ax.set_xticklabels(xname)
    ax.set_ylabel('Spinning reserve shortage (GWh)', fontsize=18)

    # When dim = 2, unit on
    list_csv = [
        os.path.sep.join([os.path.dirname(f), 'UnitOn.csv'])
        for f in list_csv
    ]
    # UnitOn
    dict_dict_uniton = dict()
    dict_df_uniton   = dict()
    for i in range(0, len(sname)):
        s = sname[i]
        fcsv = list_csv[i]
        df_tmp = pd.read_csv(fcsv, index_col=0)
        dict_dict_uniton[s] = dict()
        for c in df_tmp.columns:
            if c.startswith('ng'):
                fuel = 'NG'
            elif c.startswith('coal'):
                fuel = 'coal'
            elif c.startswith('nuc'):
                fuel = 'nuclear'
            else:
                print c
                raise TypeError
            if fuel not in dict_dict_uniton[s]:
                dict_dict_uniton[s][fuel] = np.array(df_tmp.loc[:, c].tolist())
            else:
                dict_dict_uniton[s][fuel] += np.array(df_tmp.loc[:, c].tolist())
        dict_df_uniton[s] = pd.DataFrame(dict_dict_uniton[s], columns=['NG', 'coal', 'nuclear'])

    dict_unitnon_total = dict()
    df_uniton_total_byhour = pd.DataFrame()
    for k in dict_df_uniton:
        dict_unitnon_total[k] = dict_df_uniton[k].sum(axis=1).sum(axis=0)
    for s in sname:
        df_uniton_total_byhour[s] = dict_df_uniton[s].sum(axis=1)

    plt.figure()
    ax = plt.subplot()
    x = range(len(sname))
    y = [dict_unitnon_total[k] for k in sname]
    xname = [dict_labels[s] for s in sname]
    ax.bar(x, y, color='k',alpha=0.4,)
    ax.set_ylim([9490, 9660])
    ax.set_xticks(x)
    ax.set_xticklabels(xname)

    # Wind curtailment
    list_csv = [
        os.path.sep.join([os.path.dirname(f), 'PowerGenerated.csv'])
        for f in list_csv
    ]
    dict_df_powergenerrated = dict()
    for i in range(0, len(sname)):
        s = sname[i]
        fcsv = list_csv[i]
        df_tmp = pd.read_csv(fcsv, index_col=0)
        gen_wind = [g for g in df_tmp.columns if g.startswith('wind')]
        dict_df_powergenerrated[s] = df_tmp.loc[:, gen_wind]

    dict_df_windcurtailment = dict()
    _, sdata = import_scenario_data()
    for s in sname:
        dict_tmp = dict()
        for w, t in sdata['xa'].iterkeys():
            if w not in dict_tmp:
                dict_tmp[w] = [0]*24
            i = t - 1
            dict_tmp[w][i] = sdata['xa'][w, t] - dict_df_powergenerrated[s].loc[t, w]
        dict_df_windcurtailment[s] = pd.DataFrame(dict_tmp, index=dict_df_powergenerrated[s].index)
    
    df_windcurtailment_total = pd.DataFrame()
    for s in dict_df_windcurtailment:
        df_windcurtailment_total[s] = dict_df_windcurtailment[s].sum(axis=1)

    IP()

# The following code is for Cong's one year run
################################################################################
def run_model_day(syear, smonth, sday):
    from one_year_run import read_jubeyer, return_unitont0state_fromcsv
    dir_work = 'one_year_run'
    dir_home = os.getcwd()

    t0 = time()
    if not os.path.isdir(dir_work):
        os.mkdir(dir_work)
    os.chdir(dir_work) # Go to work!
    df_power, df_load = read_jubeyer()
    gen_renewable = [g for g in df_power.columns if g.startswith('wind') or g.startswith('solar')]
    load_bus = [b for b in df_load if b.startswith('bus')]

    print '{:>20s}: {}'.format(
        'Model initiated at',
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
    )
    sys.stdout.flush()

    for solver in ['gurobi']:
        t1 = time()
        opt = SolverFactory(solver)
        y = syear
        m = smonth
        d = sday

        iselected = (df_power['Year']==y) & (df_power['Month']==m) & (df_power['Day']==d)
        df_power_tmp = df_power.loc[iselected, :].reset_index()
        dict_power = dict()
        for i, row in df_power_tmp.iterrows():
            for g in gen_renewable:
                dict_power[g, i+1] = df_power_tmp.loc[i, g]

        iselected = (df_load['Year']==y) & (df_load['Month']==m) & (df_load['Day']==d)
        df_busdemand_tmp = df_load.loc[iselected, :].reset_index()
        dict_busdemand = dict()
        for i, row in df_busdemand_tmp.iterrows():
            for b in load_bus:
                dict_busdemand[b, i+1] = df_busdemand_tmp.loc[i, b]
        df_load_tmp = df_load.loc[iselected, 'LOAD'].reset_index()
        dict_demand = dict()
        for i, row in df_load_tmp.iterrows():
            dict_demand[i+1] = df_load_tmp.loc[i, 'LOAD']

        instance = create_model() # For fairness, both solvers will create their own model
        print '{:>20s}: {}, use {:>10.2f}s'.format(
            'Model created at',
            datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
            time()-t1,
        )
        sys.stdout.flush()
        t1 = time()

        # Step 1: update power forecast and load, and parameters that are 
        # dependent on them
        instance.PowerForecast.store_values(dict_power)
        instance.BusDemand.store_values(dict_busdemand)
        instance.Demand.store_values(dict_demand)

        instance.SpinningReserveRequirement.reconstruct()
        instance.RegulatingReserveRequirement.reconstruct()

        # Step 2: Update initial minimum online/offline hours of thermal gens 
        # and parameters that are dependent on them, if not the first day (i=0)
        previousday = datetime.date(y, m, d) - datetime.timedelta(1)
        fcsv_previous = os.path.sep.join(
            [str(previousday), 'UnitOn.csv']
        )
        if os.path.isdir(fcsv_previous):
            # dict_UnitOnT0State = return_unitont0state(instance)
            dict_UnitOnT0State = return_unitont0state_fromcsv(fcsv_previous)
            instance.UnitOnT0State.store_values(dict_UnitOnT0State)
            instance.UnitOnT0.reconstruct()
            instance.InitialTimePeriodsOnLine.reconstruct()
            instance.InitialTimePeriodsOffLine.reconstruct()
            dict_PowerGeneratedT0 = dict()
            for g in instance.ThermalGenerators:
                dict_PowerGeneratedT0[g] = value(
                    instance.MinimumPowerOutput[g]*instance.UnitOnT0[g]
                )
            instance.PowerGeneratedT0.store_values(dict_PowerGeneratedT0)

        print '{:>20s}: {}, use {:>10.2f}s, solver: {}'.format(
            'Model modified at',
            datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
            time()-t1,
            solver,
        )
        sys.stdout.flush()
        t1 = time()

        # Now we can solve the UC model and save results
        results = opt.solve(instance)
        instance.solutions.load_from(results)
        # store_csvs(instance, dir_results)

        print '{:>20s}: {}, use {:>10.2f}s, solution: {:>15.2f}'.format(
            'Model solved at',
            datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
            time()-t1,
            value(instance.TotalCostObjective),
        )
        sys.stdout.flush()

    os.chdir(dir_home) # Jobs done, go home!

################################################################################
# End of Cong's one year run

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

def examine_load():
    df_load_da = pd.read_csv('/home/bxl180002/git/FlexibleRampSCUC/118bus/loads.csv', index_col=0)
    df_load_ha = pd.read_csv('/home/bxl180002/git/FlexibleRampSCUC/118bus/ha_loads.csv', index_col=['Slot'])

    df_load = pd.DataFrame(index=df_load_ha.index)
    for i in df_load.index:
        h = (i-1)/4+1
        df_load.loc[i, 'DA'] = df_load_da.loc[h, df_load_da.columns.difference(['LOAD'])].sum()
        df_load.loc[i, 'HA'] = df_load_ha.loc[i, df_load_ha.columns.difference(['LOAD'])].sum()

    IP()


if __name__ == "__main__":
    # independent_run_10_case(2012, 5, 10, True)

    t0 = time()

    # dirresults = 'results_determin_reduced_reserve'
    # if not os.path.isdir(dirresults):
    #     os.mkdir(dirresults)
    # results_done = os.listdir(dirresults)
    # os.chdir(dirresults)
    # snames, sdata = import_scenario_data()
    # snames = sdata.keys() # We would like to see the results of using xf and xa
    # for s in results_done:
    #     snames.remove(s)
    # for s in snames:
    #     solve_single_case(s)

    # t0 = time()
    # model = create_model()
    # print 'MODEL CREATED: {:>5.2f}'.format(time() - t0)
    # instance = model
    # optimizer = SolverFactory('cplex')
    # # optimizer.options['mip_cuts_gomory'] = -1
    # results = optimizer.solve(instance)
    # print 'MODEL SOLVED : {:>5.2f}'.format(time() - t0)
    # instance.solutions.load_from(results)

    # solve_after_stochastic()

    # The following code is for Cong's one year run
    ############################################################################
    # for i in range(5, 21):
    #     print "Date: {}/10".format(i)
    #     run_model_day(2011, 1, i)
    ############################################################################
    # End of Cong's one year run


    # extract_all_scenarios()

    ############################################################################
    # Sequential run starts here
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

    # This is for the 118 bus system only
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
    df_AGC_TARGET        = MyDataFrame(index=df_genfor_agc.index, columns=network.df_gen.index)
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

    # Obtain commitment statuses of slow-starting units from DAUC model
    df_uniton_ha = MyDataFrame(index=df_genfor_ha.index)
    for g, h in instance.UnitOn.iterkeys():
        if value(instance.MinimumUpTime[g]) > 1:
            v = int(value(instance.UnitOn[g, h]))
            for i in range(4*(h-1)+1, 4*h+1):
                df_uniton_ha.loc[i, g] = v

    # Create dispatch upper limits for commited slow-ramping (<1 hr) units at 
    # RTUC time scale, need to figure out a faster way, maybe combined with at 
    # the ED scale
    df_DispatchLimits_slow = MyDataFrame(index=df_uniton_ha.index)
    for g in df_uniton_ha.columns:
        r = value(instance.NominalRampDownLimit[g])/nI_ha
        rsd = value(instance.ShutdownRampLimit[g])
        pmax = value(instance.MaximumPowerOutput[g])
        for m in reversed(df_DispatchLimits_slow.index):
            tH = df_timesequence.loc[df_timesequence['tQ'] == m, 'tH'].tolist()[0]
            v = int(round(value(instance.UnitOn[ g, tH])))
            if m == df_DispatchLimits_slow.index[-1]:
                df_DispatchLimits_slow.loc[m, g] = v*pmax
            else:
                tH_next = df_timesequence.loc[df_timesequence['tQ'] == m+1, 'tH'].tolist()[0]
                v_next = int(round(value(instance.UnitOn[g, tH_next])))
                df_DispatchLimits_slow.loc[m, g] = min(
                    v*((v-v_next)*rsd) + v_next*(df_DispatchLimits_slow.loc[m+1, g]+r), 
                    pmax
                )

    # Create dispatch upper limits for commited slow-ramping (<1 hr) units at
    # the RTED time scale, need to figure out a faster way
    df_DispatchLimits_slow_ED = MyDataFrame(index=df_timesequence['t5'])
    for g in df_uniton_ha.columns:
        r = value(instance.NominalRampDownLimit[g])/nI_ed
        rsd = value(instance.ShutdownRampLimit[g])
        pmax = value(instance.MaximumPowerOutput[g])
        for m in reversed(df_DispatchLimits_slow_ED.index):
            tH = df_timesequence.loc[df_timesequence['t5'] == m, 'tH'].tolist()[0]
            v = int(round(value(instance.UnitOn[ g, tH])))
            if m == df_DispatchLimits_slow_ED.index[-1]:
                df_DispatchLimits_slow_ED.loc[m, g] = v*pmax
            else:
                tH_next = df_timesequence.loc[df_timesequence['t5'] == m+1, 'tH'].tolist()[0]
                v_next = int(round(value(instance.UnitOn[g, tH_next])))
                df_DispatchLimits_slow_ED.loc[m, g] = min(
                    v*((v-v_next)*rsd) + v_next*(df_DispatchLimits_slow_ED.loc[m+1, g]+r), 
                    pmax
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
            df_uniton_ha.loc[t_start: t_end, :].T
        ).to_dict_2d()

        dict_DispacthLimitsUpper = MyDataFrame(
            df_DispatchLimits_slow.loc[t_start: t_end, :].T
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
            dict_DispacthLimitsUpper
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

        # Create dispatch upper limits for ALL commited units
        df_DispatchLimits = MyDataFrame(
            index=[
                df_timesequence.loc[i, 't5']
                for i in df_timesequence.index
                if df_timesequence.loc[i, 'tQ'] in range(t_start, t_end+1)
            ]
        ) # Include all ED intervals in the current RTUC run
        for g in ins_ha.ThermalGenerators:
            if g in df_DispatchLimits_slow_ED.columns: # Slow units, read directly from the previous results
                df_DispatchLimits[g] = df_DispatchLimits_slow_ED.loc[df_DispatchLimits.index, g]
            else: # Fast units, we need to calculate, how to save time on this one?
                r = value(instance.NominalRampDownLimit[g])/nI_ed
                rsd = value(instance.ShutdownRampLimit[g])
                pmax = value(instance.MaximumPowerOutput[g])
                for m in reversed(df_DispatchLimits.index):
                    tQ = df_timesequence.loc[df_timesequence['t5'] == m, 'tQ'].tolist()[0]
                    v = int(round(value(ins_ha.UnitOn[g, tQ])))
                    if m == df_DispatchLimits.index[-1]:
                        df_DispatchLimits.loc[m, g] = v*pmax
                    else:
                        tQ_next = df_timesequence.loc[df_timesequence['t5'] == m+1, 'tQ'].tolist()[0]
                        v_next = int(round(value(ins_ha.UnitOn[g, tQ_next])))
                        df_DispatchLimits.loc[m, g] = min(
                            v*((v-v_next)*rsd) + v_next*(df_DispatchLimits.loc[m+1, g]+r), 
                            pmax
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
                df_DispatchLimits.loc[t_start_ed: t_end_ed, :].T
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
                    df_agc_tmp.loc[:, 'ED_DISPATCH_NEXT'] 
                    - df_ACTUAL_GENERATION.loc[int( (t_start_ed-1)*nI_agc/nI_ed+1 ), :]
                )/(nI_agc/nI_ed) # Maybe the calculation of RTED step out of the AGC loop?

                # Then, determine AGC movement for AGC responding units, we 
                # follow FESTIV's option 2, where each unit's deployed 
                # regulation is proportional to its regulation bid into the RTED market
                df_agc_tmp.loc[i_reg_up_units, 'AGC_MOVE'] = pd.concat(
                    [
                        df_agc_tmp.loc[:, 'REG_UP_AGC'],
                        -df_agc_tmp.loc[:, 'ACE_TARGET']*df_agc_tmp.loc[:, 'REG_UP_AGC']/sum_reg_up_agc
                    ],
                    axis=1,
                ).min(axis=1)
                df_agc_tmp.loc[i_reg_dn_units, 'AGC_MOVE'] = pd.concat(
                    [
                        -df_agc_tmp.loc[:, 'REG_DN_AGC'],
                        -df_agc_tmp.loc[:, 'ACE_TARGET']*df_agc_tmp.loc[:, 'REG_DN_AGC']/sum_reg_dn_agc
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
                df_AGC_SCHEDULE.loc[t_AGC, :] = df_agc_tmp.loc[:, 'AGC_BASEPOINT']
                df_AGC_MOVE.loc[t_AGC, :]     = df_agc_tmp.loc[:, 'AGC_MOVE']
                df_AGC_TARGET.loc[t_AGC, :]   = df_agc_tmp.loc[:, 'ACE_TARGET']

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

            # This is for debugging
            # if t_start_ed == 4:
            #     IP()

        # Extract initial parameters from the binding interval of the last ED run for the next RTUC run
        dict_UnitOnT0State = return_unitont0state(ins_ha, ins_ha.TimePeriods.first())
        # dict_PowerGeneratedT0 = return_powergenerated_t(ins_ha, ins_ha.TimePeriods.first())
        dict_PowerGeneratedT0 = dict_PowerGeneratedT0_ed

    df_ACE = pd.DataFrame(dict_ACE)
    
    # Write all results
    # df_ACE.to_csv('ACE.csv', index=False)
    # df_ACTUAL_GENERATION.to_csv('ACTUAL_GENERATION.csv')
    # df_AGC_SCHEDULE.to_csv('AGC_SCHEDULE.csv')
    # df_AGC_TARGET.to_csv('AGC_TARGET.csv')
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