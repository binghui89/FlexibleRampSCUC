import os, sys, platform, datetime, smtplib, multiprocessing, pandas as pd, numpy as np, matplotlib
from time import time
from pyomo.opt import SolverFactory
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
    # Find number of online/offline time intervals of thermal gens at period t
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
                t_on  = b*(t_on + b) # Number of the last consecutive online hours
                t_off = (1-b)*(t_off + 1 - b) # Number of the last consecutive offline hours
        dict_results[g] = int(round(sign(t_on)*t_on - sign(t_off)*t_off)) # This is an integer?
    return dict_results

def return_powergenerated_t(instance, t=None):
    # Find power generation levels at period t
    if not t:
        t = instance.TimePeriods.last()
    elif t not in instance.TimePeriods:
        print "WARNING: NO POWER_GENERATED CREATED."
        return None
    dict_results = dict()
    for g in instance.ThermalGenerators.iterkeys():
        dict_results[g] = value(instance.PowerGenerated[g, t])
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

    # content = 'UC model test.\n'
    # if content:
    #     content += '\n'
    #     content += 'Time consumption: {:>.2f} s'.format(time() - t0)
    #     if len(sys.argv) > 1:
    #         mail_pass = sys.argv[1]
    #         mailto_list = ['reagan.fruit@gmail.com']
    #         # content = "The model run is completed!"
    #         # if os.path.isfile('lab.out'):
    #         #     with open('lab.out', 'r') as myfile:
    #         #         content = myfile.readlines()
    #         #         content = ''.join(content)
    #         if send_mail(
    #             mailto_list,
    #             "Model run completed.",
    #             content,
    #             mail_pass
    #         ):
    #             print "Email sent successfully!"  
    #         else:  
    #             print "Email sent failed."

    # extract_all_scenarios()

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
    # csv_busload_ha        = '/home/bxl180002/git/FlexibleRampSCUC/ha_load.csv'
    # csv_genfor_ha         = '/home/bxl180002/git/FlexibleRampSCUC/ha_generator.csv'

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

    df_timesequence = pd.DataFrame(columns=['tH', 'tQ', 't5'], dtype='int')
    for i in range(1, 25):
        for j in range(1, 5):
            for k in range(1,4):
                index = (i-1)*12+(j-1)*3 + k
                df_timesequence.loc[index, 'tH'] = i
                df_timesequence.loc[index, 'tQ'] = (i-1)*4 + j
                df_timesequence.loc[index, 't5'] = index
    df_timesequence = df_timesequence.astype('int')

    network = Network(csv_bus, csv_branch, csv_ptdf, csv_gen, csv_marginalcost, csv_blockmarginalcost, csv_blockoutputlimit)
    network.df_bus['VOLL'] = 9000

    df_busload = pd.read_csv(csv_busload, index_col=0)
    df_genfor  = pd.read_csv(csv_genfor, index_col=0)
    df_busload = MyDataFrame(df_busload.loc[:, df_busload.columns.difference(['LOAD'])])
    df_genfor  = MyDataFrame(df_genfor)
    df_genfor.index = range(1, 25) # Kwami's convention: time starts from 1...
    df_genfor_nonthermal = df_genfor.loc[:, network.df_gen[network.df_gen['GEN_TYPE']!='Thermal'].index]
    df_genfor_nonthermal.fillna(0, inplace=True)

    ReserveFactor = 0.1
    RegulatingReserveFactor = 0.05

    model = create_model(
        network,
        df_busload,
        df_genfor_nonthermal,
        ReserveFactor,
        RegulatingReserveFactor,
        nI=1,
    )
    print('Model created at: {:>.2f} s'.format(time() - t0))

    instance = model
    optimizer = SolverFactory('cplex')
    results = optimizer.solve(instance)
    instance.solutions.load_from(results)
    print(
            'Model solved at: {:>.2f} s, objective: {:>.2f}'.format(
                time() - t0,
                value(instance.TotalCostObjective)
            )
        )

    # Start RTUC
    nI_ha = 4 # Number of intervals in a hour in the RTUC run
    nI_ed = 12 # Number of intervals in a hour in the RTED run

    # Prepare hourly ahead data
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

    # Obtain commitment statuses of slow-starting units from DAUC model
    df_uniton_da = MyDataFrame(index=df_genfor_ha.index)
    for g, h in instance.UnitOn.iterkeys():
        if value(instance.MinimumUpTime[g]) > 1:
            v = int(value(instance.UnitOn[g, h]))
            for i in range(4*(h-1)+1, 4*h+1):
                df_uniton_da.loc[i, g] = v

    # RTUC initiation parameters
    dict_UnitOnT0State       = None
    dict_PowerGeneratedT0    = None
    dict_PowerGeneratedT0_ed = None

    # network.df_gen.loc[network.df_gen['MINIMUM_DOWN_TIME']==1, 'MINIMUM_DOWN_TIME'] = 0.25
    # network.df_gen.loc[network.df_gen['MINIMUM_UP_TIME']==1, 'MINIMUM_UP_TIME'] = 0.25

    # Start sequential run
    ls_instance = list()
    for i_rtuc in range(1, 94):
        t_start = i_rtuc # 4*(i_rtuc-1) + 1
        t_end   = i_rtuc + 3 # 4*i_rtuc
    # for i_rtuc in range(1, 25):
    #     t_start = 4*(i_rtuc-1) + 1
    #     t_end   = 4*i_rtuc

        dict_uniton_da = MyDataFrame(
            df_uniton_da.loc[t_start: t_end, :].T
        ).to_dict_2d()

        # Create RTUC model
        i = create_model(
            network,
            df_busload_ha.loc[t_start: t_end, :],
            df_genfor_ha.loc[t_start: t_end, :],
            ReserveFactor,
            RegulatingReserveFactor,
            nI_ha,
            dict_UnitOnT0State, 
            dict_PowerGeneratedT0,
            dict_uniton_da,
        )
        print "RTUC Model {} created!".format(i_rtuc)

        # Solve RTUC model
        results = optimizer.solve(i)
        i.solutions.load_from(results)
        print(
            'RTUC Model {} solved at: {:>.2f} s, objective: {:>.2f}, penalty: {:s}'.format(
                i_rtuc, 
                time() - t0,
                value(i.TotalCostObjective),
                'N/A' # value(i.SlackPenalty)
            )
        )
        ls_instance.append(i)

        # List of time intervals in the ED run
        ls_t_ed = df_timesequence.loc[df_timesequence['tQ']==t_start, 't5'].tolist()
        for t_start_ed in ls_t_ed:
            # Obtain upper and lower dispatch limits from the RTUC model
            dict_UpperDispatchLimit, dict_LowerDispatchLimit = dispatch_limits(i, [t_start_ed], df_timesequence.astype('int'))

            # Obtain commitment statuses of slow-starting units from DAUC model
            dict_uniton_ha = dict()
            for t5 in [t_start_ed]:
                tQ = df_timesequence.loc[df_timesequence['t5'] == t5, 'tQ'].tolist()[0]
                for g in i.ThermalGenerators.iterkeys():
                    dict_uniton_ha[g, t5] = value(i.UnitOn[g, tQ])

            # Create ED model
            instance_ed = build_sced_model(
                network,
                df_busload_ed.loc[df_busload_ed.index==t_start_ed, :], # Only bus load, first dimension time starts from 1, no total load. For ED model, there should be only 1 row
                df_genfor_ed.loc[df_busload_ed.index==t_start_ed, :], # Only generation from nonthermal gens, first dim time starts from 1
                ReserveFactor,
                RegulatingReserveFactor,
                nI_ed, # Number of intervals in an hour, typically DAUC: 1, RTUC: 4
                None, # Not required for ED
                dict_PowerGeneratedT0_ed, # How many time periods the units have been on at T0 from last RTUC model
                dict_uniton_ha, # Initial power generation level at T0 from last RTUC model
                #########
                # The following parameters only apply for the ED model
                dict_UpperDispatchLimit,
                dict_LowerDispatchLimit,
            )

            # Solve the model
            results_ed = optimizer.solve(instance_ed)
            instance_ed.solutions.load_from(results)
            print (
                '    RTED Model {} solved at: {:>.2f} s, objective: {:>.2f}, penalty: {:s}'.format(
                    t_start_ed, 
                    time() - t0,
                    value(instance_ed.TotalCostObjective),
                    'N/A' # value(i.SlackPenalty)
                )
            )

            # Extract initial parameters for the next run
            dict_PowerGeneratedT0_ed = return_powergenerated_t(
                instance_ed, instance_ed.TimePeriods.last()
            )

        # Extract initial parameters for the next RTUC run
        dict_UnitOnT0State    = return_unitont0state(i, i.TimePeriods.first())
        # dict_PowerGeneratedT0 = return_powergenerated_t(i, i.TimePeriods.first())
        dict_PowerGeneratedT0 = dict_PowerGeneratedT0_ed
        # IP()
        # if t == 12:
        #     IP()
