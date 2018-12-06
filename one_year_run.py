import os, sys, platform, datetime, smtplib, multiprocessing, pandas as pd, numpy as np
from time import time
from pyomo.opt import SolverFactory
from postprocessing import store_csvs
from pyomo.environ import *
from helper import import_scenario_data, extract_uniton
from email.mime.text import MIMEText
from numpy import sign
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


# Supress output when building the model
f = open(os.devnull, 'w')
sys.stdout = f
from SCUC_RampConstraint_3 import create_model
sys.stdout = sys.__stdout__
f.close()

# The following code is for Cong's one year run
################################################################################
def read_jubeyer():
    csvload  = '/home/bxl180002/git/FlexibleRampSCUC/Jubeyer/Texas_2k_load.csv'
    csvpower = '/home/bxl180002/git/FlexibleRampSCUC/Jubeyer/Solar_Wind_Data_Texas_2k_Bus.csv'
    df_load  = pd.read_csv(csvload, index_col=0)
    df_power = pd.read_csv(csvpower, index_col=0)
    T = df_load.shape[0]
    trange = pd.date_range('1/1/2011', periods=T, freq='H')
    df_load['Year']   = trange.year
    df_load['Month']  = trange.month
    df_load['Day']    = trange.day
    df_load['Hour']   = trange.hour
    df_power['Year']  = trange.year
    df_power['Month'] = trange.month
    df_power['Day']   = trange.day
    df_power['Hour']  = trange.hour

    return df_power, df_load

def return_unitont0state(instance):
    # Find the online/offline hours of thermal gens at t = 0
    dict_results = dict()
    for g in instance.ThermalGenerators.iterkeys():
        t_on  = 0
        t_off = 0
        for t in instance.TimePeriods.iterkeys():
            b = value(instance.UnitOn[g, t])
            t_on  = b*(t_on + b) # Number of the last consecutive online hours
            t_off = (1-b)*(t_off + 1 - b) # Number of the last consecutive offline hours
        dict_results[g] = int(round(sign(t_on)*t_on - sign(t_off)*t_off)) # This is an integer?
    return dict_results

def return_unitont0state_fromcsv(fcsv):
    df_uniton = pd.read_csv(fcsv, index_col=0)
    dict_results = dict()
    for g in df_uniton.columns:
        t_on  = 0
        t_off = 0
        for i in df_uniton.index:
            b = df_uniton.loc[i, g]
            t_on  = b*(t_on + b) # Number of the last consecutive online hours
            t_off = (1-b)*(t_off + 1 - b) # Number of the last consecutive offline hours
        dict_results[g] = int(round(sign(t_on)*t_on - sign(t_off)*t_off))
    return dict_results

def run_one_year():
    T = 60
    alldays = pd.date_range('1/1/2011', periods=T, freq='D')
    dir_work = 'one_year_run'
    dir_home = os.getcwd()
    print 'Model run started at {}, T = {}'.format(
        datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
        T,
    )
    sys.stdout.flush()

    t0 = time()
    if not os.path.isdir(dir_work):
        os.mkdir(dir_work)
    os.chdir(dir_work) # Go to work!
    df_power, df_load = read_jubeyer()
    instance = create_model()
    opt = SolverFactory('cplex')
    gen_renewable = [g for g in df_power.columns if g.startswith('wind') or g.startswith('solar')]
    load_bus = [b for b in df_load if b.startswith('bus')]
    print '{:^8s}  {:^11s}  {:>15s}  {:>15s}  {:^22s}  {:^15s}'.format(
        '# of day',
        'Date',
        'T iter (s)',
        'T total (h)',
        'Finished at',
        'Objective value',
    )
    sys.stdout.flush()
    for iday in range(0, len(alldays)):
        t1 = time()
        thisday = alldays[iday]
        y = thisday.year
        m = thisday.month
        d = thisday.day
        dir_results = str(thisday.to_pydatetime().date())
        try:
            fcsv = os.path.sep.join(
                [dir_results, 'UnitOn.csv']
            )
            df = pd.read_csv(fcsv)
            solved = True # This case has already been solved.
        except IOError:
            solved = False # This case has not been solved yet.

        if solved:
            print '{:>8s}  {:>11s}  {:>15.2f}  {:>15.2f}  {:>22s}  {:>15s}'.format(
                str(iday+1)+'/'+str(T),
                str(thisday.to_pydatetime().date()),
                time() - t1,
                (time() - t0)/3600,
                datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
                'N/A'
            )
            continue

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

        # Step 1: update power forecast and load, and parameters that are 
        # dependent on them
        instance.PowerForecast.store_values(dict_power)
        instance.BusDemand.store_values(dict_busdemand)
        instance.Demand.store_values(dict_demand)

        instance.SpinningReserveRequirement.reconstruct()
        instance.RegulatingReserveRequirement.reconstruct()

        # Step 2: Update initial minimum online/offline hours of thermal gens 
        # and parameters that are dependent on them, if not the first day (i=0)
        if iday:
            # dict_UnitOnT0State = return_unitont0state(instance)
            previousday = alldays[iday-1]
            fcsv_previous = os.path.sep.join(
                [str(previousday.to_pydatetime().date()), 'UnitOn.csv']
            )
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

        # Now we can solve the UC model and save results
        results = opt.solve(instance)
        instance.solutions.load_from(results)
        store_csvs(instance, dir_results)

        print '{:>8s}  {:>11s}  {:>15.2f}  {:>15.2f}  {:>22s}  {:>15.2f}'.format(
            str(iday+1)+'/'+str(T),
            str(thisday.to_pydatetime().date()),
            time() - t1,
            (time() - t0)/3600,
            datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M:%S'),
            value(instance.TotalCostObjective),
        )
        sys.stdout.flush()

    os.chdir(dir_home) # Jobs done, go home!

################################################################################
# End of Cong's one year run

if __name__ == "__main__":

    t0 = time()

    run_one_year()

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
