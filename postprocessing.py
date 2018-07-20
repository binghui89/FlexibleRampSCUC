import os, pandas as pd
from pyomo.environ import *

def extract_var_2dim(instance, varname):
    # This function is used to extract variables indexed by 2-dim tuple with 
    # the following format: (generator, time): value.
    tmp_col = set()
    tmp_var = getattr(instance, varname)
    for k, v in tmp_var.iteritems():
        gen, t = k
        tmp_col.add(gen)
    var = pd.DataFrame(
        index=instance.TimePeriods.values(),
        columns=list(tmp_col),
    )
    for k, v in tmp_var.iteritems():
        gen, t = k
        var.at[t, gen] = value(v)
    return var # This is a pandas DataFrame object

def extract_var_1dim(instance, varname):
    # This function is used to extract variables indexed by only one element and 
    # returns a list.
    list_index = list()
    list_var   = list()
    tmp_var = getattr(instance, varname)
    for k, v in tmp_var.iteritems():
        list_index.append(k)
        list_var.append(value(v))
    return list_index, list_var

def store_csvs(instance, dirwork = None):
    # These variables are indexed with two variables, normally, they are 
    # generator and time period.
    dirhome = os.getcwd()
    if dirwork:
        if not os.path.isdir(dirwork):
            os.mkdir(dirwork)
        os.chdir(dirwork)

    vars_2dim = [
        'FlexibleRampUpAvailable',
        'FlexibleRampDnAvailable',
        'RegulatingReserveUpAvailable',
        'RegulatingReserveDnAvailable',
        'SpinningReserveUpAvailable',
        'UnitOn',
        'PowerGenerated',
        'MaximumPowerAvailable',
        'MaxWindAvailable',
        'ProductionCost',
        'StartupCost',
        'ShutdownCost',
        'BusCurtailment', # Note BusCurtailment is indexed by bus and time period
    ]
    for v_ij in vars_2dim:
        df_v_ij = extract_var_2dim(instance, v_ij)
        csvname = '.'.join([v_ij, 'csv'])
        df_v_ij.to_csv(csvname)
    
    # Note that variables indexed by only 1 index are normally indexed by time 
    # period.
    vars_1dim = [
        'Curtailment',
        'SpinningReserveShortage',
        'RegulatingReserveShortage',
        'FlexibleRampUpShortage',
        'FlexibleRampDnShortage',
        'RampingCost',
    ]
    dict_v_i = dict()
    for v_i in vars_1dim:
        list_index, list_v = extract_var_1dim(instance, v_i)
        dict_v_i[v_i] = list_v
    dict_v_i['time period'] = list_index
    df_v_i = pd.DataFrame.from_dict(dict_v_i)
    df_v_i.to_csv(
        '1dim_var.csv',
        index=False,
        columns=['time period'] + vars_1dim,
    )

    vars_0dim = [
        'TotalProductionCost',
        'TotalFixedCost',
        'TotalCurtailment',
        'TotalCurtailmentCost',
        'TotalCostObjective', # Note this is the objective function
    ]
    list_v = list()
    for vname in vars_0dim:
        tmp_v = value( getattr(instance, vname) )
        list_v.append(tmp_v)
    dict_v = {
        'Names':  vars_0dim,
        'Values': list_v,
    }
    df_v = pd.DataFrame.from_dict(dict_v)
    df_v.to_csv('0dim_var.csv', index=False)

    vars_3dim = [
        'BlockPowerGenerated',
    ]
    
    os.chdir(dirhome)
