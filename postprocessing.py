import os, pandas as pd
from pyomo.environ import *
from matplotlib import gridspec, pyplot as plt
from IPython import embed as IP

color_map = {
    'Natural gas': [0.7, 0.7, 0.7],
    'Coal':        [0.0, 0.0, 0.0],
    'Nuclear':     [0.6, 0.0, 0.8],
    'Solar':       [1.0, 1.0, 0.0],
    'Wind':        [0.0, 0.0, 1.0],
    'Hydro':       [0.4, 0.6, 0.9],
    }


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

    vars_0dim = list()
    vars_1dim = list()
    vars_2dim = list()
    vars_3dim = list()
    for v in instance.component_objects(Var, active=True):
        vdim = getattr( getattr(instance, v.local_name), 'dim')
        if vdim() == 0:
            vars_0dim.append(v.local_name)
        elif vdim() == 1:
            vars_1dim.append(v.local_name)
        elif vdim() == 2:
            vars_2dim.append(v.local_name)
        elif vdim() == 3:
            vars_3dim.append(v.local_name)
    # print vars_0dim
    # print vars_1dim
    # print vars_2dim
    # vars_2dim = [
    #     'FlexibleRampUpAvailable',
    #     'FlexibleRampDnAvailable',
    #     'RegulatingReserveUpAvailable',
    #     'RegulatingReserveDnAvailable',
    #     'SpinningReserveUpAvailable',
    #     'UnitOn',
    #     'PowerGenerated',
    #     'MaximumPowerAvailable',
    #     'MaxWindAvailable',
    #     'ProductionCost',
    #     'StartupCost',
    #     'ShutdownCost',
    #     'BusCurtailment', # Note BusCurtailment is indexed by bus and time period
    # ]
    for v_ij in vars_2dim:
        df_v_ij = extract_var_2dim(instance, v_ij)
        csvname = '.'.join([v_ij, 'csv'])
        df_v_ij.to_csv(csvname)
    
    # Note that variables indexed by only 1 index are normally indexed by time 
    # period.
    # vars_1dim = [
    #     'Curtailment',
    #     'SpinningReserveShortage',
    #     'RegulatingReserveShortage',
    #     'FlexibleRampUpShortage',
    #     'FlexibleRampDnShortage',
    #     'RampingCost',
    # ]
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

    # vars_0dim = [
    #     'TotalProductionCost',
    #     'TotalFixedCost',
    #     'TotalCurtailment',
    #     'TotalCurtailmentCost',
    #     'TotalCostObjective', # Note this is the objective function
    # ]
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

    # vars_3dim = [
    #     'BlockPowerGenerated',
    # ]
    
    os.chdir(dirhome)

def plot_power(ax, df_power_by_type, title=None):
    b = [0]*df_power_by_type.index.size
    handles = list()
    for i in df_power_by_type.columns:
        h = ax.fill_between(
            df_power_by_type.index,
            b,
            b+df_power_by_type[i]/1E3, # Convert to GW
            facecolor=color_map[i],
        )
        handles.append(h)
        b += df_power_by_type[i]/1E3
    ymax = max(b) - int( max(b) )%10 + 20
    ax.set_xlim([1, df_power_by_type.index.size])
    ax.set_ylim([0, ymax])
    # ax.set_xlabel('Time (h)')
    # ax.set_ylabel('Power (GW)')
    if title:
        ax.annotate(
            title,
            xy=(0.25, 0.9), xycoords='axes fraction',
            xytext=(.25, .9), textcoords='axes fraction',
            fontsize=10,
        )
    return handles, df_power_by_type.columns

def calculate_power_by_type(csvname):
    # csvname = 'PowerGenerated.csv'
    df_power_by_gen = pd.read_csv(csvname, index_col=0)
    resource_types = [
        'Natural gas',
        'Coal',
        'Nuclear',
        'Hydro',
        'Wind',
        'Solar',
    ]
    col = dict()
    col['Natural gas'] = [i for i in df_power_by_gen.columns if i.startswith('ng')]
    col['Coal']        = [i for i in df_power_by_gen.columns if i.startswith('coal')]
    col['Nuclear']     = [i for i in df_power_by_gen.columns if i.startswith('nuclear')]
    col['Hydro']       = [i for i in df_power_by_gen.columns if i.startswith('hydro')]
    col['Wind']        = [i for i in df_power_by_gen.columns if i.startswith('wind')]
    col['Solar']       = [i for i in df_power_by_gen.columns if i.startswith('solar')]
    dict_of_series = dict()
    for r in resource_types:
        dict_of_series[r] = df_power_by_gen[ col[r] ].sum(axis=1)
    df_power_by_type = pd.DataFrame( dict_of_series )
    return df_power_by_type

def read_determine(dircsv):
    snames = ['N' + str(i) + 'Scenario' for i in range(1, 11)]
    ReserveFactor = 0.1
    RegulatingReserveFactor = 0.05
    csv_load  = '/home/bxl180002/git/FlexibleRampSCUC/TEXAS2k_B/loads.csv'

    df_load  = pd.read_csv(csv_load,  index_col=0)
    cbus = [c for c in df_load.columns if c.startswith('bus')]
    df_load['SUMBUS'] = df_load.loc[:, cbus].sum(axis=1)
    df_load['SPNUP']  = df_load['SUMBUS']*ReserveFactor
    df_load['REGUP']  = df_load['SUMBUS']*RegulatingReserveFactor
    df_load['REGDN']  = df_load['SUMBUS']*RegulatingReserveFactor

    plt.plot(
        df_load.index, df_load['SUMBUS'], 'k+',
        df_load.index, df_load['SUMBUS'] + df_load['SPNUP'], 'r',
        df_load.index, df_load['SUMBUS'] + df_load['REGUP'], 'b',
        df_load.index, df_load['SUMBUS']-1*df_load['REGDN'], 'b',
    )

    for s in snames:
        dirscenario_ed = os.path.sep.join([dircsv, s, 'ED'])
        csv_regup = os.path.sep.join([dirscenario_ed, 'RegulatingReserveUpAvailable.csv'])
        csv_regdn = os.path.sep.join([dirscenario_ed, 'RegulatingReserveDnAvailable.csv'])
        csv_spnup = os.path.sep.join([dirscenario_ed, 'SpinningReserveUpAvailable.csv'])
        csv_power = os.path.sep.join([dirscenario_ed, 'PowerGenerated.csv'])

        df_spnup = pd.read_csv(csv_spnup, index_col=0)
        df_regup = pd.read_csv(csv_regup, index_col=0)
        df_regdn = pd.read_csv(csv_regdn, index_col=0)
        df_power = pd.read_csv(csv_power, index_col=0)

        diff_spnup = df_load['SPNUP'] - df_spnup.sum(axis=1)
        diff_regup = df_load['REGUP'] - df_regup.sum(axis=1)
        diff_regdn = df_load['REGDN'] - df_regdn.sum(axis=1)

        plt.plot(
            df_load.index, df_load['SUMBUS'] + df_spnup.sum(axis=1), 'k',
            df_load.index, df_power.sum(axis=1), 'g',
        )
    plt.show()

if __name__ == "__main__":
    # dir_data = "C:\\Users\\bxl180002\\Downloads\\results_10scenario_independent"
    # scenarios = [
    #     "Q10Scenario",
    #     "Q20Scenario",
    #     "Q30Scenario",
    #     "Q40Scenario",
    #     "Q50Scenario",
    #     "Q60Scenario",
    #     "Q70Scenario",
    #     "Q80Scenario",
    #     "Q90Scenario",
    # ]
    # fig = plt.figure(figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')
    # gs = gridspec.GridSpec(3, 3, hspace=0.05, wspace=0.05)
    # for s in scenarios:
    #     csvname = os.path.sep.join([dir_data, s, 'PowerGenerated.csv'])
    #     df_power_by_type = calculate_power_by_type(csvname)
    #     # fig = plt.figure()
    #     # ax = plt.subplot(111)
    #     ax = plt.subplot( gs[scenarios.index(s)] )
    #     handles, names = plot_power(ax, df_power_by_type, title=s)
    #     if scenarios.index(s)%3 == 0:
    #         ax.set_ylabel('Power (GW)')
    #     else:
    #         plt.setp(ax.get_yticklabels(), visible=False)
    #     if scenarios.index(s)>5:
    #         ax.set_xlabel('Time (h)')
    #     else:
    #         plt.setp(ax.get_xticklabels(), visible=False)
    # plt.subplots_adjust(left=0.10, right=0.90, top=0.99, bottom=0.15)
    # fig.legend(
    #         handles, names, 
    #         loc='lower center', 
    #         ncol=len(names)/2, 
    #         bbox_to_anchor=(0.5, 0.02),
    #         edgecolor=None
    #     )

    # plt.show()

    dircsv = '/home/bxl180002/git/FlexibleRampSCUC/results_determin'
    read_determine(dircsv)
