import os, pandas as pd
from IPython import embed as IP

def import_scenario_data(print_error=False):

    # File locations
    dir_TX2kB = os.path.sep.join( ['.',       'TEXAS2k_B'] )
    f_gen     = os.path.sep.join( [dir_TX2kB, 'generator_data_plexos_withRT.csv'] )
    f_genfor  = os.path.sep.join( [dir_TX2kB, 'generator.csv'] )
    f_genwind = os.path.sep.join( [dir_TX2kB, 'wind_generator_data.csv'] )
    dir_scenario = '/home/bxl180002/git/WindScenarioGeneration/scenario_data'

    # Names of scenarios, and names of columns of scenarios
    scenario_col_names = [
        'N1',
        'N2',
        'N3',
        'N4',
        'N5',
        'N6',
        'N7',
        'N8',
        'N9',
        'N10',
    ]
    scenario_names = [i + 'Scenario' for i in scenario_col_names]

    # Get the wind generator names in the TX2kB system
    df_gen = pd.read_csv(f_gen, index_col=0)
    wind_generator_names  =  [ x for x in df_gen.index if x.startswith('wind') ]

    # Generator name in the TX2kB system -> WIND Toolkit name
    df_genwind = pd.read_csv(f_genwind, index_col=0)
    map_wind2site = pd.Series(df_genwind.SITE_ID.values, index=df_genwind.index)

    # Read Kwami's power forecast for renewables
    df_genfor = pd.read_csv(f_genfor)

    # Now we start to extract forecasted wind power
    WindPowerForecast = dict()
    if print_error:
        print '{:<7s}  {:>7s}  {:>12s}  {:>9s}  {:>14s}'.format(
            'TX ID', 'WIND ID', 'OUTPUT SCALE', 'CAP SCALE', 'DELTA SUM (MW)'
        )
    for w in wind_generator_names:
        WIND_ID = map_wind2site[w]
        fname = os.path.sep.join(
            [
                dir_scenario,
                str(WIND_ID)+'.csv',
            ]
        )
        df_tmp = pd.read_csv(fname)

        # The following block is to find the scaling factor Kwami's using for 
        # the wind generator capacity
        actual = df_tmp.loc[:, 'xa'].reset_index(drop=True)
        for i in range(0, 24):
            if abs(df_genfor.loc[i, w] - 0)>=1E-3:
                break # Assure the denominator, df_genfor.loc[i, w], > 0
        output_scaler = df_genfor.loc[i, w]/actual[i]
        cap_scaler    = df_gen.loc[w, 'PMAX']/df_genwind.loc[w, 'SITE_CAP']
        delta  = actual.subtract(df_genfor.loc[:, w]/output_scaler)
        if print_error:
            print '{:<7s}  {:7d}  {:>12.2f}  {:>9.2f}  {:>14.2f}'.format(
                w, WIND_ID, output_scaler, cap_scaler, sum(delta)
            )

        for i in range(0, len(scenario_col_names)):
            c = scenario_col_names[i]
            s = scenario_names[i]
            if s not in WindPowerForecast:
                WindPowerForecast[s] = dict()
            for h in range(1, 25): # The time index in the UC model is from 1 to 24
                tmp_data = df_tmp.loc[h-1, c] # Hopefully there is only one element
                WindPowerForecast[s][w, h] = output_scaler*tmp_data

        # xa and xf are the actual and deterministic forecasted data
        for c in ['xa', 'xf']:
            s = c
            if c not in WindPowerForecast:
                WindPowerForecast[c] = dict()
            for h in range(1, 25):
                tmp_data = df_tmp.loc[h-1, c]
                WindPowerForecast[s][w, h] = output_scaler*tmp_data
    print "Scenario data created!"
    return scenario_names, WindPowerForecast

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
    for W in W2w:
        w = W2w[W]
        for s in PowerForecastWind_w:
            if s not in PowerForecastWind_W:
                PowerForecastWind_W[s] = dict()
            for h in range(1, 25):
                PowerForecastWind_W[s][W, h] = PowerForecastWind_w[s][w, h]
    return PowerForecastWind_W


if __name__ == "__main__":
    scenario_names, WindPowerForecast = import_scenario_data(True)
    IP()