%% Load actual 5-min data

t0 = datetime(2017, 12, 31, 23, 05, 00);
timestep = h5read('Kate/5-minute_load_actuals.h5', '/Timestep');
t5 = t0 + [1:105120]'.*duration(0, 5, 0);
load_actual = h5read('Kate/5-minute_load_actuals.h5', '/ERCOT');


%% Load DA hourly forecast

leadtime_da_h5 = repmat(duration(1, 0, 0).*[1:36], 365, 1) + duration(11, 30, 0) - duration(1, 0, 0); 
issuetime_da_h5 = datetime(2017, 12, 31, 12, 30, 0) + duration(24, 0, 0).*[0:1:364]';
timestamp_da_h5 = repmat(issuetime_da_h5, 1, size(leadtime_da_h5, 2)) + leadtime_da_h5;
load_da_h5 = h5read('Kate/day_ahead_load_forecast.h5', '/ERCOT');

load_da = reshape(load_da_h5(:, 1:24)', size(load_da_h5, 1)*24, 1);
timestamp_da = reshape(timestamp_da_h5(:, 1:24)', size(timestamp_da_h5, 1)*24, 1);

%% Load intro-hour 5-min forecast
issuetime_ha_h5 = datetime(2017, 12, 31, 23, 0, 0) + duration(0, 5, 0).*[0:1:105120-1]';
leadtime_ha_h5 = repmat(duration(0, 5, 0).*[1:24], 105120, 1);
timestamp_ha_h5 = repmat(issuetime_ha_h5, 1, size(leadtime_ha_h5, 2)) + leadtime_ha_h5;
load_ha_h5 = h5read('Kate/intra-hourly_load_forecast.h5', '/ERCOT');

% Let's use hour-ahead forecast
timestamp_ha = timestamp_ha_h5(:, 12);
load_ha = load_ha_h5(:, 12);

%% Now, distribute total load to bus level
addpath('/home/bxl180002/git/matpower_original/lib');
addpath('/home/bxl180002/git/matpower_original/lib/t');
addpath('/home/bxl180002/git/matpower_original/most');
addpath('/home/bxl180002/git/matpower_original/most/lib/t');

MPC_2k_t=loadcase('case_ACTIVSg2000.m');
MPC_2k=ext2int(MPC_2k_t);

% WARNING: The bus name in Kwami's file is different from the original
% ACTIVSg2000 syste. In the SUMMER-GO conference paper, we don't have flow
% limits so it does not matter.
Bus_Name=readtable('/petastore/ganymede/home/bxl180002/SUMMER_GO/Jubeyer/Texas_2k_Data/Texas_2k_load.csv');
B_N=Bus_Name.Properties.VariableNames;
B_N([1, end])=[]; % 1 is time and end is LOAD

Load_id=find(MPC_2k.bus(:,3)~=0 |MPC_2k.bus(:,4)~=0);


Initial_load = MPC_2k.bus(Load_id,3);
Total_real_load = sum(Initial_load);
distribution_factor = Initial_load/Total_real_load;

load_actual_bus = load_actual(:)*distribution_factor(:)';
load_da_bus = load_da(:)*distribution_factor(:)';
load_ha_bus = load_ha(:)*distribution_factor(:)';

Ta_load = array2table(load_actual_bus, 'VariableNames', B_N);
Tf_load_da = array2table(load_da_bus, 'VariableNames', B_N);
Tf_load_ha = array2table(load_ha_bus, 'VariableNames', B_N);

Ta_load = [array2table(t5, 'VariableNames', {'TIME'}) Ta_load array2table(load_actual, 'VariableNames', {'LOAD'})];
Tf_load_da = [array2table(timestamp_da, 'VariableNames', {'TIME'}) Tf_load_da array2table(load_da, 'VariableNames', {'LOAD'})];
Tf_load_ha = [array2table(timestamp_ha, 'VariableNames', {'TIME'}) Tf_load_ha array2table(load_ha, 'VariableNames', {'LOAD'})];