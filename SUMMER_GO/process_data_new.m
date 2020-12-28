% For the file from Cong in the second time 
%% Read wind data

Tf_wind_ha = readtable('Cong/Forecasts2Binghui/Wind_debug/WindBus_Forecasts_debug.csv');
Tf_wind_ha.leadtime = Tf_wind_ha.TimeStamp - Tf_wind_ha.IssueTime;
unique_leadtime_wind = unique(Tf_wind_ha.leadtime);
for i = 1:length(unique_leadtime_wind)
    tmp = Tf_wind_ha(Tf_wind_ha.leadtime==unique_leadtime_wind(i), :);
    fprintf('Wind, Leadtime: %s, %g in total.\n', unique_leadtime_wind(i), size(tmp, 1));
end

Tf_wind_da = readtable('Cong/Forecasts2Binghui/Wind_debug/WindBus_ForecastDA.csv');
Ta_wind = readtable('Cong/Forecasts2Binghui/Wind_debug/WindBus_Actual.csv');

%% Read solar data
genname_solar = {
    'solar109';
    'solar21';
    'solar22';
    'solar286';
    'solar287';
    'solar288';
    'solar289';
    'solar290';
    'solar291';
    'solar3';
    'solar362';
    'solar363';
    'solar364';
    'solar365';
    'solar433';
    'solar434';
    'solar435';
    'solar436';
    'solar437';
    'solar438';
    'solar509';
    'solar530';};
cell_Tsolar = cell(numel(genname_solar), 1);
for j = 1: numel(genname_solar)
    csvname = strcat('SolarForecast_', genname_solar{j}, '.csv');
    pathname = strcat('Cong/Forecasts2Binghui/Solar_debug/', csvname);
    T_solar = readtable(pathname);
    T_solar.leadtime = T_solar.TimeStamp - T_solar.IssueTime;
    cell_Tsolar{j} = T_solar;
    unique_leadtime_solar = unique(T_solar.leadtime);
    for i = 1:length(unique_leadtime_solar)
        tmp = T_solar(T_solar.leadtime==unique_leadtime_solar(i), :);
        fprintf('Solar: %s, Leadtime: %s, %g in total.\n', genname_solar{j}, unique_leadtime_solar(i), size(tmp, 1));
    end
end

ar_forecast = nan(size(T_solar, 1), numel(genname_solar));
ar_actual   = nan(size(T_solar, 1), numel(genname_solar));
for j = 1: numel(genname_solar)
    T_solar = cell_Tsolar{j};
    ar_forecast(:, j) = T_solar.Forecast;
    ar_actual(:, j) = T_solar.Actual;
end
Tf_solar_ha = [T_solar(:, {'TimeStamp', 'IssueTime'}) array2table(ar_forecast, 'VariableNames', genname_solar)];
Ta_solar = [T_solar(:, {'TimeStamp', 'IssueTime'}) array2table(ar_actual, 'VariableNames', genname_solar)];
Tf_solar_ha.leadtime = Tf_solar_ha.TimeStamp - Tf_solar_ha.IssueTime;
Ta_solar.leadtime = Ta_solar.TimeStamp - Ta_solar.IssueTime;

%% Check missing values for a specific lead time
% t_min = min([min(T_load.TimeStamp), min(T_wind.TimeStamp), min(T_solar.TimeStamp)]);
% t_max = max([max(T_load.TimeStamp), max(T_wind.TimeStamp), max(T_solar.TimeStamp)]);
t_min = min([min(Tf_wind_ha.TimeStamp), min(Tf_solar_ha.TimeStamp)]);
t_max = max([max(Tf_wind_ha.TimeStamp), max(Tf_solar_ha.TimeStamp)]);
time_seq_5m = [t_min: duration(0,5,0): t_max]';
% selected_load = 'bus1001';
selected_wind = 'wind0';
selected_solar = 'solar109';

% all_leadtime = [duration(0, 50, 0); duration(0, 55, 0); duration(1, 0, 0)];
all_leadtime = unique(Tf_wind_ha.leadtime);
cell_checknan = cell(size(all_leadtime, 1), 1);

for j = 1:size(all_leadtime, 1)
    selected_leadtime = all_leadtime(j);

    genname_wind = Tf_wind_ha.Properties.VariableNames(contains(Tf_wind_ha.Properties.VariableNames, 'wind'));
    Tf_wind_ha_selected = Tf_wind_ha(Tf_wind_ha.leadtime==selected_leadtime, :);
    [Lia,Locb] = ismember(time_seq_5m, Tf_wind_ha_selected.TimeStamp);
    tmp = nan(size(time_seq_5m, 1), numel(genname_wind));
    tmp(Lia, :) = Tf_wind_ha_selected{Locb(Lia), genname_wind};
    Tf_wind_ha_aligned = [array2table(time_seq_5m, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_wind)];

    genname_solar = Tf_solar_ha.Properties.VariableNames(contains(Tf_solar_ha.Properties.VariableNames, 'solar'));
    Tf_solar_ha_selected = Tf_solar_ha(Tf_solar_ha.leadtime==selected_leadtime, :);
    [Lia,Locb] = ismember(time_seq_5m, Tf_solar_ha_selected.TimeStamp);
    tmp = nan(size(time_seq_5m, 1), numel(genname_solar));
    tmp(Lia, :) = Tf_solar_ha_selected{Locb(Lia), genname_solar};
    Tf_solar_ha_aligned = [array2table(time_seq_5m, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_solar)];
    
%     genname_load = T_load.Properties.VariableNames(contains(T_load.Properties.VariableNames, 'bus'));
%     T_load_selected = T_load(T_load.leadtime==selected_leadtime, :);
%     [Lia,Locb] = ismember(time_seq, T_load_selected.TimeStamp);
%     tmp = nan(size(time_seq, 1), numel(genname_load));
%     tmp(Lia, :) = T_load_selected{Locb(Lia), genname_load};
%     T_load_aligned = [array2table(time_seq, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_load)];

    unique_dates = unique([time_seq_5m.Year, time_seq_5m.Month, time_seq_5m.Day], 'rows');
    n_nan = zeros(size(unique_dates, 1), 3);
    for i = 1:size(unique_dates, 1)
        this_y = unique_dates(i, 1);
        this_m = unique_dates(i, 2);
        this_d = unique_dates(i, 3);

%         rows_select = (T_load_aligned.TIME.Year==this_y)&(T_load_aligned.TIME.Month==this_m)&(T_load_aligned.TIME.Day==this_d);
%         n_nan(i, 1) = sum(isnan(T_load_aligned{rows_select, selected_load}));

        rows_select = (Tf_wind_ha_aligned.TIME.Year==this_y)&(Tf_wind_ha_aligned.TIME.Month==this_m)&(Tf_wind_ha_aligned.TIME.Day==this_d);
        n_nan(i, 2) = sum(isnan(Tf_wind_ha_aligned{rows_select, selected_wind}));

        rows_select = (Tf_solar_ha_aligned.TIME.Year==this_y)&(Tf_solar_ha_aligned.TIME.Month==this_m)&(Tf_solar_ha_aligned.TIME.Day==this_d);
        n_nan(i, 3) = sum(isnan(Tf_solar_ha_aligned{rows_select, selected_solar}));
    end

    T_checknan = array2table([unique_dates, n_nan], 'VariableNames', {'Year', 'Month', 'Day', 'load', 'wind', 'solar'});
    cell_checknan{j} = T_checknan;
end

% Display all dates with nan
for j = 1:size(all_leadtime, 1)
    disp(all_leadtime(j));
    T_checknan = cell_checknan{j};
    T_checknan(sum(T_checknan{:, 4:6}, 2)~=0, :)
end

%% Prepare DA, HA and actual data. Note: HA has selected a leadtime, say 1 hour
selected_leadtime = duration(1, 0, 0);
t_min_da = min([min(Tf_wind_da.INTERVAL_ENDING)]);
t_max_da = max([max(Tf_wind_da.INTERVAL_ENDING)]);
time_seq_da = [t_min_da: duration(1,0,0): t_max_da]';

t_min = min([min(Tf_wind_ha.TimeStamp), min(Tf_solar_ha.TimeStamp)]);
t_max = max([max(Tf_wind_ha.TimeStamp), max(Tf_solar_ha.TimeStamp)]);
time_seq_5m = [t_min: duration(0,5,0): t_max]';

genname_wind = Tf_wind_ha.Properties.VariableNames(contains(Tf_wind_ha.Properties.VariableNames, 'wind'));

% Solar actual
Ta_solar_selected = Ta_solar(Ta_solar.leadtime==selected_leadtime, :);
[Lia,Locb] = ismember(time_seq_5m, Ta_solar_selected.TimeStamp);
tmp = nan(size(time_seq_5m, 1), numel(genname_solar));
tmp(Lia, :) = Ta_solar_selected{Locb(Lia), genname_solar};
Ta_solar_aligned = [array2table(time_seq_5m, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_solar)];

% Solar HA forecast
Tf_solar_ha_selected = Tf_solar_ha(Tf_solar_ha.leadtime==selected_leadtime, :);
[Lia,Locb] = ismember(time_seq_5m, Tf_solar_ha_selected.TimeStamp);
tmp = nan(size(time_seq_5m, 1), numel(genname_solar));
tmp(Lia, :) = Tf_solar_ha_selected{Locb(Lia), genname_solar};
Tf_solar_ha_aligned = [array2table(time_seq_5m, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_solar)];

% Wind actual
[Lia,Locb] = ismember(time_seq_5m, Ta_wind.INTERVAL_ENDING);
tmp = nan(size(time_seq_5m, 1), numel(genname_wind));
tmp(Lia, :) = Ta_wind{Locb(Lia), genname_wind};
Ta_wind_aligned = [array2table(time_seq_5m, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_wind)];

% Wind HA forecast
Tf_wind_ha_selected = Tf_wind_ha(Tf_wind_ha.leadtime==selected_leadtime, :);
[Lia,Locb] = ismember(time_seq_5m, Tf_wind_ha_selected.TimeStamp);
tmp = nan(size(time_seq_5m, 1), numel(genname_wind));
tmp(Lia, :) = Tf_wind_ha_selected{Locb(Lia), genname_wind};
Tf_wind_ha_aligned = [array2table(time_seq_5m, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_wind)];

% Wind DA forecast
[Lia,Locb] = ismember(time_seq_da, Tf_wind_da.INTERVAL_ENDING);
tmp = nan(size(time_seq_da, 1), numel(genname_wind));
tmp(Lia, :) = Tf_wind_da{Locb(Lia), genname_wind};
Tf_wind_da_aligned = [array2table(time_seq_da, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_wind)];

% Solar DA forecast
tmp = [Tf_solar_ha_aligned{1:end, 2:end};nan(1, 22)];
tmp = reshape(tmp(:), 12, numel(tmp)/12);
tmp = mean(tmp, 1);
tmp = reshape(tmp, 8760, numel(tmp)/8760);
Tf_solar_da_aligned = [array2table(time_seq_da, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_solar)];

% Take a look at the complete sets of wind forecast
stairs(Tf_wind_da_aligned.TIME-duration(1, 0, 0), Tf_wind_da_aligned.wind0, 'g');
hold on; stairs(Ta_wind_aligned.TIME-duration(0, 5, 0), Ta_wind_aligned.wind0, 'k');
stairs(Tf_wind_ha_aligned.TIME-duration(0, 5, 0), Tf_wind_ha_aligned.wind0, 'b');

% Take a look at the complete sets of solar forecast
stairs(Tf_solar_da_aligned.TIME-duration(1, 0, 0), Tf_solar_da_aligned.solar109, 'g');
hold on; stairs(Ta_solar_aligned.TIME-duration(0, 5, 0), Ta_solar_aligned.solar109, 'k');
stairs(Tf_solar_ha_aligned.TIME-duration(0, 5, 0), Tf_solar_ha_aligned.solar109, 'b');