% For the file from Cong in the second time 

T_wind = readtable('Cong/Forecasts2Binghui/Wind_debug/WindBus_Forecasts_debug.csv');
T_wind.leadtime = T_wind.TimeStamp - T_wind.IssueTime;
unique_leadtime_wind = unique(T_wind.leadtime);
for i = 1:length(unique_leadtime_wind)
    tmp = T_wind(T_wind.leadtime==unique_leadtime_wind(i), :);
    fprintf('Leadtime: %s, %g in total.\n', unique_leadtime_wind(i), size(tmp, 1));
end

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
Tf_solar = [T_solar(:, {'TimeStamp', 'IssueTime'}) array2table(ar_forecast, 'VariableNames', genname_solar)];
Ta_solar = [T_solar(:, {'TimeStamp', 'IssueTime'}) array2table(ar_actual, 'VariableNames', genname_solar)];
Tf_solar.leadtime = Tf_solar.TimeStamp - Tf_solar.IssueTime;
Ta_solar.leadtime = Ta_solar.TimeStamp - Ta_solar.IssueTime;

%% Check missing values for a specific lead time
% t_min = min([min(T_load.TimeStamp), min(T_wind.TimeStamp), min(T_solar.TimeStamp)]);
% t_max = max([max(T_load.TimeStamp), max(T_wind.TimeStamp), max(T_solar.TimeStamp)]);
t_min = min([min(T_wind.TimeStamp), min(Tf_solar.TimeStamp)]);
t_max = max([max(T_wind.TimeStamp), max(Tf_solar.TimeStamp)]);
time_seq = [t_min: duration(0,5,0): t_max]';
% selected_load = 'bus1001';
selected_wind = 'wind0';
selected_solar = 'solar109';

all_leadtime = [duration(0, 50, 0); duration(0, 55, 0); duration(1, 0, 0)];
cell_checknan = cell(size(all_leadtime, 1), 1);

for j = 1:size(all_leadtime, 1)
    selected_leadtime = all_leadtime(j);

    genname_wind = T_wind.Properties.VariableNames(contains(T_wind.Properties.VariableNames, 'wind'));
    T_wind_selected = T_wind(T_wind.leadtime==selected_leadtime, :);
    [Lia,Locb] = ismember(time_seq, T_wind_selected.TimeStamp);
    tmp = nan(size(time_seq, 1), numel(genname_wind));
    tmp(Lia, :) = T_wind_selected{Locb(Lia), genname_wind};
    T_wind_aligned = [array2table(time_seq, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_wind)];

    genname_solar = Tf_solar.Properties.VariableNames(contains(Tf_solar.Properties.VariableNames, 'solar'));
    Tf_solar_selected = Tf_solar(Tf_solar.leadtime==selected_leadtime, :);
    [Lia,Locb] = ismember(time_seq, Tf_solar_selected.TimeStamp);
    tmp = nan(size(time_seq, 1), numel(genname_solar));
    tmp(Lia, :) = Tf_solar_selected{Locb(Lia), genname_solar};
    Tf_solar_aligned = [array2table(time_seq, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_solar)];
    
%     genname_load = T_load.Properties.VariableNames(contains(T_load.Properties.VariableNames, 'bus'));
%     T_load_selected = T_load(T_load.leadtime==selected_leadtime, :);
%     [Lia,Locb] = ismember(time_seq, T_load_selected.TimeStamp);
%     tmp = nan(size(time_seq, 1), numel(genname_load));
%     tmp(Lia, :) = T_load_selected{Locb(Lia), genname_load};
%     T_load_aligned = [array2table(time_seq, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_load)];

    unique_dates = unique([time_seq.Year, time_seq.Month, time_seq.Day], 'rows');
    n_nan = zeros(size(unique_dates, 1), 3);
    for i = 1:size(unique_dates, 1)
        this_y = unique_dates(i, 1);
        this_m = unique_dates(i, 2);
        this_d = unique_dates(i, 3);

%         rows_select = (T_load_aligned.TIME.Year==this_y)&(T_load_aligned.TIME.Month==this_m)&(T_load_aligned.TIME.Day==this_d);
%         n_nan(i, 1) = sum(isnan(T_load_aligned{rows_select, selected_load}));

        rows_select = (T_wind_aligned.TIME.Year==this_y)&(T_wind_aligned.TIME.Month==this_m)&(T_wind_aligned.TIME.Day==this_d);
        n_nan(i, 2) = sum(isnan(T_wind_aligned{rows_select, selected_wind}));

        rows_select = (Tf_solar_aligned.TIME.Year==this_y)&(Tf_solar_aligned.TIME.Month==this_m)&(Tf_solar_aligned.TIME.Day==this_d);
        n_nan(i, 3) = sum(isnan(Tf_solar_aligned{rows_select, selected_solar}));
    end

    T_checknan = array2table([unique_dates, n_nan], 'VariableNames', {'Year', 'Month', 'Day', 'load', 'wind', 'solar'});
    cell_checknan{j} = T_checknan;
end

Ta_solar_selected = Ta_solar(Ta_solar.leadtime==selected_leadtime, :);
[Lia,Locb] = ismember(time_seq, Ta_solar_selected.TimeStamp);
tmp = nan(size(time_seq, 1), numel(genname_solar));
tmp(Lia, :) = Ta_solar_selected{Locb(Lia), genname_solar};
Ta_solar_aligned = [array2table(time_seq, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_solar)];
