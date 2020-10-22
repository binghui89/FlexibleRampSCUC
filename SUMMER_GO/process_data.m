%% See how many values are missing there
T_wind = readtable('Cong/WindBus_Forecasts.csv');
T_wind.leadtime = T_wind.TimeStamp - T_wind.IssueTime;
unique_leadtime_wind = unique(T_wind.leadtime);
for i = 1:length(unique_leadtime_wind)
    tmp = T_wind(T_wind.leadtime==unique_leadtime_wind(i), :);
    fprintf('Leadtime: %s, %g in total.\n', unique_leadtime_wind(i), size(tmp, 1));
end

T_solar = readtable('Cong/22SolarSite_Forecasts.csv');
T_solar.leadtime = T_solar.TimeStamp - T_solar.IssueTime;
unique_leadtime_solar = unique(T_solar.leadtime);
for i = 1:length(unique_leadtime_solar)
    tmp = T_solar(T_solar.leadtime==unique_leadtime_solar(i), :);
    fprintf('Leadtime: %s, %g in total.\n', unique_leadtime_solar(i), size(tmp, 1));
end

T_load = load('Jubeyer/Forecast_Distributed.mat', 'T_forecast');
T_load = T_load.('T_forecast');
T_load.leadtime = T_load.TimeStamp - T_load.IssueTime;
unique_leadtime_load = unique(T_load.leadtime);
for i = 1:length(unique_leadtime_load)
    tmp = T_load(T_load.leadtime==unique_leadtime_load(i), :);
    fprintf('Leadtime: %s, %g in total.\n', unique_leadtime_load(i), size(tmp, 1));
end

%% Check missing values for a specific lead time
t_min = min([min(T_load.TimeStamp), min(T_wind.TimeStamp), min(T_solar.TimeStamp)]);
t_max = max([max(T_load.TimeStamp), max(T_wind.TimeStamp), max(T_solar.TimeStamp)]);
time_seq = [t_min: duration(0,5,0): t_max]';
selected_load = 'bus1001';
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

    genname_solar = T_solar.Properties.VariableNames(contains(T_solar.Properties.VariableNames, 'solar'));
    T_solar_selected = T_solar(T_solar.leadtime==selected_leadtime, :);
    [Lia,Locb] = ismember(time_seq, T_solar_selected.TimeStamp);
    tmp = nan(size(time_seq, 1), numel(genname_solar));
    tmp(Lia, :) = T_solar_selected{Locb(Lia), genname_solar};
    T_solar_aligned = [array2table(time_seq, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_solar)];

    genname_load = T_load.Properties.VariableNames(contains(T_load.Properties.VariableNames, 'bus'));
    T_load_selected = T_load(T_load.leadtime==selected_leadtime, :);
    [Lia,Locb] = ismember(time_seq, T_load_selected.TimeStamp);
    tmp = nan(size(time_seq, 1), numel(genname_load));
    tmp(Lia, :) = T_load_selected{Locb(Lia), genname_load};
    T_load_aligned = [array2table(time_seq, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_load)];

    unique_dates = unique([time_seq.Year, time_seq.Month, time_seq.Day], 'rows');
    n_nan = zeros(size(unique_dates, 1), 3);
    for i = 1:size(unique_dates, 1)
        this_y = unique_dates(i, 1);
        this_m = unique_dates(i, 2);
        this_d = unique_dates(i, 3);

        rows_select = (T_load_aligned.TIME.Year==this_y)&(T_load_aligned.TIME.Month==this_m)&(T_load_aligned.TIME.Day==this_d);
        n_nan(i, 1) = sum(isnan(T_load_aligned{rows_select, selected_load}));

        rows_select = (T_wind_aligned.TIME.Year==this_y)&(T_wind_aligned.TIME.Month==this_m)&(T_wind_aligned.TIME.Day==this_d);
        n_nan(i, 2) = sum(isnan(T_wind_aligned{rows_select, selected_wind}));

        rows_select = (T_solar_aligned.TIME.Year==this_y)&(T_solar_aligned.TIME.Month==this_m)&(T_solar_aligned.TIME.Day==this_d);
        n_nan(i, 3) = sum(isnan(T_solar_aligned{rows_select, selected_solar}));
    end

    T_checknan = array2table([unique_dates, n_nan], 'VariableNames', {'Year', 'Month', 'Day', 'load', 'wind', 'solar'});
    cell_checknan{j} = T_checknan;
end


%%
this_y = 2018;
this_m = 1;
this_d = 2;
selected_leadtime = duration(1, 0, 0);

genname_wind = T_wind.Properties.VariableNames(contains(T_wind.Properties.VariableNames, 'wind'));
T_wind_selected = T_wind(T_wind.leadtime==selected_leadtime, :);
[Lia,Locb] = ismember(time_seq, T_wind_selected.TimeStamp);
tmp = nan(size(time_seq, 1), numel(genname_wind));
tmp(Lia, :) = T_wind_selected{Locb(Lia), genname_wind};
T_wind_aligned = [array2table(time_seq, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_wind)];
rows_select = (T_wind_aligned.TIME.Year==this_y)&(T_wind_aligned.TIME.Month==this_m)&(T_wind_aligned.TIME.Day==this_d);
ar_wind_rtc = T_wind_aligned{rows_select, genname_wind};

genname_solar = T_solar.Properties.VariableNames(contains(T_solar.Properties.VariableNames, 'solar'));
T_solar_selected = T_solar(T_solar.leadtime==selected_leadtime, :);
[Lia,Locb] = ismember(time_seq, T_solar_selected.TimeStamp);
tmp = nan(size(time_seq, 1), numel(genname_solar));
tmp(Lia, :) = T_solar_selected{Locb(Lia), genname_solar};
T_solar_aligned = [array2table(time_seq, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_solar)];
rows_select = (T_solar_aligned.TIME.Year==this_y)&(T_solar_aligned.TIME.Month==this_m)&(T_solar_aligned.TIME.Day==this_d);
ar_solar_rtc = T_solar_aligned{rows_select, genname_solar};

T_forecast_rtc = [array2table([1:size(ar_wind_rtc, 1)]', 'VariableNames', {'Slot'}), array2table(ar_wind_rtc, 'VariableNames', genname_wind), array2table(ar_solar_rtc, 'VariableNames', genname_solar)]; 

genname_load = T_load.Properties.VariableNames(contains(T_load.Properties.VariableNames, 'bus'));
T_load_selected = T_load(T_load.leadtime==selected_leadtime, :);
[Lia,Locb] = ismember(time_seq, T_load_selected.TimeStamp);
tmp = nan(size(time_seq, 1), numel(genname_load));
tmp(Lia, :) = T_load_selected{Locb(Lia), genname_load};
T_load_aligned = [array2table(time_seq, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', genname_load)];
rows_select = (T_load_aligned.TIME.Year==this_y)&(T_load_aligned.TIME.Month==this_m)&(T_load_aligned.TIME.Day==this_d);
ar_load_rtc = T_load_aligned{rows_select, genname_load};

T_load_rtc = [array2table([1:size(ar_load_rtc, 1)]', 'VariableNames', {'Slot'}), array2table(ar_load_rtc, 'VariableNames', genname_load)];

% use hourly average as DA forecasts
ar_wind_dac = reshape(mean(reshape(ar_wind_rtc, 12, numel(ar_wind_rtc)/12), 1), 24, numel(ar_wind_rtc)/12/24);
ar_solar_dac = reshape(mean(reshape(ar_solar_rtc, 12, numel(ar_solar_rtc)/12), 1), 24, numel(ar_solar_rtc)/12/24);
ar_load_dac = reshape(mean(reshape(ar_load_rtc, 12, numel(ar_load_rtc)/12), 1), 24, numel(ar_load_rtc)/12/24);

T_forecast_dac = [array2table([1:size(ar_wind_dac, 1)]', 'VariableNames', {'Slot'}), array2table(ar_wind_dac, 'VariableNames', genname_wind), array2table(ar_solar_dac, 'VariableNames', genname_solar)]; 
T_load_dac = [array2table([1:size(ar_load_dac, 1)]', 'VariableNames', {'Slot'}), array2table(ar_load_dac, 'VariableNames', genname_load)];

%% 
T_nsr = readtable('Cong/NSRR_withLowrBound.csv');
nsr_names = {'Baseline','NSRR7D','NSRR1D','NSRR1H'};
time_seq_nsr = [min(T_nsr.TimeStamp): duration(1,0,0): max(T_nsr.TimeStamp)]';
[Lia,Locb] = ismember(time_seq_nsr, T_nsr.TimeStamp);
tmp = nan(size(time_seq_nsr, 1), numel(nsr_names));
tmp(Lia, :) = T_nsr{Locb(Lia), nsr_names};
T_nsr_aligned = [array2table(time_seq_nsr, 'VariableNames', {'TIME'}), array2table(tmp, 'VariableNames', nsr_names)];
rows_select = (T_nsr_aligned.TIME.Year==this_y)&(T_nsr_aligned.TIME.Month==this_m)&(T_nsr_aligned.TIME.Day==this_d);

ar_nsr_dac = T_nsr_aligned{rows_select, {'NSRR1D'}};
ar_nsr_rtc = repmat(T_nsr_aligned{rows_select, {'NSRR1H'}}', 12, 1);
ar_nsr_rtc = ar_nsr_rtc(:);

ar_nsr_dac_base = T_nsr_aligned{rows_select, {'Baseline'}};
ar_nsr_rtc_base = repmat(T_nsr_aligned{rows_select, {'Baseline'}}', 12, 1);
ar_nsr_rtc_base = ar_nsr_rtc_base(:);

T_nsr_dac = [array2table([1:size(ar_nsr_dac, 1)]', 'VariableNames', {'Slot'}), array2table(ar_nsr_dac, 'VariableNames', {'NSR'})];
T_nsr_rtc = [array2table([1:size(ar_nsr_rtc, 1)]', 'VariableNames', {'Slot'}), array2table(ar_nsr_rtc, 'VariableNames', {'NSR'})];
T_nsr_dac_base = [array2table([1:size(ar_nsr_dac_base, 1)]', 'VariableNames', {'Slot'}), array2table(ar_nsr_dac_base, 'VariableNames', {'NSR'})];
T_nsr_rtc_base = [array2table([1:size(ar_nsr_rtc_base, 1)]', 'VariableNames', {'Slot'}), array2table(ar_nsr_rtc_base, 'VariableNames', {'NSR'})];

% writetable(T_forecast_dac, 'da_generator.csv');
% writetable(T_load_dac, 'da_load.csv');
% writetable(T_load_rtc, 'ha_load.csv');
% writetable(T_forecast_rtc, 'ha_generator.csv');
% writetable(T_nsr_dac, 'da_nsr.csv');
% writetable(T_nsr_rtc, 'ha_nsr.csv');
% writetable(T_nsr_dac_base, 'da_nsr_base.csv');
% writetable(T_nsr_rtc_base, 'ha_nsr_base.csv');
