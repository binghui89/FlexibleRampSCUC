%% Process Kate's solar data

all_solar = {'solar109', 'solar21', 'solar22', 'solar286', 'solar287', 'solar288', 'solar289', 'solar290', 'solar291', 'solar362', 'solar363', 'solar364', 'solar365', 'solar3', 'solar433', 'solar434', 'solar435', 'solar436', 'solar437', 'solar438', 'solar509', 'solar530'};
dir_work = 'Forecasts_fromKate';
dir_home = pwd;

cd(dir_work);
for s = 1:numel(all_solar)
    sname = all_solar{s};
    
    h5filename = strcat(sname, '.h5');
    csvfilename = strcat(sname, '.csv');

    dset1 = h5read(h5filename, '/Issue_index');
    dset2 = h5read(h5filename, '/Percentile');
    dset3 = h5read(h5filename, '/Power');
    dset4 = h5read(h5filename, '/Step_index');

    issue_time = [datetime(2017, 12, 31, 23, 00, 00): duration(0, 5, 0): datetime(2018, 12, 31, 17, 55, 00)]';
    lead_time = [1:1:24]'.*duration(0, 5, 0);

    power_ha = squeeze(dset3(:, dset4==12, dset2==50)); % Hour-ahead forecast, 50 percentile
    power_5ma = squeeze(dset3(:, dset4==1, dset2==50)); % 5-min-ahead forecast, 50 percentile
    time_index_ha = issue_time + lead_time(12);
    time_index_5ma = issue_time + lead_time(1);
    % plot(time_index_ha, power_ha, time_index_5ma, power_5ma);
    % [Lia,Locb] = ismember(time_index_all, time_index_5ma);

    cell_leadstep = cell(numel(dset4), 1);
    for i = 1: size(cell_leadstep, 1)
        cell_leadstep{i} = strcat('lead', num2str(dset4(i)));
    end
    T_power_p50 = array2table(squeeze(dset3(:, :, dset2==50)), 'VariableNames', cell_leadstep);
    T_power_p50 = [array2table(issue_time, 'VariableNames', {'ISSUE_TIME'}), T_power_p50];
    writetable(T_power_p50, csvfilename);
end
cd(dir_home);