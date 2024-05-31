mat_path = '.\vehicles11-17\';
mat_files = dir (strcat(mat_path, '*.mat'));
TEN_HOURS = 3600 * 10;
PRODUCTION_FREQUENCY = 10;
snapshot_hist_names     = ["ambient_temperature" "cell_temperature" "SOC" "current" "voltage"];
snapshot_var_names      = ["snapshot_age" "maximum_capacity" "age_equivalent_full_cycles"];
bin_num = 100;

tic
for m=1:numel(mat_files)
    v=load(strcat(mat_path, mat_files(m).name));
    all_vehicles_mat = v.AllVehicles;

    for i=1:numel(all_vehicles_mat) 
        if all_vehicles_mat(i).EOT == 0
            disp(i)
        else
            for j=1:numel(snapshot_hist_names)
                all_data.(snapshot_hist_names{j}) = [];
            end
        
            snapshot_name = fields(all_vehicles_mat(i).SnapShot10h);
            number_of_snapshots = numel(snapshot_name);
            all_data.ID             = all_vehicles_mat(i).ID                        * ones(number_of_snapshots, 1);
            all_data.failure_age    = all_vehicles_mat(i).EOT                       * ones(number_of_snapshots, 1);
            all_data.status         = 0                                             * ones(number_of_snapshots, 1);
            all_data.status(end)    = all_vehicles_mat(i).Status;
            all_data.production     = ceil(i / PRODUCTION_FREQUENCY) * TEN_HOURS    * ones(number_of_snapshots, 1);
            all_data.InitialCapacity= all_vehicles_mat(i).InitialCapacity           * ones(number_of_snapshots, 1);
            all_data.climate        = all_vehicles_mat(i).climate                   * ones(number_of_snapshots, 1);
            all_data.Type           = ones(number_of_snapshots, 1);
            %Type
            lenDOD = 54001; %In a case fo changing simulation time or sampling rate of generating DOD this should change 
            index=1;
            all_vehicles_mat(i).type_change_time = [all_vehicles_mat(i).type_change_time 1];

            for k=1:length(all_vehicles_mat(i).Type)
                index2 = fix((fix(lenDOD * all_vehicles_mat(i).type_change_time(k)) + 1)*1000/TEN_HOURS);%In a case fo changing simulation time or sampling rate of generating DOD 1000 should change 
                if index2 >= (all_vehicles_mat(i).EOT/TEN_HOURS)
                    all_data.Type(index:end) = all_data.Type(index:end)*all_vehicles_mat(i).Type(k);
                    break
                else
                    all_data.Type(index:index2) = all_data.Type(index:index2)*all_vehicles_mat(i).Type(k);
                    index = index2+1;
                end
            end
            all_data.TypeChangeTimes = (k-1) * ones(number_of_snapshots, 1);        
            for k=1:numel(snapshot_var_names)  %non-histogram snapshot variables
                all_data.(snapshot_var_names{k}) = zeros(number_of_snapshots, 1);
                for j=1:number_of_snapshots
                    if snapshot_var_names(k) == "snapshot_age"
                        all_data.(snapshot_var_names{k})(j) = min([j * TEN_HOURS, all_vehicles_mat(i).EOT]);
                    else
                        all_data.(snapshot_var_names{k})(j) = all_vehicles_mat(i).SnapShot10h.(snapshot_name{j}).(snapshot_var_names{k});
                    end
                end
            end
            for k=1:numel(snapshot_hist_names) %histogram snapshot variables
                all_data.(snapshot_hist_names{k}) = zeros(number_of_snapshots, bin_num);
                for j=1:number_of_snapshots
                    all_data.(snapshot_hist_names{k})(j,:) = all_vehicles_mat(i).SnapShot10h.(snapshot_name{j}).(snapshot_hist_names{k});
                end
            end
            
            writetable(struct2table(all_data), strcat('.\vehicles11-17\all_data',num2str(fix(m/10)),'.csv'), 'WriteMode', 'append');
            clear("all_data")
        end
    end
end
toc
