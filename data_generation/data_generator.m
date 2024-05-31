rng(1)
SamplingRate = 10;
SimTime = 3600 * 1000 * 15;
C = 40; %battery capacity
MODEL = "battery_aging";
AllVehiclesInput = [];
RemoveField = {'DoD', 'DoC', 'DischargeCurrent', 'ChargeCurrent', 'AmbientTemperature', 'BatteryUsagTS'};
tic
% generating inputs for simulation
%for num=1:84
% for num=1:1
%     AllVehiclesInput = [];
%     disp(num)
%     for i=1:60
%     AllVehiclesInput = [AllVehiclesInput Input_Generator(SimTime ,(60*(num-1))+i, C)];
%     end
%     Vnames = strcat('./AllVehiclesInputnew', num2str(num),'.mat');
%     save(Vnames,'AllVehiclesInput','-mat')
% end 
toc
for i=1:1
     tic
    disp(i)
    Vnames = strcat('AllVehiclesInputnew', num2str(i),'.mat');
    AllVehiclesInput = load(Vnames);
    AllVehiclesInput = AllVehiclesInput.AllVehiclesInput;

    %AllVehicles = Run_Simulation(AllVehiclesInput, MODEL , "parallel", SimTime);
    %AllVehicles = Run_Simulation(AllVehiclesInput, MODEL , "serie", SimTime);
    AllVehicles = Run_Simulation(AllVehiclesInput((i*1)-0:i*1), MODEL , "serie", SimTime);
    disp("sim done")
    parfor num=1:numel(AllVehicles)  
        AllVehicles(num).SnapShot10h = MakeSnapshots(AllVehicles(num), 10*3600, SamplingRate);
    end 
    disp("Hist done")
    AllVehicles = rmfield(AllVehicles, RemoveField);
    Vnames = strcat('./Vehicles/Vehicle', num2str(i),'.mat');
    save(Vnames,'AllVehicles','-mat')
    disp("Save done")
    clear("AllVehicles")
    toc
end 
%Input Data Generator 

%Runsimulation In serie or parallel 
function AllVehicles = Run_Simulation(AllVehicles, Model , run_type, SimTime)
    %generate battery 
    for i=1:numel(AllVehicles)
        in(i) = Simulink.SimulationInput(Model);
        in(i) = in(i).setVariable("Vehicle", AllVehicles(i));
    end
    %In Serie or Parallel
    if run_type == "serie"
        out = sim(in);    
    elseif run_type == "parallel"
        out = parsim(in);
    else 
        print("incorrect run type")
    end 
    %results 
    Variables = fields(out(1).BatteryUsage);
    for i=1:numel(AllVehicles)
        AllVehicles(i).BatteryUsagTS = out(i).BatteryUsage;
        
        AllVehicles(i).EOT = AllVehicles(i).BatteryUsagTS.(Variables{1}).time(end);
        if AllVehicles(i).EOT<=SimTime
            AllVehicles(i).Status = 1; %failur
        else
            AllVehicles(i).Status = 0; %sensored
        end
    end 
end

function Vehicle  = Input_Generator(SimTime ,num, C)
    
    Vehicle.ID = num;
    TYPES = [1 2 3];
    rates = [0.4 0.7 1.5 ];
    rate = randsample(rates, 1);
    max_change_time = 1;
    [Vehicle.Type, Vehicle.type_change_time] = generate_changing_times(TYPES, rate, max_change_time);

        


    
    Vehicle.InitialCapacity = 40 + rand()*3.2; %it is around 8% error    
    %DoD
    Vehicle.DoD.Time = [0:1000:SimTime];
    Vehicle.DoD.Amplitude = ceil(TypeChangeDist(Vehicle.Type, 10, 90, numel(Vehicle.DoD.Time),Vehicle.type_change_time));
    Vehicle.DoD.SampleTime = 0;

    %DoC
    Vehicle.DoC.Time = [0:1000:SimTime];
    Vehicle.DoC.Amplitude = DoCgen(Vehicle.DoD.Amplitude, 98, 10);
%     Vehicle.DoC.Time = [0]*SimTime;
%     Vehicle.DoC.Amplitude = 70;

    Vehicle.DoC.SampleTime = 0;

    %DischargeCurrent
    Vehicle.DischargeCurrent.Time = [0:1000:SimTime];
    Vehicle.DischargeCurrent.Amplitude = getVehicleDist(4, 0.25*C, 1*C, numel(Vehicle.DischargeCurrent.Time));
    Vehicle.DischargeCurrent.SampleTime = 0;
    
    %ChargeCurrent
    RandRatio = 0.8*rand()+0.1; %[0.1 0.9]

    Vehicle.ChargeCurrent.Time = [0:1000:SimTime];
    Vehicle.ChargeCurrent.Amplitude = datasample([1/8 1],numel(Vehicle.ChargeCurrent.Time),'Weights',[RandRatio, 1-RandRatio])*C;
    Vehicle.ChargeCurrent.SampleTime = 0;

    %AmbientTemperature
    climate_type = [1,2,3,4];
    Vehicle.climate = randsample(climate_type,1);
    temp = Climate(Vehicle.climate);
    Vehicle.AmbientTemperature.Time = [0:3600:SimTime];
    sinInput = ((Vehicle.AmbientTemperature.Time)+num*3600)/(SimTime/40);
    Vehicle.AmbientTemperature.Amplitude = rescale(sin(sinInput),temp.min,temp.max);
    Vehicle.AmbientTemperature.SampleTime = 0;

    Vehicle.temp=temp;

end 


%Vehicle Input Distribution Generator
function RandVec = TypeChangeDist(VehicleType, min, max, num, change_time)
    
    if isempty(change_time)
        RandVec = getVehicleDist(VehicleType(1), min, max, num);
    else
        RandVec = zeros(1,num);
        index = 1;
        for i=1:length(change_time)
            numtype = fix(num * change_time(i)) - index + 1;
            RandVec(index: index + numtype-1) = getVehicleDist(VehicleType(i), min, max,numtype) ;
            index = index + numtype;
        end
        RandVec(index: num) = getVehicleDist(VehicleType(end), min, max,num-index+1);
    end
end

function RandVec = getVehicleDist(VehicleType, min, max, num)
     if VehicleType == 4
        R = 0.15*rand()+0.15; %[0.15 0.30]
    else
        R = 0.15*rand()+0.2; %[0.20 0.35]
    end 
    pd = makedist('Rician', 'sigma', R, 's', 0);%
    pd = truncate(pd, 0, 1);
    RandVec = min + (max - min) * (random(pd, 1,num));
    if VehicleType == 1 
     RandVec = max - RandVec + min;
    elseif VehicleType == 3
        RandVec(1:fix(num/2)) = max - RandVec(1:fix(num/2)) + min;
        RandVec = RandVec(randperm(length(RandVec)));
    end
end

function temp = Climate(climateType)

    switch climateType
        case 1 %cold climate
            lowrange=[-25,-5];
            uprange=[10,15];
            temp.min = (lowrange(2)-lowrange(1)) * rand() + lowrange(1);
            temp.max = (uprange(2)-uprange(1)) * rand() + uprange(1);
        case 2 %hot climate
            lowrange=[10,20];
            uprange=[35,45];
            temp.min = (lowrange(2)-lowrange(1)) * rand() + lowrange(1);
            temp.max = (uprange(2)-uprange(1)) * rand() + uprange(1);      
        case 3 %high difference climate
            lowrange=[-20,-15];
            diff=[35,45];
            temp.min = (lowrange(2)-lowrange(1)) * rand() + lowrange(1);
            temp.max = temp.min + ((diff(2)-diff(1)) * rand() + diff(1));    
        case 4 %low difference climate
            lowrange=[20,25];
            diff=[5,10];
            temp.min = (lowrange(2)-lowrange(1)) * rand() + lowrange(1);
            temp.max = temp.min + ((diff(2)-diff(1)) * rand() + diff(1)); 
    end

end 

function Vec = DoCgen (inVec, Max,mindiff)
    B = rand()*1.8+1.7;
    Rand = betarnd(8 , B,size(inVec));
    Lband = min(inVec+mindiff,Max);
    Lband = max(Lband,80);
    Vec = round(((Max - Lband) .* Rand) + Lband);
end

function [type,changing_times] = generate_changing_times(TYPES, rate, simulation_time)
changing_times=exprnd(rate);
type = [randsample(TYPES, 1)];
    while changing_times(end) < simulation_time
        rand = changing_times(end) + exprnd(rate);
        changing_times = [changing_times rand];
    end
    changing_times=changing_times(1:end-1);
    for i=1:length(changing_times)
        type = [type randsample(TYPES(TYPES ~= type(end)), 1)];
    end

end

% function [type,changing_times] = generate_Type_change(TYPES, rate, simulation_time)
%     while true
% 
%         rand = exprnd(rate);
%         if rand<simulation_time
%             changing_times = [rand];
%             type = [randsample(TYPES, 1)];
%             break
%         end
%     end
%     while changing_times(end) < simulation_time
%         rand = changing_times(end) + exprnd(rate);
%         if rand > simulation_time
%             break
%         else
%             changing_times = [changing_times rand];
%             type = [type randsample(TYPES(TYPES ~= type(end)), 1)];
%         end
%     end
% end
% 



    