function [] = runMOSearch_add_indicators(execution_id, configuration, csv_file, h_status)
%RUNSEARCH Executes a search process according to the given parameters
% global variables for light-weight fitness function evaluation
global A Cost Features H BM;
configuration = "mo-" + configuration;

%% Read the dataset
T = readtable(csv_file);
Cost = table2array(T(:,19));          % test execution time is column 19
Features = table2array(T(:,1:16));    % ignore the time stamp (columns 17, 18)
H = h_status;

% === Extract safety labels (assuming column 17 is 'safety') ===
if ismember('safety', T.Properties.VariableNames)
    safety_labels = T.safety;
    is_unsafe = strcmp(safety_labels, 'unsafe');
elseif size(T,2) >= 17
    safety_labels = T{:,17};
    is_unsafe = strcmp(safety_labels, 'unsafe');
else
    error('Safety label column not found!');
end
total_faults = sum(is_unsafe);
total_cost = sum(Cost);

%% Normalize features (min-max)
[m, n] = size(Features);
for i = 1:n
    min_val = min(Features(:,i));
    max_val = max(Features(:,i));
    if max_val > min_val
        Features(:,i) = (Features(:,i) - min_val) / (max_val - min_val);
    else
        Features(:,i) = zeros(size(Features(:,i)));
    end
end

%% Apply PCA if needed
if ~strcmp("mo-default-GA", configuration)
    [coeff, scores, latent, tsd, variance, mu] = pca(Features);
    if strcmp("mo-10_feature_GA", configuration)
        Features = scores(:,1:10);
    else
        Features = scores(:,1:6);
    end
end

if H
    configuration = "hybrid-" + configuration;
end

[pathstr, name, ext] = fileparts(csv_file);
BM = name;

%% Compute pairwise Euclidean distance
Differences = pdist(Features, "seuclidean");
Differences = squareform(Differences);
A = Differences;

%% Genetic Algorithm setup
options = optimoptions('gamultiobj');
options = optimoptions(options, 'PopulationType', 'custom');
options = optimoptions(options, 'PopulationSize', 200);
options = optimoptions(options, 'MaxGenerations', 4000);
options = optimoptions(options, 'CreationFcn', @initialPopulation);
options = optimoptions(options, 'CrossoverFcn', @permutationCrossover);
options = optimoptions(options, 'MutationFcn', @permutationMutation);
options = optimoptions(options, 'FunctionTolerance', 0);
options = optimoptions(options, 'MaxStallGenerations', 100);
options = optimoptions(options, 'ParetoFraction', 0.8);
options = optimoptions(options, 'Display', 'off');
options = optimoptions(options, 'PlotFcn', { @gaplotpareto });

tic
[x, fval, exitflag, output, population, score] = gamultiobj(@objectives, m, [], [], [], [], [], [], [], options);
elapsed_time = toc;

%% Output directory
output_dir = fullfile("../data_add_indicators", name, configuration, int2str(execution_id), "");
mkdir(output_dir);

%% Save best permutations
writematrix(x, fullfile(output_dir, "best-permutation.csv"));

%% Evaluate Pareto solutions: APFDc, Recall@20, Effort@20
APFD_c = zeros(size(x,1), 1);
Recall20 = zeros(size(x,1), 1);
Effort20 = zeros(size(x,1), 1);

for i = 1:size(x,1)
    perm = x(i,:);
    sorted_is_unsafe = is_unsafe(perm);
    sorted_cost = Cost(perm);
    sorted_safety = safety_labels(perm);
    
    cum_cost = cumsum(sorted_cost);
    cum_faults = cumsum(sorted_is_unsafe);
    
    % APFDc
    if total_faults > 0 && total_cost > 0
        APFD_c(i,1) = trapz(cum_cost, cum_faults) / (total_cost * total_faults);
    else
        APFD_c(i,1) = 0;
    end
    
    % Recall@20 and Effort@20
    if total_faults > 0 && total_cost > 0
        % Recall@20
        time_20 = 0.2 * total_cost;
        idx_20 = find(cum_cost >= time_20, 1, 'first');
        if isempty(idx_20), idx_20 = length(cum_cost); end
        Recall20(i,1) = cum_faults(idx_20) / total_faults;
        
        % Effort@20
        target_faults_20 = 0.2 * total_faults;
        idx_effort = find(cum_faults >= target_faults_20, 1, 'first');
        if isempty(idx_effort), idx_effort = length(cum_cost); end
        Effort20(i,1) = cum_cost(idx_effort) / total_cost;
    else
        Recall20(i,1) = 0;
        Effort20(i,1) = 0;
    end
    
    % === Create sorted details table after fix (ensure all variables are column vectors) ===
    perm_vec = x(i,:)';
    sorted_safety_i = safety_labels(perm_vec);
    sorted_is_unsafe_i = is_unsafe(perm_vec);
    sorted_cost_i = Cost(perm_vec);

    cum_cost_i = cumsum(sorted_cost_i);
    cum_faults_i = cumsum(sorted_is_unsafe_i);

    result_table = table(...
        (1:length(perm_vec))', ...                    % Rank
        perm_vec(:), ...                              % Original_Index
        sorted_safety_i(:), ...                       % Safety_Label
        logical(sorted_is_unsafe_i(:)), ...           % Is_Unsafe
        sorted_cost_i(:), ...                         % Duration_Seconds
        cum_cost_i(:), ...                            % Cumulative_Time
        cum_faults_i(:), ...                          % Cumulative_Faults
        'VariableNames', {'Rank', 'Original_Index', 'Safety_Label', 'Is_Unsafe', 'Duration_Seconds', 'Cumulative_Time', 'Cumulative_Faults'});
    
    writetable(result_table, fullfile(output_dir, sprintf('sorted_pareto_%d.csv', i)));
end

max_ga_APFD = max(APFD_c);
std_ga_APFD = std(APFD_c);

%% Random baseline (500 runs)
rand_APFD = zeros(1,500);
rand_Recall20 = zeros(1,500);
rand_Effort20 = zeros(1,500);

for i = 1:500
    rand_perm = randperm(m);
    sorted_is_unsafe = is_unsafe(rand_perm);
    sorted_cost = Cost(rand_perm);
    
    cum_cost = cumsum(sorted_cost);
    cum_faults = cumsum(sorted_is_unsafe);
    
    if total_faults > 0 && total_cost > 0
        rand_APFD(1,i) = trapz(cum_cost, cum_faults) / (total_cost * total_faults);
        
        time_20 = 0.2 * total_cost;
        idx_20 = find(cum_cost >= time_20, 1, 'first');
        if isempty(idx_20), idx_20 = m; end
        rand_Recall20(1,i) = cum_faults(idx_20) / total_faults;
        
        target_faults_20 = 0.2 * total_faults;
        idx_effort = find(cum_faults >= target_faults_20, 1, 'first');
        if isempty(idx_effort), idx_effort = m; end
        rand_Effort20(1,i) = cum_cost(idx_effort) / total_cost;
    else
        rand_APFD(1,i) = 0;
        rand_Recall20(1,i) = 0;
        rand_Effort20(1,i) = 0;
    end
end

% Save random results
writematrix(rand_APFD, fullfile(output_dir, "rands_APFD.csv"));
writematrix(rand_Recall20, fullfile(output_dir, "rands_Recall20.csv"));
writematrix(rand_Effort20, fullfile(output_dir, "rands_Effort20.csv"));

avg_rand_APFD = mean(rand_APFD,2);
std_rand_APFD = std(rand_APFD);
avg_rand_Recall20 = mean(rand_Recall20,2);
std_rand_Recall20 = std(rand_Recall20);
avg_rand_Effort20 = mean(rand_Effort20,2);
std_rand_Effort20 = std(rand_Effort20);

%% Save final results
mat = ["config", "execution_id", "solution_id", "cost", "diversity", "APFD", "Recall20", "Effort20", ...
       "elapsed_time", ...
       "avg_rand_APFD", "std_rand_APFD", ...
       "avg_rand_Recall20", "std_rand_Recall20", ...
       "avg_rand_Effort20", "std_rand_Effort20"];

for i = 1:size(APFD_c,1)
    mat = [mat; ...
        configuration, ...
        execution_id, ...
        i, ...
        fval(i,1), ...
        fval(i,2), ...
        APFD_c(i), ...
        Recall20(i), ...
        Effort20(i), ...
        elapsed_time, ...
        avg_rand_APFD, std_rand_APFD, ...
        avg_rand_Recall20, std_rand_Recall20, ...
        avg_rand_Effort20, std_rand_Effort20];
end

writematrix(mat, fullfile(output_dir, "results.csv"));

%% Plot
figure;
hold on;
for i = 1:size(x,1)
    perm = x(i,:);
    cum_cost = cumsum(Cost(perm));
    cum_faults = cumsum(is_unsafe(perm));
    plot(cum_cost, cum_faults, 'r');
end
for i = 1:500
    perm = randperm(m);
    cum_cost = cumsum(Cost(perm));
    cum_faults = cumsum(is_unsafe(perm));
    plot(cum_cost, cum_faults, 'b', 'HandleVisibility', 'off');
end
title('Fault detection capability');
xlabel('Execution cost (s)');
ylabel('Number of failures');
hold off;
exportgraphics(gcf, fullfile(output_dir, "plot.png"));

end