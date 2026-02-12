function [] = runSearch_add_indicators(execution_id, configuration, csv_file, h_status)
%RUNSEARCH Executes a search process according to the given parameters
% global variables for light-weight fitness function evaluation
global A Cost Features H BM;

%% Read the dataset
T = readtable(csv_file);
Cost = table2array(T(:,19));          % test execution time is column 19
Features = table2array(T(:,1:16));    % ignore the time stamp (columns 17, 18)

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

H = h_status;

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

if ~strcmp("default-GA", configuration)
    [coeff, scores, latent, tsd, variance, mu] = pca(Features);
    if strcmp("10_feature_GA", configuration)
        Features = scores(:,1:10);
    else
        Features = scores(:,1:6);
    end
end

[pathstr, name, ext] = fileparts(csv_file);
BM = name;

%% Compute pairwise (Euclidean) distance between test cases
Differences = pdist(Features, "seuclidean");
Differences = squareform(Differences);
A = Differences;

%% Genetic Algorithm setup (Single Objective)
options = optimoptions('ga');
options = optimoptions(options, 'PopulationType', 'custom');
options = optimoptions(options, 'PopulationSize', 100);
options = optimoptions(options, 'MaxGenerations', 4000);
options = optimoptions(options, 'CreationFcn', @initialPopulation);
options = optimoptions(options, 'SelectionFcn', @selectionroulette);
options = optimoptions(options, 'CrossoverFcn', @permutationCrossover);
options = optimoptions(options, 'MutationFcn', @permutationMutation);
options = optimoptions(options, 'FunctionTolerance', 0);
options = optimoptions(options, 'MaxStallGenerations', 100);
options = optimoptions(options, 'Display', 'off');
options = optimoptions(options, 'PlotFcn', { @gaplotbestf, @gaplotscorediversity });

tic
[x, fval, exitflag, output, population, score] = ga(@fitness, m, [], [], [], [], [], [], [], [], options);
elapsed_time = toc;

%% Output directory
output_dir = fullfile("../data_add_indicators", name, configuration, int2str(execution_id), "");
mkdir(output_dir);

%% Fault detection analysis for the best solution (x)
[a, b] = faultDetection(x, T, Cost);
plot(a, b, 'r');
hold on

% APFDc
APFD_c = trapz(a, b) / (max(a) * max(b) + eps);

% === New: Recall@20 and Effort@20 ===
sorted_is_unsafe = is_unsafe(x);
sorted_cost = Cost(x);

cum_cost = cumsum(sorted_cost);
cum_faults = cumsum(sorted_is_unsafe);

if total_faults > 0 && total_cost > 0
    % Recall@20
    time_20 = 0.2 * total_cost;
    idx_20 = find(cum_cost >= time_20, 1, 'first');
    if isempty(idx_20), idx_20 = length(cum_cost); end
    Recall20 = cum_faults(idx_20) / total_faults;
    
    % Effort@20
    target_faults_20 = 0.2 * total_faults;
    idx_effort = find(cum_faults >= target_faults_20, 1, 'first');
    if isempty(idx_effort), idx_effort = length(cum_cost); end
    Effort20 = cum_cost(idx_effort) / total_cost;
else
    Recall20 = 0;
    Effort20 = 0;
end

% === Save sorting details (for manual review) ===
perm_vec = x(:); % Ensure column vector
sorted_safety = safety_labels(perm_vec);
sorted_is_unsafe_vec = sorted_is_unsafe(:);
sorted_cost_vec = sorted_cost(:);

result_table = table(...
    (1:length(perm_vec))', ...                    % Rank
    perm_vec(:), ...                              % Original_Index
    sorted_safety(:), ...                         % Safety_Label
    logical(sorted_is_unsafe_vec(:)), ...         % Is_Unsafe
    sorted_cost_vec(:), ...                       % Duration_Seconds
    cum_cost(:), ...                              % Cumulative_Time
    cum_faults(:), ...                            % Cumulative_Faults
    'VariableNames', {'Rank', 'Original_Index', 'Safety_Label', 'Is_Unsafe', 'Duration_Seconds', 'Cumulative_Time', 'Cumulative_Faults'});

writetable(result_table, fullfile(output_dir, "best_sorted.csv"));

%% Random baseline (500 runs)
rand_APFD = zeros(1,500);
rand_Recall20 = zeros(1,500);
rand_Effort20 = zeros(1,500);

for i = 1:500
    rand_perm = randperm(m);
    [a, b] = faultDetection(rand_perm, T, Cost);
    plot(a, b, 'b', 'HandleVisibility', 'off');
    rand_APFD(1,i) = trapz(a, b) / (max(a) * max(b) + eps);
    
    % For random baseline
    sorted_is_unsafe_rand = is_unsafe(rand_perm);
    sorted_cost_rand = Cost(rand_perm);
    
    cum_cost_rand = cumsum(sorted_cost_rand);
    cum_faults_rand = cumsum(sorted_is_unsafe_rand);
    
    if total_faults > 0 && total_cost > 0
        % Recall@20
        time_20_rand = 0.2 * total_cost;
        idx_20_rand = find(cum_cost_rand >= time_20_rand, 1, 'first');
        if isempty(idx_20_rand), idx_20_rand = m; end
        rand_Recall20(1,i) = cum_faults_rand(idx_20_rand) / total_faults;
        
        % Effort@20
        target_faults_20_rand = 0.2 * total_faults;
        idx_effort_rand = find(cum_faults_rand >= target_faults_20_rand, 1, 'first');
        if isempty(idx_effort_rand), idx_effort_rand = m; end
        rand_Effort20(1,i) = cum_cost_rand(idx_effort_rand) / total_cost;
    else
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
mat = ["config", "execution_id", "best_fitness_value", "APFD", "Recall20", "Effort20", ...
       "elapsed_time", ...
       "avg_rand_APFD", "std_rand_APFD", ...
       "avg_rand_Recall20", "std_rand_Recall20", ...
       "avg_rand_Effort20", "std_rand_Effort20"];

mat = [mat; ...
    configuration, ...
    execution_id, ...
    fval, ...
    APFD_c, ...
    Recall20, ...
    Effort20, ...
    elapsed_time, ...
    avg_rand_APFD, std_rand_APFD, ...
    avg_rand_Recall20, std_rand_Recall20, ...
    avg_rand_Effort20, std_rand_Effort20];

writematrix(mat, fullfile(output_dir, "results.csv"));

title('Fault detection capability');
xlabel('Execution cost (s)');
ylabel('Number of failures');

hold off;
exportgraphics(gcf, fullfile(output_dir, "plot.png"));

end