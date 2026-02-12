function [APFD_g, Recall20_g, Effort20_g, permutation_out] = runGreedy_add_indicators(configuration, csv_file)

% csv_file;
T = readtable(csv_file);
Cost = table2array(T(:,19));
Features = table2array(T(:,1:16));

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

% === Sampling logic (to prevent insufficient memory) ===
[m, n] = size(Features);
sample_size = 2000; % For example: limit to 2000 samples
if m > sample_size
    fprintf('Original sample count: %d, too large. Sampling to: %d\n', m, sample_size);
    sample_idx = randperm(m, sample_size);
    Features = Features(sample_idx, :);
    Cost_sampled = Cost(sample_idx);
    safety_labels_sampled = safety_labels(sample_idx);
    is_unsafe_sampled = is_unsafe(sample_idx);
    m = sample_size;
    use_sampling = true;
else
    fprintf('Sample count: %d, no need to sample\n', m);
    Cost_sampled = Cost;
    safety_labels_sampled = safety_labels;
    is_unsafe_sampled = is_unsafe;
    sample_idx = 1:m;
    use_sampling = false;
end

Differences = pdist(Features, "seuclidean");
Differences = squareform(Differences);

maximum = max(max(Differences));
[x, y] = find(Differences == maximum);

firstTest = x(1);
secondTest = x(2);
permutation_size = size(Features, 1);

permutation = [firstTest, secondTest];

for index = 3:permutation_size
    max_avg = 0;
    max_index = 0;
    for c = 1:permutation_size
        if ~ismember(c, permutation)
            dists = 0;
            for already_selected_test_index = 1:(index-1)
                dists = dists + Differences(c, permutation(already_selected_test_index));
            end
            avg = dists / Cost_sampled(c);
            if avg > max_avg
                max_avg = avg;
                max_index = c;
            end
        end
    end
    % Now, we add the detected max
    permutation(index) = max_index;
end

% Fault detection (on sampled data)
T_sampled = T(sample_idx, :);
[a, b] = faultDetection(permutation, T_sampled, Cost_sampled);

% APFDc
APFD_g_single = trapz(a, b) / (max(a) * max(b) + eps);

% === Calculate Recall@20 and Effort@20 using sampled data ===
sorted_is_unsafe_sampled = is_unsafe_sampled(permutation);
sorted_cost_sampled = Cost_sampled(permutation);

cum_cost_sampled = cumsum(sorted_cost_sampled);
cum_faults_sampled = cumsum(sorted_is_unsafe_sampled);

total_faults = sum(is_unsafe_sampled);
total_cost = sum(Cost_sampled);

if total_faults > 0 && total_cost > 0
    % Recall@20
    time_20 = 0.2 * total_cost;
    idx_20 = find(cum_cost_sampled >= time_20, 1, 'first');
    if isempty(idx_20), idx_20 = length(cum_cost_sampled); end
    Recall20_g_single = cum_faults_sampled(idx_20) / total_faults;
    
    % Effort@20
    target_faults_20 = 0.2 * total_faults;
    idx_effort = find(cum_faults_sampled >= target_faults_20, 1, 'first');
    if isempty(idx_effort), idx_effort = length(cum_cost_sampled); end
    Effort20_g_single = cum_cost_sampled(idx_effort) / total_cost;
else
    Recall20_g_single = 0;
    Effort20_g_single = 0;
end

% Create arrays with single values for compatibility with return format
APFD_g = [APFD_g_single];  % Return as array format
Recall20_g = [Recall20_g_single];  % Return as array format
Effort20_g = [Effort20_g_single];  % Return as array format

% === Save sorting details (using original data to obtain labels, etc.) ===
output_dir = "../data_add_indicators/";
[pathstr, name, ext] = fileparts(csv_file);
output_file = fullfile(output_dir, strcat(name, "_greedy_sorted.csv"));

% permutation is the sampled index (1 to sample_size), mapped back to original CSV row numbers
original_indices = sample_idx(permutation);

% Extract corresponding labels and cost from original T
sorted_safety_orig = T{original_indices, 'safety'};
sorted_is_unsafe_orig = is_unsafe(original_indices);
sorted_cost_orig = Cost(original_indices);

cum_cost_orig = cumsum(sorted_cost_orig);
cum_faults_orig = cumsum(sorted_is_unsafe_orig);

result_table = table(...
    (1:length(original_indices))', ...             % Rank
    original_indices(:), ...                        % Original_Index (in original CSV)
    sorted_safety_orig(:), ...                     % Safety_Label
    logical(sorted_is_unsafe_orig(:)), ...         % Is_Unsafe
    sorted_cost_orig(:), ...                       % Duration_Seconds
    cum_cost_orig(:), ...                          % Cumulative_Time
    cum_faults_orig(:), ...                        % Cumulative_Faults
    'VariableNames', {'Rank', 'Original_Index', 'Safety_Label', 'Is_Unsafe', 'Duration_Seconds', 'Cumulative_Time', 'Cumulative_Faults'});

writetable(result_table, output_file);

% Return the permutation (sampled indices)
permutation_out = permutation;

end