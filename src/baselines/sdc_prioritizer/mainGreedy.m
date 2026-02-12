benchmarks = [
    "../datasets/selected_csv_dataset/BeamNG_RF_0_7_Complete.csv"
    "../datasets/fullroad/BeamNG_AI/BeamNG_RF_1/BeamNG_RF_1_Complete.csv" 
    "../datasets/selected_csv_dataset/BeamNG_RF_1_2_Complete.csv"
    "../datasets/fullroad/BeamNG_AI/BeamNG_RF_1_5/BeamNG_RF_1_5_selected.csv"
    "../datasets/selected_csv_dataset/BeamNG_RF_1_7_Complete.csv"
    "../datasets/fullroad/BeamNG_AI/BeamNG_RF_2/BeamNG_RF_2_Complete.csv"
    "../datasets/fullroad/Driver_AI/DriverAI_Complete.csv"
    ];

mat = ["benchmark", "config", ...
       "APFD_mean", "APFD_median", ...
       "Recall20_mean", "Recall20_median", ...
       "Effort20_mean", "Effort20_median", ...
       "elapsed_time"];

results_data = [];

for benchmark_index = 1 : length(benchmarks)
    tic
    [APFD, Recall20, Effort20, ~] = runGreedy_add_indicators("greedy", benchmarks(benchmark_index));
    elapsed_time = toc;
    [filepath, name, ext] = fileparts(benchmarks(benchmark_index));
    
    APFD_mean = mean(APFD); APFD_median = median(APFD);
    Recall20_mean = mean(Recall20); Recall20_median = median(Recall20);
    Effort20_mean = mean(Effort20); Effort20_median = median(Effort20);
    
    results_data = [results_data; ...
                    name, "Greedy", ...
                    num2str(APFD_mean), num2str(APFD_median), ...
                    num2str(Recall20_mean), num2str(Recall20_median), ...
                    num2str(Effort20_mean), num2str(Effort20_median), ...
                    num2str(elapsed_time)];
end

mat = [mat; results_data];

output_dir = "../data_add_indicators/";
writematrix(mat, fullfile(output_dir, "greedy_results_new.csv"));