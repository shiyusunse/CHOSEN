%benchmarks = ["../datasets/fullroad/Driver_AI/DriverAI_Complete.csv"];
%benchmarks = ["../datasets/fullroad/BeamNG_AI/BeamNG_RF_1_5/BeamNG_RF_1_5_selected.csv"]
benchmarks = ["../datasets/selected_csv_dataset/BeamNG_RF_0_7_Complete.csv"] 
configurations = ["10_feature_GA"]
is_hybrid = [false]
for h_status = 1 : length(is_hybrid)
    for benchmark_index = 1 : length(benchmarks)
        for config_index = 1 : length(configurations)
            for i = 25:30
                runSearch_add_indicators(i,configurations(config_index),benchmarks(benchmark_index),is_hybrid(h_status))
            end
        end
    end
end
