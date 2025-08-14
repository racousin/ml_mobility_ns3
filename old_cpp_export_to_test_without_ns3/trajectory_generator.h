#ifndef TRAJECTORY_GENERATOR_H
#define TRAJECTORY_GENERATOR_H

#include <torch/script.h>
#include <vector>
#include <string>
#include <map>
#include "json.hpp"

struct Trajectory {
    std::vector<std::vector<double>> points;
    std::string mode;
    int length;
};

class TrajectoryGenerator {
public:
    TrajectoryGenerator(const std::string& model_path,
                        const std::string& metadata_path,
                        const std::string& scalers_path,
                        const std::string& device_str = "cpu");

    std::vector<Trajectory> generate(const std::string& transport_mode, int trip_length, int n_samples = 1);

private:
    void load_model(const std::string& model_path);
    void load_metadata(const std::string& metadata_path);
    void load_scalers(const std::string& scalers_path);
    int mode_to_idx(const std::string& mode);
    std::string idx_to_mode(int idx);
    std::vector<std::vector<double>> inverse_transform(const torch::Tensor& trajectory_tensor);

    torch::jit::script::Module module;
    torch::Device device;
    std::map<std::string, int> mode_to_idx_map;
    std::map<int, std::string> idx_to_mode_map;
    std::vector<double> trajectory_scaler_mean;
    std::vector<double> trajectory_scaler_scale;
    
    // Model configuration
    int sequence_length = 2000;
    int input_dim = 3;
    int latent_dim = 64;
};

#endif // TRAJECTORY_GENERATOR_H