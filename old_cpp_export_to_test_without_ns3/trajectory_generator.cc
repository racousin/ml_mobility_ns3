#include "trajectory_generator.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

TrajectoryGenerator::TrajectoryGenerator(const std::string& model_path,
                                         const std::string& metadata_path,
                                         const std::string& scalers_path,
                                         const std::string& device_str)
    : device(device_str) {
    
    std::cout << "Initializing TrajectoryGenerator..." << std::endl;
    std::cout << "Device: " << device_str << std::endl;
    
    load_model(model_path);
    load_metadata(metadata_path);
    load_scalers(scalers_path);
    
    std::cout << "TrajectoryGenerator initialized successfully!" << std::endl;
}

void TrajectoryGenerator::load_model(const std::string& model_path) {
    std::cout << "Loading model from: " << model_path << std::endl;
    
    try {
        module = torch::jit::load(model_path);
        module.eval();
        module.to(device);
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model: " + std::string(e.what()));
    }
}

void TrajectoryGenerator::load_metadata(const std::string& metadata_path) {
    std::cout << "Loading metadata from: " << metadata_path << std::endl;
    
    std::ifstream file(metadata_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open metadata file: " + metadata_path);
    }
    
    json metadata;
    file >> metadata;
    
    // Load transport modes
    auto transport_modes = metadata["transport_modes"];
    for (size_t i = 0; i < transport_modes.size(); ++i) {
        std::string mode = transport_modes[i];
        mode_to_idx_map[mode] = static_cast<int>(i);
        idx_to_mode_map[static_cast<int>(i)] = mode;
    }
    
    std::cout << "Loaded " << transport_modes.size() << " transport modes:" << std::endl;
    for (const auto& pair : mode_to_idx_map) {
        std::cout << "  " << pair.first << " -> " << pair.second << std::endl;
    }
    
    // Update model configuration from metadata
    if (metadata.contains("model_config")) {
        auto config = metadata["model_config"];
        if (config.contains("sequence_length")) {
            sequence_length = config["sequence_length"];
        }
        if (config.contains("input_dim")) {
            input_dim = config["input_dim"];
        }
        if (config.contains("latent_dim")) {
            latent_dim = config["latent_dim"];
        }
    }
    
    std::cout << "Model config - sequence_length: " << sequence_length 
              << ", input_dim: " << input_dim 
              << ", latent_dim: " << latent_dim << std::endl;
}

void TrajectoryGenerator::load_scalers(const std::string& scalers_path) {
    std::cout << "Loading scalers from: " << scalers_path << std::endl;
    
    std::ifstream file(scalers_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open scalers file: " + scalers_path);
    }
    
    json scalers;
    file >> scalers;
    
    // Load trajectory scaler
    if (scalers.contains("trajectory")) {
        auto traj_scaler = scalers["trajectory"];
        
        if (traj_scaler.contains("mean")) {
            trajectory_scaler_mean = traj_scaler["mean"].get<std::vector<double>>();
        }
        if (traj_scaler.contains("scale")) {
            trajectory_scaler_scale = traj_scaler["scale"].get<std::vector<double>>();
        }
        
        std::cout << "Loaded trajectory scaler:" << std::endl;
        std::cout << "  Mean: [";
        for (size_t i = 0; i < trajectory_scaler_mean.size(); ++i) {
            std::cout << trajectory_scaler_mean[i];
            if (i < trajectory_scaler_mean.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "  Scale: [";
        for (size_t i = 0; i < trajectory_scaler_scale.size(); ++i) {
            std::cout << trajectory_scaler_scale[i];
            if (i < trajectory_scaler_scale.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

int TrajectoryGenerator::mode_to_idx(const std::string& mode) {
    auto it = mode_to_idx_map.find(mode);
    if (it == mode_to_idx_map.end()) {
        throw std::runtime_error("Unknown transport mode: " + mode);
    }
    return it->second;
}

std::string TrajectoryGenerator::idx_to_mode(int idx) {
    auto it = idx_to_mode_map.find(idx);
    if (it == idx_to_mode_map.end()) {
        throw std::runtime_error("Unknown mode index: " + std::to_string(idx));
    }
    return it->second;
}

std::vector<std::vector<double>> TrajectoryGenerator::inverse_transform(const torch::Tensor& trajectory_tensor) {
    // Convert tensor to CPU and get data
    auto cpu_tensor = trajectory_tensor.to(torch::kCPU);
    auto accessor = cpu_tensor.accessor<float, 2>();
    
    std::vector<std::vector<double>> result;
    result.reserve(accessor.size(0));
    
    for (int i = 0; i < accessor.size(0); ++i) {
        std::vector<double> point(input_dim);
        for (int j = 0; j < input_dim; ++j) {
            // Inverse transform: x = (x_scaled * scale) + mean
            double scaled_value = static_cast<double>(accessor[i][j]);
            point[j] = (scaled_value * trajectory_scaler_scale[j]) + trajectory_scaler_mean[j];
        }
        result.push_back(point);
    }
    
    return result;
}

std::vector<Trajectory> TrajectoryGenerator::generate(const std::string& transport_mode, 
                                                     int trip_length, 
                                                     int n_samples) {
    std::cout << "Generating " << n_samples << " trajectory(s) for mode '" 
              << transport_mode << "' with length " << trip_length << std::endl;
    
    // Validate inputs
    if (trip_length > sequence_length) {
        throw std::runtime_error("Trip length (" + std::to_string(trip_length) + 
                                ") exceeds maximum sequence length (" + std::to_string(sequence_length) + ")");
    }
    
    int mode_idx = mode_to_idx(transport_mode);
    
    // Prepare input tensors (these will be reused for each sample)
    // The model expects: x (dummy input), transport_mode, length
    auto dummy_input = torch::randn({1, sequence_length, input_dim}, torch::dtype(torch::kFloat32)).to(device);
    auto mode_tensor = torch::tensor({mode_idx}, torch::dtype(torch::kLong)).to(device);
    auto length_tensor = torch::tensor({trip_length}, torch::dtype(torch::kLong)).to(device);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(dummy_input);
    inputs.push_back(mode_tensor);
    inputs.push_back(length_tensor);
    
    std::vector<Trajectory> results;
    results.reserve(n_samples);
    
    // Generate trajectories one by one (since model only generates 1 at a time)
    for (int sample = 0; sample < n_samples; ++sample) {
        torch::Tensor single_trajectory;
        try {
            auto output = module.forward(inputs);
            single_trajectory = output.toTensor();
        } catch (const std::exception& e) {
            throw std::runtime_error("Model inference failed: " + std::string(e.what()));
        }
        
        std::cout << "Generated tensor shape for sample " << (sample + 1) 
                  << ": " << single_trajectory.sizes() << std::endl;
        
        // Extract the trajectory (tensor is [1, seq_len, 3], we want [0])
        auto trajectory_tensor = single_trajectory[0].slice(0, 0, trip_length);
        
        // Inverse transform to real-world coordinates
        auto points = inverse_transform(trajectory_tensor);
        
        Trajectory traj;
        traj.points = points;
        traj.mode = transport_mode;
        traj.length = trip_length;
        
        results.push_back(traj);
    }
    
    std::cout << "Successfully generated " << results.size() << " trajectory(s)" << std::endl;
    return results;
}