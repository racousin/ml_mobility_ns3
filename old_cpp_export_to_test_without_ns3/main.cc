#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include "trajectory_generator.h"

int main(int argc, char* argv[]) {
    std::cout << "=== Netmob25 Trajectory Generator Test ===" << std::endl;
    
    try {
        // Initialize the generator
        TrajectoryGenerator generator(
            "model.pt",        // TorchScript model
            "metadata.json",   // Metadata
            "scalers.json",    // Scalers
            "cpu"             // Device
        );
        
        // Test with different transport modes
        std::vector<std::string> modes = {"WALKING", "CAR", "BIKE"};
        int trip_length = 100;  // Generate 100 points
        
        for (const auto& mode : modes) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Generating trajectory for mode: " << mode << std::endl;
            std::cout << "========================================" << std::endl;
            
            // Generate one trajectory
            auto trajectories = generator.generate(mode, trip_length, 1);
            
            if (!trajectories.empty()) {
                const auto& traj = trajectories[0];
                std::cout << "Generated trajectory with " << traj.points.size() << " points" << std::endl;
                
                // Print first 10 points and last 10 points to show movement
                std::cout << "\nFirst 10 points (lat, lon, speed):" << std::endl;
                for (size_t i = 0; i < std::min(size_t(10), traj.points.size()); ++i) {
                    std::cout << "  Point " << std::setw(3) << i << ": ";
                    std::cout << std::fixed << std::setprecision(6);
                    std::cout << "(" << traj.points[i][0] << ", " 
                              << traj.points[i][1] << ", " 
                              << std::setprecision(2) << traj.points[i][2] << ")" << std::endl;
                }
                
                if (traj.points.size() > 20) {
                    std::cout << "\n  ... (middle points omitted) ...\n" << std::endl;
                    
                    std::cout << "Last 10 points (lat, lon, speed):" << std::endl;
                    for (size_t i = traj.points.size() - 10; i < traj.points.size(); ++i) {
                        std::cout << "  Point " << std::setw(3) << i << ": ";
                        std::cout << std::fixed << std::setprecision(6);
                        std::cout << "(" << traj.points[i][0] << ", " 
                                  << traj.points[i][1] << ", " 
                                  << std::setprecision(2) << traj.points[i][2] << ")" << std::endl;
                    }
                }
                
                // Calculate total distance traveled
                double total_distance = 0.0;
                for (size_t i = 1; i < traj.points.size(); ++i) {
                    double dlat = traj.points[i][0] - traj.points[i-1][0];
                    double dlon = traj.points[i][1] - traj.points[i-1][1];
                    double dist = std::sqrt(dlat*dlat + dlon*dlon);
                    total_distance += dist;
                }
                
                std::cout << "\nTrajectory statistics:" << std::endl;
                std::cout << "  Mode: " << traj.mode << std::endl;
                std::cout << "  Points: " << traj.points.size() << std::endl;
                std::cout << "  Approx total movement: " << std::fixed << std::setprecision(6) 
                          << total_distance << " degrees" << std::endl;
                
                // Check if trajectory is actually moving
                if (total_distance > 0.0001) {
                    std::cout << "  Status: MOVING ✓" << std::endl;
                } else {
                    std::cout << "  Status: STATIONARY ✗" << std::endl;
                }
            }
            
            // Small delay between modes
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        std::cout << "\n=== Test completed successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}