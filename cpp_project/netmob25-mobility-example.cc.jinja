/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Simple test for Netmob25 trajectory generation
 */

#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netmob25-mobility-model.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Netmob25SimpleTest");

int
main (int argc, char *argv[])
{
  uint32_t nNodes = 1;
  double simTime = 10.0;
  
  CommandLine cmd (__FILE__);
  cmd.AddValue ("nNodes", "Number of nodes", nNodes);
  cmd.AddValue ("simTime", "Simulation time (seconds)", simTime);
  cmd.Parse (argc, argv);

  // Enable debug logging
  LogComponentEnable ("Netmob25MobilityModel", LOG_LEVEL_INFO);

  std::cout << "=== Netmob25 Simple Trajectory Test ===" << std::endl;
  std::cout << "Testing trajectory generation with " << nNodes << " nodes" << std::endl;
  std::cout << "Simulation time: " << simTime << " seconds" << std::endl;
  std::cout << "Model path: model.pt" << std::endl;
  std::cout << "========================================" << std::endl;

  // Create nodes
  NodeContainer nodes;
  nodes.Create (nNodes);

  // Setup mobility with Netmob25MobilityModel
  MobilityHelper mobility;
  
  // Configure the model with explicit parameters
  mobility.SetMobilityModel ("ns3::Netmob25MobilityModel",
                            "StartTime", TimeValue (Seconds (0.0)),
                            "UpdateInterval", TimeValue (Seconds (2)),  // Update every 2s
                            "ModelPath", StringValue ("model.pt"),
                            "TransportMode", StringValue ("WALKING"),
                            "TripLength", UintegerValue (100));
  
  mobility.Install (nodes);

  // Position printing function
  auto printPositions = [&nodes, nNodes]() {
    std::cout << "\n[" << Simulator::Now ().GetSeconds () << "s] Node positions:" << std::endl;
    for (uint32_t i = 0; i < nNodes; ++i)
      {
        Ptr<MobilityModel> mob = nodes.Get (i)->GetObject<MobilityModel> ();
        if (mob)
          {
            Vector pos = mob->GetPosition ();
            Vector vel = mob->GetVelocity ();
            std::cout << "  Node " << i << ": Position(" << pos.x << ", " << pos.y << ", " << pos.z 
                      << ") Velocity(" << vel.x << ", " << vel.y << ", " << vel.z << ")" << std::endl;
          }
        else
          {
            std::cout << "  Node " << i << ": No mobility model!" << std::endl;
          }
      }
  };

  // Print initial positions
  std::cout << "\nInitial positions:" << std::endl;
  printPositions();

  // Schedule position printing every second
  for (double t = 1.0; t <= simTime; t += 1.0)
    {
      Simulator::Schedule (Seconds (t), printPositions);
    }

  // Run simulation
  std::cout << "\nStarting simulation..." << std::endl;
  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  
  // Print final positions
  std::cout << "\nFinal positions:" << std::endl;
  printPositions();
  
  Simulator::Destroy ();

  std::cout << "\nSimulation completed!" << std::endl;
  
  return 0;
}