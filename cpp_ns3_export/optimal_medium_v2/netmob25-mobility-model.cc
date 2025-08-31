/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2024 optimal_medium_v2 - ML Mobility NS3 Project
 */

#include "netmob25-mobility-model.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "geographic-positions.h"

#ifdef HAVE_LIBTORCH
#include <torch/script.h>
#include <iostream>
#endif

namespace ns3 {

#ifdef HAVE_LIBTORCH
// Implementation class definition - keeps LibTorch types private
class Netmob25MobilityModel::Impl {
public:
  torch::jit::script::Module model;
  torch::Device device = torch::kCPU;
  bool modelLoaded = false;
  
  Impl() : device(torch::kCPU), modelLoaded(false) {}
};
#endif

NS_LOG_COMPONENT_DEFINE ("Netmob25MobilityModel");

NS_OBJECT_ENSURE_REGISTERED (Netmob25MobilityModel);

TypeId
Netmob25MobilityModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Netmob25MobilityModel")
    .SetParent<MobilityModel> ()
    .SetGroupName ("Mobility")
    .AddConstructor<Netmob25MobilityModel> ()
    .AddAttribute ("StartTime",
                   "Time to start mobility",
                   TimeValue (Seconds (0.0)),
                   MakeTimeAccessor (&Netmob25MobilityModel::m_startTime),
                   MakeTimeChecker ())
    .AddAttribute ("UpdateInterval", 
                   "Time interval between position updates",
                   TimeValue (Seconds (1.0)),
                   MakeTimeAccessor (&Netmob25MobilityModel::m_updateInterval),
                   MakeTimeChecker ())
    .AddAttribute ("ScaleX",
                   "Deprecated - no longer used",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&Netmob25MobilityModel::m_scaleX),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("ScaleY",
                   "Deprecated - no longer used", 
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&Netmob25MobilityModel::m_scaleY),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("ModelPath",
                   "Path to the ML model file (model.pt)",
                   StringValue ("model.pt"),
                   MakeStringAccessor (&Netmob25MobilityModel::m_modelPath),
                   MakeStringChecker ())
    .AddAttribute ("TransportMode",
                   "Transport mode for trajectory generation (WALKING, CYCLING, DRIVING, TRANSPORT)",
                   StringValue ("WALKING"),
                   MakeStringAccessor (&Netmob25MobilityModel::m_transportMode),
                   MakeStringChecker ())
    .AddAttribute ("TripLength",
                   "Number of trajectory points to generate (max 2000)",
                   UintegerValue (200),
                   MakeUintegerAccessor (&Netmob25MobilityModel::m_tripLength),
                   MakeUintegerChecker<uint32_t> (1, 2000));
  return tid;
}

Netmob25MobilityModel::Netmob25MobilityModel ()
  : m_currentStep (0),
    m_totalSteps (0),
    m_startTime (Seconds (0.0)),
    m_updateInterval (Seconds (1.0)),
    m_scaleX (1.0),
    m_scaleY (1.0),
    m_modelPath ("model.pt"),
    m_transportMode ("WALKING"),
    m_tripLength (200),
    m_currentPosition (Vector (0.0, 0.0, 0.0)),
    m_currentVelocity (Vector (0.0, 0.0, 0.0)),
    m_lastUpdate (Seconds (0.0)),
    m_initialized (false)
{
  NS_LOG_FUNCTION (this);
#ifdef HAVE_LIBTORCH
  m_pImpl = std::make_unique<Impl>();
#endif
}

Netmob25MobilityModel::~Netmob25MobilityModel ()
{
  NS_LOG_FUNCTION (this);
  if (m_updateEvent.IsPending ())
    {
      m_updateEvent.Cancel ();
    }
}

void
Netmob25MobilityModel::SetTrajectory (const std::vector<Vector>& trajectory)
{
  NS_LOG_FUNCTION (this << trajectory.size ());
  m_trajectory = trajectory;
  m_totalSteps = trajectory.size ();
  m_currentStep = 0;
  
  if (!m_trajectory.empty ())
    {
      m_currentPosition = m_trajectory[0];
      NotifyCourseChange ();
    }
}

void
Netmob25MobilityModel::SetModelPath (const std::string& modelPath)
{
  NS_LOG_FUNCTION (this << modelPath);
  m_modelPath = modelPath;
}

void
Netmob25MobilityModel::SetSimulationParams (Time startTime, Time updateInterval, double scaleX, double scaleY)
{
  NS_LOG_FUNCTION (this << startTime << updateInterval << scaleX << scaleY);
  m_startTime = startTime;
  m_updateInterval = updateInterval;
  m_scaleX = scaleX;
  m_scaleY = scaleY;
}

Vector
Netmob25MobilityModel::DoGetPosition (void) const
{
  NS_LOG_FUNCTION (this);
  
  if (!m_initialized)
    {
      // Initialize on first access
      const_cast<Netmob25MobilityModel*>(this)->Initialize ();
    }
    
  return m_currentPosition;
}

void
Netmob25MobilityModel::DoSetPosition (const Vector& position)
{
  NS_LOG_FUNCTION (this << position);
  m_currentPosition = position;
  m_lastUpdate = Simulator::Now ();
  NotifyCourseChange ();
}

Vector
Netmob25MobilityModel::DoGetVelocity (void) const
{
  NS_LOG_FUNCTION (this);
  return m_currentVelocity;
}

void
Netmob25MobilityModel::Initialize (void)
{
  NS_LOG_FUNCTION (this);
  
  if (m_initialized)
    {
      return;
    }
  
  // Always generate trajectory using ML model if empty
  if (m_trajectory.empty ())
    {
      NS_LOG_INFO ("Generating trajectory with ML model");
      GenerateMLTrajectory ();
      
      if (m_trajectory.empty ())
        {
          NS_LOG_ERROR ("Failed to generate ML trajectory - no fallback available");
        }
    }
  
  // Set initial position
  if (!m_trajectory.empty ())
    {
      m_currentPosition = m_trajectory[0];
    }
  
  // Schedule first update
  Time delay = m_startTime > Simulator::Now () ? m_startTime - Simulator::Now () : Seconds (0);
  m_updateEvent = Simulator::Schedule (delay, &Netmob25MobilityModel::UpdatePosition, this);
  
  m_initialized = true;
  m_lastUpdate = Simulator::Now ();
  
  NotifyCourseChange ();
  
  NS_LOG_INFO ("Netmob25MobilityModel initialized with " << m_totalSteps << " trajectory points");
}

void
Netmob25MobilityModel::UpdatePosition (void)
{
  NS_LOG_FUNCTION (this);
  
  if (m_currentStep < m_totalSteps)
    {
      Vector oldPosition = m_currentPosition;
      
      // Update position from trajectory
      m_currentPosition = m_trajectory[m_currentStep];
      
      // Calculate velocity
      Time timeDelta = m_updateInterval;
      if (timeDelta.GetSeconds () > 0)
        {
          m_currentVelocity = Vector (
            (m_currentPosition.x - oldPosition.x) / timeDelta.GetSeconds (),
            (m_currentPosition.y - oldPosition.y) / timeDelta.GetSeconds (),
            0.0
          );
        }
      
      m_currentStep++;
      m_lastUpdate = Simulator::Now ();
      
      NotifyCourseChange ();
      
      // Schedule next update if more steps available
      if (m_currentStep < m_totalSteps)
        {
          m_updateEvent = Simulator::Schedule (m_updateInterval, &Netmob25MobilityModel::UpdatePosition, this);
        }
      
      NS_LOG_DEBUG ("Updated position to (" << m_currentPosition.x << ", " << m_currentPosition.y << 
                   ") at step " << m_currentStep << "/" << m_totalSteps);
    }
}

void
Netmob25MobilityModel::GenerateMLTrajectory (void)
{
  NS_LOG_FUNCTION (this);
  
#ifdef HAVE_LIBTORCH
  try
    {
      NS_LOG_INFO ("Generating trajectory using ML model: " << m_modelPath);
      
      // Load model if not already loaded
      if (!m_pImpl->modelLoaded)
        {
          LoadModel ();
        }
      
      // Generate trajectory points using configured trip length
      auto points = GenerateFromModel (m_tripLength);
      
      if (!points.empty ())
        {
          m_trajectory.clear ();
          
          // Convert generated points to NS-3 coordinates
          // Reference point (Paris coordinates as specified)
          double refLat = 48.852737;
          double refLon = 2.350699;
          double refAlt = 0.0;
          
          // COMMENTED OUT: Convert reference point to Cartesian coordinates
          Vector refCartesian = GeographicPositions::GeographicToCartesianCoordinates (
            refLat, refLon, refAlt, GeographicPositions::WGS84);
          
          for (const auto& point : points)
            {
              // Unscale the model outputs using StandardScaler parameters
              // From data/processed_old/scalers.pkl: mean=[48.82317302, 2.35717274, 20.50435739], scale=[0.12837157, 0.1965312, 29.59223145]
              double lat = point[0] * 0.12837157 + 48.82317302;  // unscale latitude
              double lon = point[1] * 0.1965312 + 2.35717274;    // unscale longitude
              double alt = 0.0;  // altitude
              
              // COMMENTED OUT: Convert from geographic to Cartesian coordinates
              Vector cartesianPos = GeographicPositions::GeographicToCartesianCoordinates (
                lat, lon, alt, GeographicPositions::WGS84);
              
              // Apply reference point offset to get relative coordinates
              Vector relativePos = Vector (
                cartesianPos.x - refCartesian.x,
                cartesianPos.y - refCartesian.y,
                cartesianPos.z - refCartesian.z
              );
              
              // Directly use lat/lon for now to check model output
              // Add to trajectory
              m_trajectory.push_back (relativePos);
            }
          
          m_totalSteps = m_trajectory.size ();
          NS_LOG_INFO ("Generated ML trajectory with " << m_totalSteps << " points");
        }
      else
        {
          NS_LOG_ERROR ("Failed to generate trajectory - empty result");
        }
    }
  catch (const std::exception& e)
    {
      NS_LOG_ERROR ("Failed to generate ML trajectory: " << e.what ());
    }
#else
  NS_LOG_WARN ("LibTorch not available, cannot use ML generation");
#endif
}

#ifdef HAVE_LIBTORCH
void
Netmob25MobilityModel::LoadModel (void)
{
  NS_LOG_FUNCTION (this);
  
  try
    {
      // Load the TorchScript model
      m_pImpl->model = torch::jit::load (m_modelPath);
      m_pImpl->model.to (m_pImpl->device);
      m_pImpl->model.eval ();
      
      m_pImpl->modelLoaded = true;
      NS_LOG_INFO ("Model loaded successfully from: " << m_modelPath);
    }
  catch (const c10::Error& e)
    {
      NS_LOG_ERROR ("Error loading model: " << e.what ());
      throw std::runtime_error ("Error loading model: " + std::string (e.what ()));
    }
}

std::vector<std::vector<double>>
Netmob25MobilityModel::GenerateFromModel (int trip_length)
{
  NS_LOG_FUNCTION (this << trip_length);
  
  if (!m_pImpl->modelLoaded)
    {
      throw std::runtime_error ("Model not loaded!");
    }
  
  std::vector<std::vector<double>> trajectory;
  
  // Disable gradient computation for inference
  torch::NoGradGuard no_grad;
  
  try
    {
      // Create input trajectory tensor (random noise as starting point)
      // The model expects (batch_size, sequence_length, input_dim)
      int sequence_length = 2000;  // Max sequence length the model supports
      int input_dim = 3;  // lat, lon, speed
      torch::Tensor x = torch::randn ({1, sequence_length, input_dim}, m_pImpl->device);
      
      // Map transport mode string to index
      int mode_idx = 0;  // Default to WALKING
      if (m_transportMode == "WALKING") mode_idx = 0;
      else if (m_transportMode == "CYCLING") mode_idx = 1;
      else if (m_transportMode == "DRIVING") mode_idx = 2;
      else if (m_transportMode == "TRANSPORT") mode_idx = 3;
      else
        {
          NS_LOG_WARN ("Unknown transport mode: " << m_transportMode << ", using WALKING");
        }
      
      // Create transport mode tensor
      torch::Tensor transport_mode = torch::tensor ({mode_idx}, torch::kLong).to (m_pImpl->device);
      
      // Create length tensor
      torch::Tensor length = torch::tensor ({trip_length}, torch::kLong).to (m_pImpl->device);
      
      // Forward pass through the model
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back (x);
      inputs.push_back (transport_mode);
      inputs.push_back (length);
      
      // Get output
      auto output = m_pImpl->model.forward (inputs);
      torch::Tensor traj_tensor;
      
      // Handle different output types
      if (output.isTensor ())
        {
          traj_tensor = output.toTensor ();
        }
      else if (output.isTuple ())
        {
          auto tuple = output.toTuple ();
          if (tuple->elements ().size () > 0)
            {
              traj_tensor = tuple->elements ()[0].toTensor ();
            }
        }
      
      // Move to CPU and convert to vector
      traj_tensor = traj_tensor.to (torch::kCPU);
      
      // Check tensor dimensions and extract data
      if (traj_tensor.dim () == 3)
        {
          // [batch, seq_len, features]
          auto accessor = traj_tensor.accessor<float, 3> ();
          for (int i = 0; i < accessor.size (1) && i < trip_length; i++)
            {
              std::vector<double> point;
              // Extract lat and lon (first two features)
              point.push_back (static_cast<double> (accessor[0][i][0])); // lat
              point.push_back (static_cast<double> (accessor[0][i][1])); // lon
              trajectory.push_back (point);
            }
        }
      else if (traj_tensor.dim () == 2)
        {
          // [seq_len, features]
          auto accessor = traj_tensor.accessor<float, 2> ();
          for (int i = 0; i < accessor.size (0) && i < trip_length; i++)
            {
              std::vector<double> point;
              point.push_back (static_cast<double> (accessor[i][0])); // lat
              point.push_back (static_cast<double> (accessor[i][1])); // lon
              trajectory.push_back (point);
            }
        }
      
      NS_LOG_INFO ("Generated " << trajectory.size () << " trajectory points");
    }
  catch (const c10::Error& e)
    {
      NS_LOG_ERROR ("Error during inference: " << e.what ());
      throw std::runtime_error ("Error during inference: " + std::string (e.what ()));
    }
  
  return trajectory;
}
#endif

} // namespace ns3