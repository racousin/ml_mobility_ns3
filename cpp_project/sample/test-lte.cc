#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

using namespace ns3;

void
CourseChangeCallback(Ptr<const MobilityModel> mobility)
{
    std::cout << Simulator::Now().As(Time::S) << " Movement detected (Node "
              << mobility->GetObject<Node>()->GetId() << ")!\n";
}

void
MoveNode(Ptr<Node> node, const Vector& pos)
{
    Ptr<MobilityModel> mobilityModel = node->GetObject<MobilityModel>();
    NS_ASSERT_MSG(mobilityModel, "Node doesn't have a mobility model");
    mobilityModel->SetPosition(pos);

    Vector newPos = mobilityModel->GetPosition();

    std::cout << Simulator::Now().As(Time::S) << " Node " << node->GetId() << " | MoveRight, Pos ("
              << newPos.x << "," << newPos.y << "," << newPos.z << ")\n";
}

int main(int argc, char *argv[]){
  
  CommandLine cmd;
  cmd.Parse (argc, argv);
  
  Ptr<LteHelper> lteHelper = CreateObject<LteHelper> ();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper> ();
  lteHelper->SetEpcHelper (epcHelper);
  lteHelper->SetAttribute("PathlossModel", StringValue("ns3::Cost231PropagationLossModel"));
  
  NodeContainer enbNodes;
  NodeContainer ueNodes;
  enbNodes.Create (1);
  ueNodes.Create (2);
  
  Ptr<Node> n0 = ueNodes.Get(0);
  Ptr<Node> n1 = ueNodes.Get(1);
  
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (enbNodes);
  
  Ptr<ListPositionAllocator> positionAllocator = CreateObject<ListPositionAllocator>();
    positionAllocator->Add(Vector(0, 0, 0));
  mobility.SetPositionAllocator(positionAllocator);
  mobility.Install (n0);
  
  Ptr<ConstantPositionMobilityModel> mob1 = CreateObject<ConstantPositionMobilityModel>();
  n1->AggregateObject(mob1);
  mob1->SetPosition(Vector(0, 0, 0));
  mob1->TraceConnectWithoutContext("CourseChange", MakeBoundCallback(&CourseChangeCallback));
  Simulator::ScheduleWithContext(n1->GetId(), Seconds(2), &MoveNode, n1, Vector(100, 0, 0));
  Simulator::ScheduleWithContext(n1->GetId(), Seconds(4), &MoveNode, n1, Vector(200, 0, 0));
  Simulator::ScheduleWithContext(n1->GetId(), Seconds(6), &MoveNode, n1, Vector(300, 0, 0));
  
  NetDeviceContainer enbDevs;
  NetDeviceContainer ueDevs;
  enbDevs = lteHelper->InstallEnbDevice (enbNodes);
  ueDevs = lteHelper->InstallUeDevice (ueNodes);
  
  InternetStackHelper tcpip;
  tcpip.Install (ueNodes);
  
  Ipv4InterfaceContainer ueIpAddrs;
  ueIpAddrs = epcHelper->AssignUeIpv4Address (ueDevs);
  
  lteHelper->Attach (ueDevs, enbDevs.Get (0));
  
  UdpEchoServerHelper echoServer(9);
  ApplicationContainer serverApps = echoServer.Install (ueNodes.Get(1));
  serverApps.Start (Seconds(1.0));
  serverApps.Stop (Seconds(10.0));
  
  UdpEchoClientHelper echoClient (ueIpAddrs.GetAddress(1), 9);
  echoClient.SetAttribute ("MaxPackets", UintegerValue(1));
  echoClient.SetAttribute ("Interval", TimeValue(Seconds (1.0)));
  echoClient.SetAttribute ("PacketSize", UintegerValue(1024));
  ApplicationContainer clientApps = echoClient.Install (ueNodes.Get(0));
  clientApps.Start (Seconds (6.0));
  clientApps.Stop (Seconds (10.0));
  
  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
  lteHelper->EnableTraces ();
  
  Simulator::Stop (Seconds(10.0));
  Simulator::Run();
  Simulator::Destroy();
  return 0;
}
