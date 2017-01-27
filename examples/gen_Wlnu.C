// Modified from the example file by Chris Rogan

#define COMPILER (!defined(__CINT__) && !defined(__CLING__))
#if defined(__MAKECINT__) || defined(__ROOTCLING__) || COMPILER
#include "RestFrames/RestFrames.hh"
#else
RestFrames::RFKey ensure_autoload(1);
#endif

#include <TFile.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <TVector3.h>

using namespace RestFrames;

void gen_Wlnu(const std::string& output_name = "Wlnu.root"){

  double mW = 80.385; // GeV, PDG 2016
  double wW = 2.085;
  
  // Number of events to generate
  int Ngen = 1000000;

  /////////////////////////////////////////////////////////////////////////////////////////
  g_Log << LogInfo << "Initializing generator frames and tree..." << LogEnd;
  /////////////////////////////////////////////////////////////////////////////////////////
  LabGenFrame       LAB_Gen("LAB_Gen","LAB");
  ResonanceGenFrame W_Gen("W_Gen","W");
  VisibleGenFrame   L_Gen("L_Gen","#it{l}");
  InvisibleGenFrame NU_Gen("NU_Gen","#nu");

  //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
  
  LAB_Gen.SetChildFrame(W_Gen);
  W_Gen.AddChildFrame(L_Gen);
  W_Gen.AddChildFrame(NU_Gen);
 
  if(LAB_Gen.InitializeTree())
    g_Log << LogInfo << "...Successfully initialized generator tree" << LogEnd;
  else
    g_Log << LogError << "...Failed initializing generator tree" << LogEnd;

  //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
                
  // set W pole mass and width
  W_Gen.SetMass(mW);                   W_Gen.SetWidth(wW);

  // set lepton pT and eta cuts
  L_Gen.SetPtCut(20.);                 L_Gen.SetEtaCut(2.5);  

  if(LAB_Gen.InitializeAnalysis())
    g_Log << LogInfo << "...Successfully initialized generator analysis" << std::endl << LogEnd;
  else
    g_Log << LogError << "...Failed initializing generator analysis" << LogEnd;
  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////////////////
  g_Log << LogInfo << "Initializing reconstruction frames and trees..." << LogEnd;
  /////////////////////////////////////////////////////////////////////////////////////////
  LabRecoFrame       LAB("LAB","LAB");
  DecayRecoFrame     W("W","W");
  VisibleRecoFrame   L("L","#it{l}");
  InvisibleRecoFrame NU("NU","#nu");

  //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
  
  LAB.SetChildFrame(W);
  W.AddChildFrame(L);
  W.AddChildFrame(NU);

  if(LAB.InitializeTree())
    g_Log << LogInfo << "...Successfully initialized reconstruction trees" << LogEnd;
  else
    g_Log << LogError << "...Failed initializing reconstruction trees" << LogEnd;

  //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

  // Now we add invisible jigsaws
  InvisibleGroup INV("INV","Neutrino Jigsaws");
  INV.AddFrame(NU);

  // Set the neutrino mass
  SetMassInvJigsaw MassJigsaw("MassJigsaw","m_{#nu} = 0");
  INV.AddJigsaw(MassJigsaw);

  // Set the neutrino rapidity
  SetRapidityInvJigsaw RapidityJigsaw("RapidityJigsaw","#eta_{#nu} = #eta_{#it{l}}");
  INV.AddJigsaw(RapidityJigsaw);
  RapidityJigsaw.AddVisibleFrame(L);

  if(LAB.InitializeAnalysis())
    g_Log << LogInfo << "...Successfully initialized analyses" << LogEnd;
  else
    g_Log << LogError << "...Failed initializing analyses" << LogEnd;
  
  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////

  TFile *fout = new TFile (output_name.c_str(), "RECREATE");
  TTree *tW = new TTree ("Wto1LMET", "Wto1LMET");
  tW->SetDirectory (fout);
  TTree *tR = new TTree ("Rndm1LMET", "Rndm1LMET");
  tR->SetDirectory (fout);
  
  TLorentzVector LEP;
  TVector3 MET;

  float L_pT, L_eta, L_phi, L_E;
  float MET_mag, MET_phi;

  tW->Branch ("L_pT",    &L_pT,    "L_pT/F");
  tW->Branch ("L_eta",   &L_eta,   "L_eta/F");
  tW->Branch ("L_phi",   &L_phi,   "L_phi/F");
  tW->Branch ("L_E",     &L_E,     "L_E/F");
  tW->Branch ("MET_mag", &MET_mag, "MET_mag/F");
  tW->Branch ("MET_phi", &MET_phi, "MET_phi/F");
  
  tR->Branch ("L_pT",    &L_pT,    "L_pT/F");
  tR->Branch ("L_eta",   &L_eta,   "L_eta/F");
  tR->Branch ("L_phi",   &L_phi,   "L_phi/F");
  tR->Branch ("L_E",     &L_E,     "L_E/F");
  tR->Branch ("MET_mag", &MET_mag, "MET_mag/F");
  tR->Branch ("MET_phi", &MET_phi, "MET_phi/F");

  for(int igen = 0; igen < Ngen; igen++){
    if(igen%((std::max(Ngen,10))/10) == 0)
      g_Log << LogInfo << "Generating event " << igen << " of " << Ngen << LogEnd;

    // generate event
    LAB_Gen.ClearEvent();                                // clear the gen tree

    pTWoMW = LAB_Gen.GetRandom();
    LAB_Gen.SetPToverM(pTWoMW);                          // give the W some Pt
    double PzW = mW*(2.*LAB_Gen.GetRandom()-1.);
    LAB_Gen.SetLongitudinalMomentum(PzW);                // give the W some Pz
     
    LAB_Gen.AnalyzeEvent();                              // generate a new event

    // analyze event
    LAB.ClearEvent();                               // clear the reco tree
      
    L.SetLabFrameFourVector(L_Gen.GetFourVector()); // Set lepton 4-vec
      
    MET = LAB_Gen.GetInvisibleMomentum();       // Get the MET from gen tree
    MET.SetZ(0.);
    INV.SetLabFrameThreeVector(MET);                     // Set the MET in reco tree
      
    LAB.AnalyzeEvent();                             // analyze the event

    LEP = L.GetFourVector ();

    // write the reconstructed branches
    L_pT = LEP.Pt();    L_eta = LEP.Eta();    L_phi = LEP.Phi();    L_E = LEP.E();
    MET_mag = MET.Mag();    MET_phi = MET.Phi();

    tW->Fill ();

    // randomize phi
    LEP.SetPhi (2. * TMath::Pi() * gRandom->Rndm());
    MET.SetPhi (2. * TMath::Pi() * gRandom->Rndm());

    // write the randomized branches
    L_pT = LEP.Pt();    L_eta = LEP.Eta();    L_phi = LEP.Phi();    L_E = LEP.E();
    MET_mag = MET.Mag();    MET_phi = MET.Phi();

    tR->Fill ();
  }

  tW->Write ();
  tR->Write ();
  fout->Close ();
}

# ifndef __CINT__ // main function for stand-alone compilation
int main(){
  gen_Wlnu();
  return 0;
}
#endif
