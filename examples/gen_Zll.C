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

using namespace RestFrames;

void gen_Zll(const std::string& output_name = "Zll.root"){

  double mZ = 91.188; // GeV, PDG 2016
  double wZ = 2.495;

  // Number of events to generate
  int Ngen = 1000000;

  /////////////////////////////////////////////////////////////////////////////////////////
  g_Log << LogInfo << "Initializing generator frames and tree..." << LogEnd;
  /////////////////////////////////////////////////////////////////////////////////////////
  LabGenFrame       LAB_Gen("LAB_Gen","LAB");
  ResonanceGenFrame Z_Gen("Z_Gen","Z");
  VisibleGenFrame   Lp_Gen("Lp_Gen","#it{l}^{+}");
  VisibleGenFrame   Lm_Gen("Lm_Gen","#it{l}^{-}");

  //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//
  
  LAB_Gen.SetChildFrame(Z_Gen);
  Z_Gen.AddChildFrame(Lp_Gen);
  Z_Gen.AddChildFrame(Lm_Gen);
 
  if(LAB_Gen.InitializeTree())
    g_Log << LogInfo << "...Successfully initialized generator tree" << LogEnd;
  else
    g_Log << LogError << "...Failed initializing generator tree" << LogEnd;

  //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

  // Set Z pole mass and width
  Z_Gen.SetMass(mZ);                   Z_Gen.SetWidth(wZ);

  // set lepton pT and eta cuts
  Lp_Gen.SetPtCut(15.);                 Lp_Gen.SetEtaCut(2.5);
  Lm_Gen.SetPtCut(15.);                 Lm_Gen.SetEtaCut(2.5); 

  if(LAB_Gen.InitializeAnalysis())
    g_Log << LogInfo << "...Successfully initialized generator analysis" << std::endl << LogEnd;
  else
    g_Log << LogError << "...Failed initializing generator analysis" << LogEnd;
  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////////////////
  g_Log << LogInfo << "Initializing reconstruction frames and trees..." << LogEnd;
  /////////////////////////////////////////////////////////////////////////////////////////
  LabRecoFrame     LAB("LAB","LAB");
  DecayRecoFrame   Z("Z","Z");
  VisibleRecoFrame Lp("Lp","#it{l}^{+}");
  VisibleRecoFrame Lm("Lm","#it{l}^{-}");

  //-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//-//

  LAB.SetChildFrame(Z);
  Z.AddChildFrame(Lp);
  Z.AddChildFrame(Lm);

  if(LAB.InitializeTree())
    g_Log << LogInfo << "...Successfully initialized reconstruction trees" << LogEnd;
  else
    g_Log << LogError << "...Failed initializing reconstruction trees" << LogEnd;

  if(LAB.InitializeAnalysis())
    g_Log << LogInfo << "...Successfully initialized analyses" << LogEnd;
  else
    g_Log << LogError << "...Failed initializing analyses" << LogEnd;

  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  
  TFile *fout = new TFile (output_name.c_str(), "RECREATE");
  TTree *tZ = new TTree ("Zto2LOS", "Zto2LOS");
  tZ->SetDirectory (fout);
  TTree *tR = new TTree ("Rndm2LOS", "Rndm2LOS");
  tR->SetDirectory (fout);

  TLorentzVector LP;
  TLorentzVector LM;

  float LP_pT, LP_eta, LP_phi, LP_E;
  float LM_pT, LM_eta, LM_phi, LM_E;

  tZ->Branch ("LP_pT",  &LP_pT,  "LP_pT/F");
  tZ->Branch ("LP_eta", &LP_eta, "LP_eta/F");
  tZ->Branch ("LP_phi", &LP_phi, "LP_phi/F");
  tZ->Branch ("LP_E",   &LP_E,   "LP_E/F");
  tZ->Branch ("LM_pT",  &LM_pT,  "LM_pT/F");
  tZ->Branch ("LM_eta", &LM_eta, "LM_eta/F");
  tZ->Branch ("LM_phi", &LM_phi, "LM_phi/F");
  tZ->Branch ("LM_E",   &LM_E,   "LM_E/F");

  tR->Branch ("LP_pT",  &LP_pT,  "LP_pT/F");
  tR->Branch ("LP_eta", &LP_eta, "LP_eta/F");
  tR->Branch ("LP_phi", &LP_phi, "LP_phi/F");
  tR->Branch ("LP_E",   &LP_E,   "LP_E/F");
  tR->Branch ("LM_pT",  &LM_pT,  "LM_pT/F");
  tR->Branch ("LM_eta", &LM_eta, "LM_eta/F");
  tR->Branch ("LM_phi", &LM_phi, "LM_phi/F");
  tR->Branch ("LM_E",   &LM_E,   "LM_E/F");

  for(int igen = 0; igen < Ngen; igen++)
    {
      if(igen%((std::max(Ngen,10))/10) == 0)
	g_Log << LogInfo << "Generating event " << igen << " of " << Ngen << LogEnd;

      // generate event
      LAB_Gen.ClearEvent();                             // clear the gen tree

      double PTZ = mZ*gRandom->Rndm();
      LAB_Gen.SetTransverseMomentum(PTZ);               // give the Z some Pt
      double PzZ = mZ*(2.*gRandom->Rndm()-1.);
      LAB_Gen.SetLongitudinalMomentum(PzZ);             // give the Z some Pz

      LAB_Gen.AnalyzeEvent();                           // generate a new event

      // analyze event
      LAB.ClearEvent();                                 // clear the reco tree

      Lp.SetLabFrameFourVector(Lp_Gen.GetFourVector(), 1); // Set lepton 4-vec and charge
      Lm.SetLabFrameFourVector(Lm_Gen.GetFourVector(),-1); // Set lepton 4-vec and charge

      LAB.AnalyzeEvent();                               // analyze the event

      LP = Lp.GetFourVector ();
      LM = Lm.GetFourVector ();

      // write the reconstructed branches
      LP_pT = LP.Pt();    LP_eta = LP.Eta();    LP_phi = LP.Phi();    LP_E = LP.E();
      LM_pT = LM.Pt();    LM_eta = LM.Eta();    LM_phi = LM.Phi();    LM_E = LM.E();

      tZ->Fill ();

      // randomize phi
      LP.SetPhi (2. * TMath::Pi() * gRandom->Rndm());
      LM.SetPhi (2. * TMath::Pi() * gRandom->Rndm());

      // write the randomized branches
      LP_pT = LP.Pt();    LP_eta = LP.Eta();    LP_phi = LP.Phi();    LP_E = LP.E();
      LM_pT = LM.Pt();    LM_eta = LM.Eta();    LM_phi = LM.Phi();    LM_E = LM.E();

      tR->Fill ();
    }
    
  tZ->Write ();
  tR->Write ();
  fout->Close ();
}

# ifndef __CINT__ // main function for stand-alone compilation
int main(){
  gen_Zll();
  return 0;
}
#endif
