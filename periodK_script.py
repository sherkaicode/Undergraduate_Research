import ROOT
import json
import numpy as np
import os
import gc

ROOT.gErrorIgnoreLevel = ROOT.kFatal  # suppress non-fatal warnings

# Load metadata
with open("ATLAS.json", "r") as f:
    metadata = json.load(f)

def get_root_links(run):
    links = []
    for meta_run in metadata["metadata"]["_file_indices"]:
        if meta_run["key"].split("_")[3][2:] == run:
            for root_file in meta_run["files"]:
                links.append(root_file["uri"])
    return links

def analyze_file(file_path):
    """Analyze a single ROOT file and return preselection and SR counts."""
    chain = ROOT.TChain("CollectionTree")
    added = chain.Add(file_path)
    if not added:
        print(f"Failed to add file: {file_path}")
        return 0, 0

    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("MET_Core_AnalysisMETAuxDyn.sumet", 1)
    chain.SetBranchStatus("MET_Core_AnalysisMETAuxDyn.mpx", 1)
    chain.SetBranchStatus("MET_Core_AnalysisMETAuxDyn.mpy", 1)
    chain.SetBranchStatus("AnalysisJetsAuxDyn.pt", 1)
    chain.SetBranchStatus("AnalysisJetsAuxDyn.eta", 1)
    chain.SetBranchStatus("AnalysisJetsAuxDyn.phi", 1)

    jets_pt = ROOT.vector('float')()
    jets_eta = ROOT.vector('float')()
    jets_phi = ROOT.vector('float')()
    met = ROOT.vector('float')()
    met_x = ROOT.vector('float')()
    met_y = ROOT.vector('float')()

    chain.SetBranchAddress("AnalysisJetsAuxDyn.pt", jets_pt)
    chain.SetBranchAddress("AnalysisJetsAuxDyn.eta", jets_eta)
    chain.SetBranchAddress("AnalysisJetsAuxDyn.phi", jets_phi)
    chain.SetBranchAddress("MET_Core_AnalysisMETAuxDyn.sumet", met)
    chain.SetBranchAddress("MET_Core_AnalysisMETAuxDyn.mpx", met_x)
    chain.SetBranchAddress("MET_Core_AnalysisMETAuxDyn.mpy", met_y)

    preS = 0
    SR = 0
    max_events = chain.GetEntries()
    for i in range(max_events):
        if chain.LoadTree(i) < 0:
            continue
        chain.GetEntry(i)

        try:
            phi_met = np.arctan2(met_y[0], met_x[0])
            if np.isnan(phi_met):
                continue
        except:
            continue

        if met[0] / 1000 < 250 or len(jets_pt) < 2:
            continue

        if not any(min(abs(phi - phi_met), 2 * np.pi - abs(phi - phi_met)) < 2.0 for phi in jets_phi):
            continue

        preS += 1

        ht = sum(jets_pt) / 1000
        if met[0] / 1000 >= 600 and ht >= 600:
            SR += 1

    # Memory cleanup
    chain.Reset()
    del chain, jets_pt, jets_eta, jets_phi, met, met_x, met_y
    gc.collect()

    return preS, SR

def analyze_run(run_number, file_list):
    """Analyze all files in a run, sequentially."""
    print(f"\nAnalyzing run {run_number}")
    run_preS = 0
    run_SR = 0

    for file_path in file_list:
        print(f"  File: {file_path}")
        try:
            preS, SR = analyze_file(file_path)
            run_preS += preS
            run_SR += SR
        except Exception as e:
            print(f"    [ERROR] Failed to process {file_path}: {e}")

    print(f"Run {run_number} totals: Preselection = {run_preS}, Signal Region = {run_SR}")
    return run_preS, run_SR

def main():
    runs_periodK = [
        "309375",
        "309390",
        "309440",
        "309516",
        "309640",
        "309674",
        "309759"
    ]


    total_preS = 0
    total_SR = 0

    for run in runs_periodK:
        links = get_root_links(run)
        preS, SR = analyze_run(run, links)
        total_preS += preS
        total_SR += SR

    print("\n=== PeriodK Summary ===")
    print(f"Total Preselection: {total_preS}")
    print(f"Total Signal Region: {total_SR}")

if __name__ == "__main__":
    main()
