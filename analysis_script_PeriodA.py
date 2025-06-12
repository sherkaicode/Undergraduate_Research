import ROOT
import json
import numpy as np
import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT.gErrorIgnoreLevel = ROOT.kFatal  # Show only fatal errors

# Load metadata
with open("ATLAS.json", "r") as f:
    metadata = json.load(f)

def get_root_links(run):
    """Retrieve ROOT file links for a given run number."""
    links = []
    for meta_run in metadata["metadata"]["_file_indices"]:
        if meta_run["key"].split("_")[3][2:] == run:
            for root_file in meta_run["files"]:
                links.append(root_file["uri"])
    return links

def analyze_run(run_links, run_number):
    """Analyze ROOT files for a given run using TChain (1 TChain per run)."""
    preS_count = 0
    SR_count = 0

    chain = ROOT.TChain("CollectionTree")
    for link in run_links:
        result = chain.Add(link)
        if result == 0:
            print(f"Warning: Could not add {link}")
        else:
            print(f"Added file: {link}")

    nEntries = chain.GetEntries()
    print(f"Run {run_number} has {nEntries} events")

    # Enable only needed branches
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("MET_Core_AnalysisMETAuxDyn.sumet", 1)
    chain.SetBranchStatus("MET_Core_AnalysisMETAuxDyn.mpx", 1)
    chain.SetBranchStatus("MET_Core_AnalysisMETAuxDyn.mpy", 1)
    chain.SetBranchStatus("AnalysisJetsAuxDyn.pt", 1)
    chain.SetBranchStatus("AnalysisJetsAuxDyn.eta", 1)
    chain.SetBranchStatus("AnalysisJetsAuxDyn.phi", 1)

    # Create containers
    jets_pt = ROOT.vector('float')()
    jets_eta = ROOT.vector('float')()
    jets_phi = ROOT.vector('float')()
    met = ROOT.vector('float')()
    met_x = ROOT.vector('float')()
    met_y = ROOT.vector('float')()

    # Set branch addresses
    chain.SetBranchAddress("AnalysisJetsAuxDyn.pt", jets_pt)
    chain.SetBranchAddress("AnalysisJetsAuxDyn.eta", jets_eta)
    chain.SetBranchAddress("AnalysisJetsAuxDyn.phi", jets_phi)
    chain.SetBranchAddress("MET_Core_AnalysisMETAuxDyn.sumet", met)
    chain.SetBranchAddress("MET_Core_AnalysisMETAuxDyn.mpx", met_x)
    chain.SetBranchAddress("MET_Core_AnalysisMETAuxDyn.mpy", met_y)

    max_events = nEntries # cap to reduce memory usage
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

        has_valid_jet = any(
            min(abs(phi - phi_met), 2 * np.pi - abs(phi - phi_met)) < 2.0
            for phi in jets_phi
        )
        if not has_valid_jet:
            continue

        preS_count += 1

        ht = np.sum(jets_pt) / 1000
        if met[0] / 1000 >= 600 and ht >= 600:
            SR_count += 1

    # Memory cleanup
    chain.Reset()
    del chain, jets_pt, jets_eta, jets_phi, met, met_x, met_y
    gc.collect()

    return preS_count, SR_count

def analyze_period(period_name, run_dict):
    print(f"\nAnalyzing {period_name}")
    total_preS, total_SR = 0, 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(analyze_run, links, run_number): run_number
            for run_number, links in run_dict.items()
        }

        for future in as_completed(futures):
            run_number = futures[future]
            try:
                preS, SR = future.result()
                print(f"Run {run_number}: Preselection = {preS}, Signal Region = {SR}")
                total_preS += preS
                total_SR += SR
            except Exception as e:
                print(f"[ERROR] Run {run_number} failed: {e}")

    print(f"{period_name} Totals: Preselection = {total_preS}, Signal Region = {total_SR}")

def prepare_runs():
    # Only PeriodA
    return {
        "PeriodA": {
            run: get_root_links(run)
            for run in [
                "297730", "298595", "298609", "298633", "298687", "298690", "298771", "298773",
                "298862", "298967", "299055", "299144", "299147", "299184", "299241", "299243",
                "299278", "299288", "299315", "299340", "299343", "299390", "299584", "300279", "300287"
            ]
        }
    }

if __name__ == "__main__":
    periods = prepare_runs()
    for period_name, run_dict in periods.items():
        analyze_period(period_name, run_dict)

