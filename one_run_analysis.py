import uproot
import ROOT
import numpy as np
import requests
import re
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: monitor memory usage
def print_mem(label=""):
    mem = psutil.virtual_memory()
    print(f"{label} | Memory used: {mem.used/1e9:.2f} GB / {mem.total/1e9:.2f} GB")

def get_root_links(run_number):
    base_url = f"https://eospublic.cern.ch/eos/opendata/atlas/rucio/data16_13TeV/"
    container_url = f"{base_url}data16_13TeV.003{run_number}.physics_Main.merge.DAOD_PHYSLITE.fXXXX_pXXX/"

    response = requests.get(container_url)
    if response.status_code != 200:
        print(f"Failed to retrieve links for run {run_number}")
        return []

    # Extract ROOT file links from HTML
    root_links = re.findall(r'href="([^"]+\.root(?:\.\d+)?)"', response.text)
    full_links = [f"root://eospublic.cern.ch//eos/opendata/atlas/rucio/data16_13TeV/{run_number}/{link}" for link in root_links]

    return full_links

def analyze_run(run_links, run_number):
    preS_count = 0
    SR_count = 0

    for link in run_links:
        print_mem(f"Before processing {link}")
        chain = ROOT.TChain("CollectionTree")
        result = chain.Add(link)
        if result == 0:
            print(f"Could not add {link}")
            continue

        nEntries = chain.GetEntries()
        if nEntries == 0:
            continue

        # Set branches
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

        for i in range(min(nEntries, 10000)):
            if chain.LoadTree(i) < 0:
                continue
            chain.GetEntry(i)

            try:
                phi_met = np.arctan2(met_y[0], met_x[0])
            except Exception:
                continue

            if np.isnan(phi_met) or met[0] / 1000 < 250 or len(jets_pt) < 2:
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

        print_mem(f"After processing {link}")

    return preS_count, SR_count

def analyze_period(period_name, run_dict):
    total_preS = 0
    total_SR = 0

    with ThreadPoolExecutor(max_workers=1) as executor:  # only 1 to avoid memory overflow
        future_to_run = {
            executor.submit(analyze_run, links, run): run
            for run, links in run_dict.items()
        }

        for future in as_completed(future_to_run):
            run = future_to_run[future]
            try:
                preS, SR = future.result()
                print(f"Run {run}: preS = {preS}, SR = {SR}")
                total_preS += preS
                total_SR += SR
            except Exception as exc:
                print(f"Run {run} generated an exception: {exc}")

    print(f"{period_name}: total preS = {total_preS}, total SR = {total_SR}")

def main():
    # Only PeriodA run for now
    PeriodA_runs = {
        "297730": get_root_links("297730")
    }
    analyze_period("PeriodA", PeriodA_runs)

if __name__ == "__main__":
    main()
