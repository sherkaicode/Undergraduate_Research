import ROOT
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Suppress warnings, show only fatal errors
ROOT.gErrorIgnoreLevel = ROOT.kFatal

# Load metadata from JSON file
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
    """Analyze ROOT files for a given run using TChain."""
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
    print(f"Run {run_number} has total {nEntries} events")

    # Disable all branches and enable only required ones
    chain.SetBranchStatus("*", 0)
    chain.SetBranchStatus("MET_Core_AnalysisMETAuxDyn.sumet", 1)
    chain.SetBranchStatus("MET_Core_AnalysisMETAuxDyn.mpx", 1)
    chain.SetBranchStatus("MET_Core_AnalysisMETAuxDyn.mpy", 1)
    chain.SetBranchStatus("AnalysisJetsAuxDyn.pt", 1)
    chain.SetBranchStatus("AnalysisJetsAuxDyn.eta", 1)
    chain.SetBranchStatus("AnalysisJetsAuxDyn.phi", 1)

    # Define vectors for branch data
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

    max_events = min(nEntries, 100000)  # limit for safety
    for i in range(max_events):
        if chain.LoadTree(i) < 0:
            print(f"Failed to load entry {i}, skipping.")
            continue
        chain.GetEntry(i)

        try:
            phi_met = np.arctan2(met_y[0], met_x[0])
        except Exception:
            continue

        if np.isnan(phi_met):
            continue

        # Preselection criteria
        if met[0] / 1000 < 250:
            continue

        if len(jets_pt) < 2:
            continue

        has_valid_jet = False
        for phi in jets_phi:
            delta_phi = min(abs(phi - phi_met), 2 * np.pi - abs(phi - phi_met))
            if delta_phi < 2.0:
                has_valid_jet = True
                break
        if not has_valid_jet:
            continue

        preS_count += 1

        # Signal region criteria
        ht = np.sum(jets_pt) / 1000
        if met[0] / 1000 < 600 or ht < 600:
            continue

        SR_count += 1

    return preS_count, SR_count


def analyze_period(period_name, run_dict):
    print(f"\nAnalyzing {period_name}")
    total_preS, total_SR = 0, 0

    with ThreadPoolExecutor(max_workers=5) as executor:  # Limit to reduce server pressure
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

def main():
    periods = {
        "PeriodA": PeriodA_runs,
        "PeriodB": PeriodB_runs,
        "PeriodC": PeriodC_runs,
        "PeriodD": PeriodD_runs,
        "PeriodE": PeriodE_runs,
        "PeriodF": PeriodF_runs,
        "PeriodG": PeriodG_runs,
        "PeriodI": PeriodI_runs,
        "PeriodK": PeriodK_runs
    }

    for period_name, run_dict in periods.items():
        analyze_period(period_name, run_dict)

# Load all run links up front to avoid repeated I/O
def prepare_runs():
    global PeriodA_runs, PeriodB_runs, PeriodC_runs, PeriodD_runs, PeriodE_runs
    global PeriodF_runs, PeriodG_runs, PeriodI_runs, PeriodK_runs

    def make_run_dict(runs):
        return {run: get_root_links(run) for run in runs}

    PeriodA_runs = make_run_dict([
        "297730", "298595", "298609", "298633", "298687", "298690", "298771", "298773",
        "298862", "298967", "299055", "299144", "299147", "299184", "299241", "299243",
        "299278", "299288", "299315", "299340", "299343", "299390", "299584", "300279", "300287"
    ])

    PeriodB_runs = make_run_dict([
        "300908", "300863", "300800", "300784", "300687", "300655", "300600",
        "300571", "300540", "300487", "300418", "300415", "300345"
    ])

    PeriodC_runs = make_run_dict([
        "302393", "302391", "302380", "302347", "302300", "302269", "302265",
        "302137", "302053", "301973", "301932", "301918", "301915", "301912"
    ])

    PeriodD_runs = make_run_dict([
        "303560", "303499", "303421", "303338", "303304", "303291", "303266", "303264",
        "303208", "303201", "303079", "303059", "303007", "302956", "302925", "302919",
        "302872", "302831", "302829", "302737"
    ])

    PeriodE_runs = make_run_dict([
        "303892", "303846", "303832", "303819", "303817", "303811", "303726", "303638"
    ])

    PeriodF_runs = make_run_dict([
        "304494", "304431", "304409", "304337", "304308", "304243", "304211",
        "304198", "304178", "304128", "304008", "304006", "303943"
    ])

    PeriodG_runs = make_run_dict([
        "306714", "306657", "306655", "306556", "306451", "306448", "306442", "306419",
        "306384", "306310", "306278", "306269", "305920", "305811", "305777", "305735",
        "305727", "305723", "305674", "305671", "305618", "305571", "305543", "305380", "305293"
    ])

    PeriodI_runs = make_run_dict([
        "308084", "308047", "307935", "307861", "307732", "307716", "307710", "307656",
        "307619", "307601", "307569", "307539", "307514", "307454", "307394", "307358",
        "307354", "307306", "307259", "307195", "307126", "307124"
    ])

    PeriodK_runs = make_run_dict([
        "309759", "309674", "309640", "309516", "309440", "309390", "309375"
    ])

if __name__ == "__main__":
    prepare_runs()
    main()

