"""One-command offline acceptance. This entry cannot launch AnyLogic."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from business_graph_summary_v1.offline_acceptance import main

if __name__ == '__main__':
    main()
