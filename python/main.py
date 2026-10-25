"""
This just calls export_for_tableau.export_all(), which pulls everything
from Postgres, runs the significance tests and churn model, and writes
the Tableau-ready CSVs to ../tableau_exports/.
"""

from export_for_tableau import export_all

if __name__ == "__main__":
    export_all()
