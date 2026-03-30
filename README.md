Hotspot Finder - 3 Province Rebuild

Scope narrowed from whole-Canada to three provinces based on available data:
- Ontario
- British Columbia
- Quebec

Pathogens keep true source province labels.
Chemicals use a province-scope approximation from WWTP receiving-water context:
- Great Lakes -> Ontario
- Pacific Ocean -> British Columbia
- St. Lawrence River -> Quebec

Run:
cd hotspot_three_province_tool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd src
python3 -m streamlit run app.py
