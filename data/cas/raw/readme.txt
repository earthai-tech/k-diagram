README for Raw Data Samples

This directory provides small sample files to illustrate the expected structure for the raw datasets used in our paper. The full datasets are not included in this repository due to their large size.

[cite_start]For instance, the egms_point.csv sample contains just over 240 observations to demonstrate the required format.

To run the full data preparation pipeline, you must download the complete datasets from their official sources listed below. After downloading, please process and arrange the data to match the column structure of the sample files in this directory.

---
Data Sources

1.  Wind (gefcom_hourly.csv):
    GEFCom2014 Hourly Wind Power Forecasting
    https://iea-wind.org/task51/task51-information-portal/benchmarks 

2.  Hydrology (camels_timeseries.csv):
    CAMELS-US Daily Streamflow
    https://ral.ucar.edu/solutions/products/camels 

3.  Land Subsidence (egms_point.csv):
    European Ground Motion Service (EGMS)
    https://sdi.eea.europa.eu/catalogue/srv/api/records/7eb207d6-0a62-4280-b1ca-f4ad1d9f91c3

---
File Structure Guide

-   `gefcom_hourly.csv`: Expects columns for zone ID, hourly timestamp, and target power output.
-   `camels_timeseries.csv`: Expects columns for basin ID, daily timestamp, and streamflow in mm/day.
-   `egms_point.csv`: Expects columns for point ID, acquisition datetime, and line-of-sight displacement.

For more details on the data schema and processing, please refer to the main README in the parent directory (`data/cas/`).