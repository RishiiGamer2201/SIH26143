import cdsapi

dataset = "reanalysis-era5-single-levels"

request = {
    "product_type": ["reanalysis"],

    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],

    "year": ["2023"],
    "month": ["01"],

    "day": [
        "01", "02", "03", "04",
        "05", "06", "07"
    ],

    "time": [
        "00:00", "01:00", "02:00", "03:00",
        "04:00", "05:00", "06:00", "07:00",
        "08:00", "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00", "15:00",
        "16:00", "17:00", "18:00", "19:00",
        "20:00", "21:00", "22:00", "23:00"
    ],

    # ERA5 area order:
    # [North, West, South, East]
    "area": [
        29.139636,
        -92.477933,
        25.106677,
        -88.379742
    ],

    "data_format": "netcdf",
    "download_format": "unarchived",
}

client = cdsapi.Client()

print("Submitting ERA5 request...")

client.retrieve(
    dataset,
    request,
    "era5_wind_10m.nc"
)

print("DOWNLOAD COMPLETE")
