# oilwatch run e008_v3_scene: incident_001

T0 2023-01-03 00:01:42+00:00 · scene S1A_IW_GRDH_1SDV_20230103T000142_20230103T000207_046612_059620_257F

> Rankings are evidence of spatio-temporal consistency between AIS tracks and modelled oil drift, never proof of culpability. source_type is 'unknown' for every AIS track until vessel metadata and an offshore-infrastructure database are joined; motion_state is not evidence of type.

## Slick events (E008 policy v1)

| potential_rank | cluster_id | component_ids | area_m2 | confidence | wind_speed_m_s | wind_class | slick_potential | eligible | selected | ineligible_reason |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 3,4 | 9.75e+06 | 0.981 | 7.81 | valid | 0.981 | True | True |  |
| 2 | 7 | 12,13 | 1.65e+06 | 0.967 | 8.02 | valid | 0.967 | True | True |  |
| 3 | 3 | 5 | 4.93e+06 | 0.966 | 7.85 | valid | 0.966 | True | True |  |
| 4 | 15 | 25,26,27 | 1.38e+06 | 0.964 | 8.71 | valid | 0.964 | True | True |  |
| 5 | 1 | 1,2,6 | 2.29e+06 | 0.957 | 7.76 | valid | 0.957 | True | True |  |
| 6 | 12 | 19,20,21,22 | 7.35e+05 | 0.927 | 8.4 | valid | 0.926 | True | True |  |
| 7 | 16 | 28,29 | 4.04e+05 | 0.904 | 8.02 | valid | 0.888 | True | True |  |
| 8 | 22 | 45,46,47,48,49 | 7.38e+05 | 0.886 | 8.49 | valid | 0.886 | True | True |  |
| 9 | 21 | 44 | 4.32e+05 | 0.896 | 8.45 | valid | 0.884 | True | True |  |
| 10 | 17 | 30,31,32,34,37,38,39 | 1.34e+06 | 0.882 | 8.64 | valid | 0.882 | True | True |  |
| 11 | 19 | 40,41,42 | 9.2e+05 | 0.863 | 8.69 | valid | 0.863 | True | True |  |
| 12 | 18 | 33,35,36 | 4.34e+05 | 0.873 | 8.69 | valid | 0.862 | True | True |  |
| 13 | 10 | 16,17 | 2.09e+05 | 0.924 | 8.63 | valid | 0.81 | True | False |  |
| 14 | 5 | 8,9,10 | 2.97e+05 | 0.838 | 7.79 | valid | 0.795 | True | False |  |
| 15 | 13 | 23 | 1.23e+05 | 0.885 | 8.69 | valid | 0.625 | True | False |  |
| 16 | 8 | 14 | 6.99e+04 | 0.872 | 8.09 | valid | 0.439 | True | False |  |
| 17 | 20 | 43 | 6.56e+04 | 0.901 | 8.44 | valid | 0.434 | True | False |  |
| 18 | 4 | 7 | 4.5e+04 | 0.824 | 7.78 | valid | 0.298 | True | False |  |
| 19 | 14 | 24 | 2.37e+04 | 0.727 | 8.37 | valid | 0.153 | True | False |  |
| 20 | 11 | 18 | 2.75e+04 | 0.619 | 8.4 | valid | 0.149 | True | False |  |
| 21 | 9 | 15 | 3.17e+03 | 0.626 | 8.05 | valid | 0.0195 | False | False | area 3168 m2 < 23000 (below detector's reliable size) |
| 22 | 6 | 11 | 2.81e+03 | 0.623 | 7.81 | valid | 0.0173 | False | False | area 2811 m2 < 23000 (below detector's reliable size) |

## Event 2: 9.75 km², 8.6 km long, wind 7.8 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 34}. Age estimate 28.0–29.0 h (forward best window of top supported suspect (OVERSEAS CHINOOK)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | lead only | OVERSEAS CHINOOK | 366495000 | 0.31 | 0.00424 | 0.504 | 5 | 0.333 | 29 | 0.133 | 0.333 | 0.0507 | False | 28 | 29 | forward+backward | 255 | 0 | moving | 3 | False | False | False | False | nan | nan | False |
| 2 | consistent | lead only | FAIRCHEM BLADE | 371085000 | 0.123 | 0.00434 | 0.644 | 5 | 0 | 30 | 0.0206 | 0.111 | 0.00104 | False | 29 | 30 | forward+backward | 200 | 0 | moving | 1 | False | False | False | False | nan | nan | False |
| 3 | consistent | lead only | DUBAI GLAMOUR | 538003526 | 0 | 0.00407 | 0.284 | 4 | 0.222 | 21 | 0 | 0 | 0 | False | 0 | 1 | backward only | 113 | 0 | moving | 0 | False | False | False | False | nan | nan | False |
| 4 | consistent | lead only | NOBLE FAYE KOZACK | 636015856 | 0 | 0.00915 | 0.581 | 3 | 0.667 | 48 | 0 | 0 | 0 | False | 0 | 1 | backward only | 37.8 | 0 | stationary | 0 | False | False | False | False | ais_stationary | 36.9 | False |
| 5 | weak | lead only | URSA TLP | 367063220 | 0.866 | 0.0105 | 0.124 | 1 | 0.778 | 5 | 0.422 | 1 | 0.0938 | False | 4 | 6 | forward only | 0.455 | 1 | stationary | 5 | False | False | False | False | nan | nan | False |
| 6 | weak | lead only | YUNNAN | 477855700 | 0.581 | 0.00354 | 0.8 | 8 | 0 | 35 | 0.564 | 0.889 | 0.203 | False | 34 | 35 | forward+backward | 90.8 | 0.641 | moving | 1 | False | True | True | False | nan | nan | False |
| 7 | weak | lead only | PEACEFUL LADY | 367094770 | 0.452 | 0.00411 | 0.362 | 6 | 0.333 | 47 | 0.294 | 0.333 | 0.0765 | False | 41 | 44 | forward+backward | 23.2 | 0.692 | moving | 3 | False | True | False | False | nan | nan | False |
| 8 | weak | lead only | SCARLET CARDINAL | 351684000 | 0.376 | 0.00369 | 0.769 | 7 | 0 | 35 | 0.226 | 0.704 | 0.0688 | False | 34 | 35 | forward+backward | 73.4 | 0 | mixed | 1 | False | False | False | False | nan | nan | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 1.12 | 11.2 | W=1136 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 12.3 | 12.3 | sigma=W/4=284 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:OVERSEAS CHINOOK | 28 | 29 | F=0.13, FSS1km=0.05 |
| backward_peak_age:OVERSEAS CHINOOK | 29 | 29 | median over members |
| forward_best_window:FAIRCHEM BLADE | 29 | 30 | F=0.02, FSS1km=0.00 |
| backward_peak_age:FAIRCHEM BLADE | 30 | 30 | median over members |
| backward_peak_age:DUBAI GLAMOUR | 21 | 21 | median over members |
| sar_darkness | 2 | 24 | damping 5.6 dB, edge sharpness 0.83 -> intermediate (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 7: 1.65 km², 2.0 km long, wind 8.0 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 28}. Age estimate 15.0–16.0 h (forward best window of top supported suspect (NOBLE FAYE KOZACK)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | lead only | NOBLE FAYE KOZACK | 636015856 | 0.975 | 0.117 | 0.8 | 1 | 1 | 16 | 0.435 | 0.444 | 0.263 | False | 15 | 16 | forward+backward | 18.2 | 0 | stationary | 0 | False | False | False | False | ais_stationary | 17.2 | False |
| 2 | consistent | lead only | GEMI | 338281000 | 0.365 | 0.0156 | 0.415 | 2 | 1 | 17 | 0.457 | 0.333 | 0.238 | False | 12 | 13 | forward+backward | 18.1 | 5.86e-117 | moving | 6 | False | True | False | False | nan | nan | False |
| 3 | consistent | lead only | BULK CONCORD | 636018377 | 0.0535 | 0.00578 | 0.178 | 3 | 0.667 | 11 | 0.0266 | 0.111 | 0.0164 | False | 11 | 12 | forward+backward | 162 | 0.0824 | moving | 0 | False | False | False | False | nan | nan | False |
| 4 | consistent | lead only | HARVEY CHAMPION | 366840000 | 0 | 0.00238 | 0.373 | 5 | 0 | 48 | 0 | 0 | 0 | False | 0 | 1 | backward only | 170 | 0 | moving | 2 | False | False | False | False | nan | nan | False |
| 5 | consistent | lead only | FANTASY ISLAND | 367789310 | 0 | 0.00293 | 0.367 | 4 | 0.333 | 48 | 0 | 0 | 0 | False | 0 | 1 | backward only | 37.5 | 0.0486 | mixed | 3 | False | False | False | False | nan | nan | False |
| 6 | weak | lead only | STOLT SNELAND | 319627000 | 0.0466 | 0.00186 | 0.428 | 6 | 0 | 37 | 0.0627 | 0.222 | 0.0819 | False | 36 | 37 | forward+backward | 231 | 0 | moving | 0 | False | False | False | False | nan | nan | False |
| 7 | not supported | lead only | MSC NEW HAVEN | 255803670 | 0 | 0.000805 | 0.298 | 9 | 0 | 41 | 0 | 0 | 0 | False | 0 | 1 | backward only | 139 | 0.0912 | moving | 0 | False | False | False | False | nan | nan | False |
| 8 | not supported | lead only | SUNNY VICTORY | 257084660 | 0 | 1.64e-05 | 0.00367 | 18 | 0 | 37 | 0 | 0 | 0 | False | 0 | 1 | neither | 74.8 | 0.612 | moving | 0 | True | False | False | False | nan | nan | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.587 | 5.87 | W=824 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 9.36 | 9.36 | sigma=W/4=206 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:NOBLE FAYE KOZACK | 15 | 16 | F=0.43, FSS1km=0.26 |
| backward_peak_age:NOBLE FAYE KOZACK | 16 | 16 | median over members |
| forward_best_window:GEMI | 12 | 13 | F=0.46, FSS1km=0.24 |
| backward_peak_age:GEMI | 17 | 17 | median over members |
| forward_best_window:BULK CONCORD | 11 | 12 | F=0.03, FSS1km=0.02 |
| backward_peak_age:BULK CONCORD | 11 | 11 | median over members |
| sar_darkness | 2 | 24 | damping 4.2 dB, edge sharpness 0.87 -> intermediate (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 3: 4.93 km², 11.5 km long, wind 7.8 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 38}. Age estimate 18.0–19.0 h (forward best window of top supported suspect (PRELUDE)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | lead only | PRELUDE | 374788000 | 0.606 | 0.00792 | 0.474 | 1 | 0.667 | 19 | 0.145 | 0.481 | 0.08 | False | 18 | 19 | forward+backward | 286 | 0.223 | moving | 1 | False | False | False | False | nan | nan | False |
| 2 | consistent | lead only | NEFELI | 249390000 | 0.598 | 0.00285 | 0.694 | 4 | 0.259 | 38 | 0.394 | 0.963 | 0.22 | False | 37 | 38 | forward+backward | 122 | 0 | moving | 1 | False | False | False | False | nan | nan | False |
| 3 | consistent | lead only | OVERSEAS CHINOOK | 366495000 | 0.36 | 0.00319 | 0.4 | 3 | 0.667 | 29 | 0.128 | 0.444 | 0.0665 | False | 28 | 29 | forward+backward | 250 | 0 | moving | 3 | False | False | False | False | nan | nan | False |
| 4 | weak | lead only | MARS TLP | 367101870 | 0.508 | 0.00796 | 0.047 | 2 | 1 | 2 | 0.102 | 0.37 | 0.00749 | False | 2 | 3 | forward only | 3.73 | 0 | stationary | 0 | False | False | False | False | ais_stationary | 1.95 | False |
| 5 | weak | lead only | RONG HUA WAN | 413527660 | 0.382 | 0.00175 | 0.707 | 9 | 0 | 43 | 0.262 | 0.667 | 0.124 | False | 42 | 43 | forward+backward | 200 | 0 | moving | 0 | False | False | False | False | nan | nan | False |
| 6 | weak | lead only | JIPRO ISIS | 370069000 | 0.352 | 0.00208 | 0.735 | 7 | 0 | 41 | 0.187 | 0.667 | 0.0794 | False | 41 | 42 | forward+backward | 122 | 0 | moving | 0 | False | False | False | False | nan | nan | False |
| 7 | weak | lead only | SCARLET CARDINAL | 351684000 | 0.281 | 0.00167 | 0.311 | 8 | 0 | 35 | 0.149 | 0.444 | 0.0706 | False | 34 | 35 | forward+backward | 73.9 | 0 | mixed | 1 | False | False | False | False | nan | nan | False |
| 8 | weak | lead only | CARNIVAL GLORY | 357659000 | 0.188 | 0.00108 | 0.116 | 12 | 0 | 26 | 0.103 | 0.37 | 0.0328 | False | 25 | 26 | forward only | 185 | 0.333 | moving | 0 | False | False | False | False | nan | nan | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.158 | 1.58 | W=430 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 5.37 | 5.37 | sigma=W/4=107 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:PRELUDE | 18 | 19 | F=0.15, FSS1km=0.08 |
| backward_peak_age:PRELUDE | 19 | 19 | median over members |
| forward_best_window:NEFELI | 37 | 38 | F=0.39, FSS1km=0.22 |
| backward_peak_age:NEFELI | 38 | 38 | median over members |
| forward_best_window:OVERSEAS CHINOOK | 28 | 29 | F=0.13, FSS1km=0.07 |
| backward_peak_age:OVERSEAS CHINOOK | 29 | 29 | median over members |
| sar_darkness | 2 | 24 | damping 4.4 dB, edge sharpness 0.86 -> intermediate (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 15: 1.38 km², 2.2 km long, wind 8.7 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 40}. Age estimate 6.0–7.0 h (forward best window of top supported suspect (MILLIE)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | high-confidence flag (calibrated) | MILLIE | 368179260 | 1 | 0.149 | 0.766 | 1 | 1 | 8 | 0.722 | 0.667 | 0.423 | False | 6 | 7 | forward+backward | 3.98 | 0 | loitering | 5 | False | False | False | False | nan | nan | False |
| 2 | consistent | supported | HOLSTEIN | 366048300 | 0.735 | 0.0805 | 0.756 | 3 | 1 | 8 | 0.722 | 0.667 | 0.421 | False | 6 | 7 | forward+backward | 3.84 | 1.1e-210 | stationary | 5 | False | False | False | False | ais_stationary | 3.8 | False |
| 3 | consistent | lead only | HARVEY EXPLORER | 369286000 | 0.619 | 0.064 | 0.695 | 3 | 0.667 | 9 | 0.646 | 0.667 | 0.297 | False | 8 | 9 | forward+backward | 143 | 0.0565 | mixed | 3 | False | False | False | False | nan | nan | False |
| 4 | consistent | lead only | ARGOS PLATFORM | 368166560 | 0.232 | 0.0162 | 0.587 | 4 | 0.333 | 42 | 0.357 | 0.333 | 0.419 | False | 38 | 39 | forward+backward | 27.3 | 0 | stationary | 6 | False | False | False | False | ais_stationary | 27.3 | False |
| 5 | weak | lead only | SEVEN OCEANS | 235053116 | 0.158 | 0.00742 | 0.566 | 11 | 0 | 41 | 0.362 | 0.296 | 0.206 | False | 39 | 40 | forward+backward | 27.3 | 0 | mixed | 3 | False | True | False | False | ais_stationary | 27.2 | False |
| 6 | weak | lead only | PELICAN ISLAND | 367684260 | 0.153 | 0.00882 | 0.666 | 9 | 0 | 45 | 0.286 | 0.333 | 0.224 | False | 46 | 47 | forward+backward | 50.8 | 0.143 | mixed | 4 | False | False | False | True | nan | nan | False |
| 7 | weak | lead only | BP MAD DOG PLATFORM | 369494392 | 0.14 | 0.00842 | 0.648 | 10 | 0 | 45 | 0.251 | 0.333 | 0.245 | False | 45 | 46 | forward+backward | 33.7 | 0 | stationary | 5 | False | False | False | False | nan | nan | False |
| 8 | weak | lead only | TUCKER CANDIES | 338002000 | 0.131 | 0.0078 | 0.436 | 7 | 0 | 44 | 0.237 | 0.333 | 0.177 | False | 42 | 43 | forward+backward | 201 | 0 | moving | 1 | False | False | False | False | nan | nan | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.334 | 3.34 | W=622 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 7.36 | 7.36 | sigma=W/4=156 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:MILLIE | 6 | 7 | F=0.72, FSS1km=0.42 |
| backward_peak_age:MILLIE | 8 | 8 | median over members |
| forward_best_window:HOLSTEIN | 6 | 7 | F=0.72, FSS1km=0.42 |
| backward_peak_age:HOLSTEIN | 8 | 8 | median over members |
| forward_best_window:HARVEY EXPLORER | 8 | 9 | F=0.65, FSS1km=0.30 |
| backward_peak_age:HARVEY EXPLORER | 9 | 9 | median over members |
| sar_darkness | 2 | 24 | damping 3.1 dB, edge sharpness 0.69 -> intermediate (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 1: 2.29 km², 3.3 km long, wind 7.8 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 36}. Age estimate 4.0–6.0 h (forward best window of top supported suspect (URSA TLP)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | lead only | URSA TLP | 367063220 | 1 | 0.0164 | 0.201 | 1 | 1 | 6 | 0.321 | 0.778 | 0.083 | False | 4 | 6 | forward+backward | 6.61 | 0.978 | stationary | 5 | False | False | False | False | nan | nan | False |
| 2 | consistent | lead only | NOBLE FAYE KOZACK | 636015856 | 0.644 | 0.0161 | 0.667 | 2 | 1 | 48 | 0.136 | 0.296 | 0.0687 | False | 47 | 48 | forward+backward | 38.9 | 0 | stationary | 0 | False | False | False | True | ais_stationary | 37 | False |
| 3 | consistent | lead only | PEACEFUL LADY | 367094770 | 0.619 | 0.00805 | 0.624 | 4 | 0.296 | 47 | 0.251 | 0.296 | 0.0798 | False | 46 | 47 | forward+backward | 28.7 | 0.785 | moving | 3 | False | False | False | True | nan | nan | False |
| 4 | consistent | lead only | DUBAI GLAMOUR | 538003526 | 0.498 | 0.00713 | 0.498 | 5 | 0.333 | 21 | 0.183 | 0.444 | 0.0791 | False | 20 | 21 | forward+backward | 113 | 0 | moving | 0 | False | False | False | False | nan | nan | False |
| 5 | weak | lead only | AQUAVITA SUN | 548978000 | 0.512 | 0.00649 | 0.755 | 6 | 0 | 27 | 0.213 | 0.667 | 0.0864 | False | 26 | 27 | forward+backward | 78.5 | 0 | mixed | 1 | False | False | False | False | nan | nan | False |
| 6 | weak | lead only | IKAN LEBAN | 538005604 | 0.426 | 0.0031 | 0.722 | 8 | 0 | 35 | 0.308 | 0.926 | 0.344 | False | 35 | 36 | forward+backward | 138 | 0.216 | moving | 0 | False | False | False | False | nan | nan | False |
| 7 | weak | lead only | FAIRCHEM BLADE | 371085000 | 0.361 | 0.0031 | 0.435 | 9 | 0 | 30 | 0.222 | 0.815 | 0.202 | False | 29 | 30 | forward+backward | 202 | 0 | moving | 1 | False | False | False | False | nan | nan | False |
| 8 | weak | lead only | MTM LONDON | 566199000 | 0.348 | 0.00474 | 0.261 | 6 | 0 | 18 | 0.135 | 0.407 | 0.128 | False | 17 | 18 | forward+backward | 211 | 0.0624 | moving | 2 | False | False | False | False | nan | nan | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.428 | 4.28 | W=704 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 8.18 | 8.18 | sigma=W/4=176 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:URSA TLP | 4 | 6 | F=0.32, FSS1km=0.08 |
| backward_peak_age:URSA TLP | 6 | 6 | median over members |
| forward_best_window:NOBLE FAYE KOZACK | 47 | 48 | F=0.14, FSS1km=0.07 |
| backward_peak_age:NOBLE FAYE KOZACK | 48 | 48 | median over members |
| forward_best_window:PEACEFUL LADY | 46 | 47 | F=0.25, FSS1km=0.08 |
| backward_peak_age:PEACEFUL LADY | 47 | 47 | median over members |
| sar_darkness | 2 | 24 | damping 3.3 dB, edge sharpness 0.78 -> intermediate (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 12: 0.73 km², 3.2 km long, wind 8.4 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 24}. Age estimate 7.0–9.0 h (forward best window of top supported suspect (MARCO POLO TLP)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | lead only | MARCO POLO TLP | 366908680 | 1 | 0.0943 | 0.642 | 2 | 1 | 9 | 0.606 | 0.852 | 0.162 | False | 7 | 9 | forward+backward | 4.09 | 0 | mixed | 7 | False | True | False | False | ais_moving | 3.66 | False |
| 2 | consistent | lead only | TLP SHENZI | 366050100 | 0.73 | 0.074 | 0.624 | 2 | 1 | 16 | 0.411 | 0.407 | 0.252 | False | 15 | 17 | forward+backward | 11.7 | 0 | stationary | 4 | False | True | False | False | ais_stationary | 11.1 | False |
| 3 | consistent | lead only | ISLAND VENTURE | 577358000 | 0.482 | 0.0469 | 0.679 | 3 | 0.667 | 43 | 0.283 | 0.333 | 0.244 | False | 43 | 44 | forward+backward | 24.9 | 0 | stationary | 2 | False | False | False | False | ais_stationary | 24.4 | False |
| 4 | consistent | lead only | WEST AURIGA | 538009623 | 0.379 | 0.0278 | 0.796 | 4 | 0 | 42 | 0.296 | 0.333 | 0.218 | False | 42 | 43 | forward+backward | 24.4 | 0 | stationary | 6 | False | False | False | False | ais_silent_at_t0 | 23.7 | False |
| 5 | consistent | lead only | DEEPWATER INVICTUS | 538004610 | 0.332 | 0.0228 | 0.156 | 5 | 0.333 | 29 | 0.277 | 0.259 | 0.0699 | False | 14 | 15 | forward+backward | 7.58 | 0 | stationary | 5 | False | True | False | False | ais_stationary | 7.24 | False |
| 6 | weak | lead only | FAST LEOPARD | 367653160 | 0.248 | 0.0118 | 0.87 | 6 | 0 | 42 | 0.297 | 0.333 | 0.274 | False | 41 | 42 | forward+backward | 194 | 0 | moving | 5 | False | False | False | False | nan | nan | False |
| 7 | weak | lead only | BP ATLANTIS PLATFORM | 369494390 | 0.189 | 0.014 | 0.78 | 7 | 0 | 44 | 0.146 | 0.148 | 0.0404 | False | 38 | 39 | forward+backward | 27.5 | 0 | stationary | 5 | False | True | False | False | ais_stationary | 27 | False |
| 8 | weak | lead only | PELICAN ISLAND | 367684260 | 0.0616 | 0.0046 | 0.711 | 8 | 0 | 37 | 0.0472 | 0.0741 | 0.0114 | False | 36 | 37 | forward+backward | 17 | 0.311 | mixed | 4 | False | True | False | False | nan | nan | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.0424 | 0.424 | W=227 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 3.1 | 3.1 | sigma=W/4=57 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:MARCO POLO TLP | 7 | 9 | F=0.61, FSS1km=0.16 |
| backward_peak_age:MARCO POLO TLP | 9 | 9 | median over members |
| forward_best_window:TLP SHENZI | 15 | 17 | F=0.41, FSS1km=0.25 |
| backward_peak_age:TLP SHENZI | 16 | 16 | median over members |
| forward_best_window:ISLAND VENTURE | 43 | 44 | F=0.28, FSS1km=0.24 |
| backward_peak_age:ISLAND VENTURE | 43 | 43 | median over members |
| sar_darkness | 6 | 48 | damping 2.5 dB, edge sharpness 0.74 -> weak/weathered-looking (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 16: 0.40 km², 1.6 km long, wind 8.0 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 5}. Age estimate 0.1–12.0 h (no drift-supported suspect: width/Okubo/persistence only).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | not supported | lead only | MAGNOLIA STATE | 338302000 | 0 | 2.35e-05 | 0.0112 | 5 | 0.333 | 46 | 0 | 0 | 0 | False | 0 | 1 | neither | 365 | 0.285 | moving | 0 | False | False | False | False | nan | nan | False |
| 2 | not supported | lead only | OVERSEAS CHINOOK | 366495000 | 0 | 7.87e-07 | 5.51e-05 | 6 | 0 | 21 | 0 | 0 | 0 | False | 0 | 1 | neither | 153 | 0 | moving | 3 | False | False | False | False | nan | nan | False |
| 3 | not supported | lead only | PELICAN STATE | 367353110 | 0 | 8.92e-05 | 0.0457 | 2 | 0.667 | 45 | 0 | 0 | 0 | False | 0 | 1 | neither | 362 | 0.332 | moving | 0 | False | False | False | False | nan | nan | False |
| 4 | not supported | lead only | BLUE SEA II | 367763610 | 0 | 4.11e-05 | 0.0219 | 3 | 0.667 | 47 | 0 | 0 | 0 | False | 0 | 1 | neither | 41.5 | 0.997 | moving | 1 | True | False | False | False | nan | nan | False |
| 5 | not supported | lead only | nan | 941202904 | 0 | 0.000166 | 0.0506 | 1 | 1 | 46 | 0 | 0 | 0 | False | 0 | 1 | neither | 9.96 | 1 | undetermined | 0 | True | False | True | False | nan | nan | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.0547 | 0.547 | W=256 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 3.45 | 3.45 | sigma=W/4=64 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| sar_darkness | 6 | 48 | damping 2.5 dB, edge sharpness 0.65 -> weak/weathered-looking (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 22: 0.74 km², 3.5 km long, wind 8.5 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 23}. Age estimate 13.0–14.0 h (forward best window of top supported suspect (OVERSEAS CASCADE)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | lead only | OVERSEAS CASCADE | 368589000 | 1 | 0.0051 | 0.564 | 1 | 1 | 47 | 0.209 | 0.593 | 0.102 | False | 13 | 14 | forward+backward | 280 | 0.178 | moving | 1 | False | False | False | False | nan | nan | False |
| 2 | consistent | lead only | DEER ISLAND | 368059520 | 0.272 | 0.00146 | 0.603 | 5 | 0 | 47 | 0.0541 | 0.222 | 0.1 | False | 47 | 48 | forward+backward | 78.5 | 0 | moving | 4 | False | False | False | True | nan | nan | False |
| 3 | consistent | lead only | BIG FOOT | 367550980 | 0 | 0.00191 | 0.149 | 4 | 0.333 | 48 | 0 | 0 | 0 | False | 0 | 1 | backward only | 26.5 | 0 | stationary | 6 | False | False | False | False | nan | nan | False |
| 4 | consistent | lead only | DEEP RUNNER | 367643110 | 0 | 0.00117 | 0.255 | 5 | 0.333 | 45 | 0 | 0 | 0 | False | 0 | 1 | backward only | 209 | 0.0318 | moving | 2 | False | False | False | False | nan | nan | False |
| 5 | consistent | lead only | SEAPEAK BAHRAIN | 408919000 | 0 | 0.000741 | 0.143 | 4 | 0.333 | 33 | 0 | 0 | 0 | False | 0 | 1 | backward only | 309 | 0.0805 | moving | 0 | False | False | False | False | nan | nan | False |
| 6 | consistent | lead only | BW PIONEER | 563086300 | 0 | 0.00177 | 0.156 | 2 | 0.667 | 44 | 0 | 0 | 0 | False | 0 | 1 | backward only | 41.5 | 0 | stationary | 5 | False | False | False | False | nan | nan | False |
| 7 | weak | lead only | OSG ENDURANCE | 367501540 | 0.231 | 0.000368 | 0.0104 | 7 | 0.333 | 10 | 0.154 | 0.407 | 0.216 | False | 10 | 11 | forward only | 235 | 0.217 | moving | 2 | False | False | False | False | nan | nan | False |
| 8 | not supported | lead only | SEVEN OCEANS | 235053116 | 0 | 1.26e-07 | 4.48e-05 | 22 | 0 | 48 | 0 | 0 | 0 | False | 0 | 1 | neither | 34.2 | 0 | mixed | 3 | False | False | False | False | ais_stationary | 33.5 | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.0355 | 0.355 | W=208 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 2.89 | 2.89 | sigma=W/4=52 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:OVERSEAS CASCADE | 13 | 14 | F=0.21, FSS1km=0.10 |
| backward_peak_age:OVERSEAS CASCADE | 47 | 47 | median over members |
| forward_best_window:DEER ISLAND | 47 | 48 | F=0.05, FSS1km=0.10 |
| backward_peak_age:DEER ISLAND | 47 | 47 | median over members |
| backward_peak_age:BIG FOOT | 48 | 48 | median over members |
| sar_darkness | 6 | 48 | damping 2.2 dB, edge sharpness 0.47 -> weak/weathered-looking (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 21: 0.43 km², 1.4 km long, wind 8.5 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 24}. Age estimate 46.0–47.0 h (forward best window of top supported suspect (OVERSEAS CASCADE)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | lead only | OVERSEAS CASCADE | 368589000 | 0.59 | 0.00463 | 0.837 | 1 | 1 | 43 | 0.0452 | 0.148 | 0.0706 | False | 46 | 47 | forward+backward | 277 | 0.0967 | moving | 1 | False | False | False | True | nan | nan | False |
| 2 | consistent | lead only | DEER ISLAND | 368059520 | 0.234 | 0.00122 | 0.555 | 4 | 0.333 | 47 | 0.0271 | 0.111 | 0.0169 | False | 47 | 48 | forward+backward | 83.7 | 0 | moving | 4 | False | False | False | True | nan | nan | False |
| 3 | weak | lead only | OSG ENDURANCE | 367501540 | 0.0242 | 2.71e-06 | 7.67e-05 | 18 | 0 | 10 | 0.13 | 0.333 | 0.167 | False | 10 | 11 | forward only | 237 | 0.184 | moving | 2 | False | False | False | False | nan | nan | False |
| 4 | not supported | lead only | SEVEN OCEANS | 235053116 | 0 | 1.03e-06 | 0.000345 | 19 | 0 | 48 | 0 | 0 | 0 | False | 0 | 1 | neither | 30.6 | 0 | mixed | 3 | False | False | False | False | ais_stationary | 30.5 | False |
| 5 | not supported | lead only | POLARCUS ALIMA | 311012300 | 0 | 4.06e-06 | 0.000842 | 13 | 0 | 44 | 0 | 0 | 0 | False | 0 | 1 | neither | 47.5 | 0.271 | moving | 4 | False | False | False | False | nan | nan | False |
| 6 | not supported | lead only | TUCKER CANDIES | 338002000 | 0 | 3.29e-06 | 0.00121 | 17 | 0 | 48 | 0 | 0 | 0 | False | 0 | 1 | neither | 234 | 0 | moving | 1 | False | False | False | False | nan | nan | False |
| 7 | not supported | lead only | BREEZE | 338189000 | 0 | 7.46e-06 | 0.00501 | 12 | 0 | 48 | 0 | 0 | 0 | False | 0 | 1 | neither | 233 | 0 | moving | 2 | False | False | False | False | nan | nan | False |
| 8 | not supported | lead only | MAGNOLIA STATE | 338302000 | 0 | 0.000138 | 0.0445 | 7 | 0 | 43 | 0 | 0 | 0 | False | 0 | 1 | neither | 310 | 0.308 | moving | 0 | False | False | False | False | nan | nan | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.0827 | 0.827 | W=313 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 4.09 | 4.09 | sigma=W/4=78 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:OVERSEAS CASCADE | 46 | 47 | F=0.05, FSS1km=0.07 |
| backward_peak_age:OVERSEAS CASCADE | 43 | 43 | median over members |
| forward_best_window:DEER ISLAND | 47 | 48 | F=0.03, FSS1km=0.02 |
| backward_peak_age:DEER ISLAND | 47 | 47 | median over members |
| sar_darkness | 6 | 48 | damping 2.1 dB, edge sharpness 0.61 -> weak/weathered-looking (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 17: 1.34 km², 3.6 km long, wind 8.6 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 35}. Age estimate 43.0–44.0 h (forward best window of top supported suspect (OVERSEAS CASCADE)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | lead only | OVERSEAS CASCADE | 368589000 | 0.867 | 0.0043 | 0.705 | 2 | 0.667 | 40 | 0.129 | 0.296 | 0.151 | False | 43 | 44 | forward+backward | 284 | 0.0396 | moving | 1 | False | False | False | False | nan | nan | False |
| 2 | consistent | lead only | BIG FOOT | 367550980 | 0 | 0.00572 | 0.234 | 1 | 0.667 | 41 | 0 | 0 | 0 | False | 0 | 1 | backward only | 23.6 | 0 | stationary | 6 | False | False | False | False | nan | nan | False |
| 3 | weak | lead only | DEER ISLAND | 368059520 | 0.459 | 0.00148 | 0.74 | 7 | 0.333 | 48 | 0.104 | 0.296 | 0.109 | False | 47 | 48 | forward+backward | 81.9 | 0 | moving | 4 | False | False | False | True | nan | nan | False |
| 4 | weak | lead only | KERRY EXPRESS | 548704000 | 0.318 | 0.00168 | 0.437 | 7 | 0.222 | 37 | 0.0442 | 0.111 | 0.0169 | False | 36 | 37 | forward+backward | 283 | 0.318 | moving | 3 | False | False | False | False | nan | nan | False |
| 5 | weak | lead only | MAERSK OHIO | 367775000 | 0.275 | 0.00244 | 0.146 | 6 | 0.111 | 19 | 0.0229 | 0.111 | 0.0034 | False | 18 | 19 | forward+backward | 283 | 0.322 | moving | 0 | False | False | False | False | nan | nan | False |
| 6 | weak | lead only | MAGNOLIA STATE | 338302000 | 0.252 | 0.00173 | 0.618 | 10 | 0 | 42 | 0.0271 | 0.148 | 0.0492 | False | 41 | 43 | forward+backward | 287 | 0.265 | moving | 0 | False | False | False | False | nan | nan | False |
| 7 | weak | lead only | C-CONSTRUCTOR | 368196410 | 0.132 | 0.0014 | 0.129 | 8 | 0.333 | 48 | 0.00913 | 0.0741 | 0.0116 | False | 46 | 48 | forward only | 10.3 | 0 | loitering | 2 | False | False | False | True | ais_stationary | 9.56 | False |
| 8 | weak | lead only | PELICAN STATE | 367353110 | 0.11 | 0.000572 | 0.171 | 14 | 0 | 42 | 0.0155 | 0.0741 | 0.00217 | False | 42 | 43 | forward+backward | 284 | 0.327 | moving | 0 | False | False | False | False | nan | nan | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.121 | 1.21 | W=377 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 4.8 | 4.8 | sigma=W/4=94 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:OVERSEAS CASCADE | 43 | 44 | F=0.13, FSS1km=0.15 |
| backward_peak_age:OVERSEAS CASCADE | 40 | 40 | median over members |
| backward_peak_age:BIG FOOT | 41 | 41 | median over members |
| sar_darkness | 6 | 48 | damping 2.0 dB, edge sharpness 0.53 -> weak/weathered-looking (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 19: 0.92 km², 4.4 km long, wind 8.7 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 30}. Age estimate 38.0–39.0 h (forward best window of top supported suspect (BIG FOOT)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | lead only | BIG FOOT | 367550980 | 0.536 | 0.0167 | 0.613 | 1 | 0.667 | 39 | 0.0349 | 0.111 | 0.0141 | False | 38 | 39 | forward+backward | 19.9 | 0 | stationary | 6 | False | False | False | False | nan | nan | False |
| 2 | consistent | lead only | OVERSEAS CASCADE | 368589000 | 0.326 | 0.00344 | 0.781 | 2 | 0.667 | 44 | 0.0628 | 0.222 | 0.131 | False | 46 | 47 | forward+backward | 286 | 0.165 | moving | 1 | False | False | False | True | nan | nan | False |
| 3 | consistent | lead only | DEEP RUNNER | 367643110 | 0.141 | 0.00148 | 0.403 | 5 | 0 | 44 | 0.0271 | 0.111 | 0.0308 | False | 44 | 45 | forward+backward | 198 | 0.0275 | moving | 2 | False | False | False | False | nan | nan | False |
| 4 | consistent | lead only | SHELIA BORDELON | 367655260 | 0 | 0.00197 | 0.45 | 3 | 0.667 | 48 | 0 | 0 | 0 | False | 0 | 1 | backward only | 3.79 | 0.44 | moving | 3 | False | False | False | False | ais_silent_at_t0 | 1.73 | True |
| 5 | weak | lead only | MAGNOLIA STATE | 338302000 | 0.26 | 0.00113 | 0.435 | 10 | 0 | 42 | 0.122 | 0.296 | 0.215 | False | 41 | 42 | forward+backward | 285 | 0.0996 | moving | 0 | False | False | False | False | nan | nan | False |
| 6 | weak | lead only | KERRY EXPRESS | 548704000 | 0.112 | 0.00142 | 0.24 | 10 | 0 | 37 | 0.0178 | 0.0741 | 0.0099 | False | 36 | 37 | forward+backward | 281 | 0.166 | moving | 3 | False | False | False | False | nan | nan | False |
| 7 | weak | lead only | DEER ISLAND | 368059520 | 0.103 | 0.00119 | 0.513 | 7 | 0 | 48 | 0.0182 | 0.0741 | 0.0402 | False | 47 | 48 | forward+backward | 77.8 | 0 | moving | 4 | False | False | False | True | nan | nan | False |
| 8 | weak | lead only | PELICAN STATE | 367353110 | 0.0628 | 0.000463 | 0.177 | 14 | 0 | 42 | 0.0173 | 0.0741 | 0.00199 | False | 42 | 43 | forward+backward | 282 | 0.188 | moving | 0 | False | False | False | False | nan | nan | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.0359 | 0.359 | W=210 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 2.9 | 2.9 | sigma=W/4=52 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:BIG FOOT | 38 | 39 | F=0.03, FSS1km=0.01 |
| backward_peak_age:BIG FOOT | 39 | 39 | median over members |
| forward_best_window:OVERSEAS CASCADE | 46 | 47 | F=0.06, FSS1km=0.13 |
| backward_peak_age:OVERSEAS CASCADE | 44 | 44 | median over members |
| forward_best_window:DEEP RUNNER | 44 | 45 | F=0.03, FSS1km=0.03 |
| backward_peak_age:DEEP RUNNER | 44 | 44 | median over members |
| sar_darkness | 6 | 48 | damping 2.0 dB, edge sharpness 0.61 -> weak/weathered-looking (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Event 18: 0.43 km², 2.7 km long, wind 8.7 m/s (valid)

Funnel: {'ais_vessels_in_window': 1682, 'positioned_ge_min_hours': 1591, 'within_gate_of_backward_cloud': 34}. Age estimate 1.0–2.0 h (forward best window of top supported suspect (SHELIA BORDELON)).

| suspect_rank | tier | v31_decision | VesselName | MMSI | combined | bwd_weighted | bwd_point | bwd_rank_median | bwd_top3_fraction | bwd_peak_age_median | fwd_F | fwd_member_support | fss_1km | fss_skilful | fwd_window_start_h | fwd_window_end_h | direction_agreement | dist_to_slick_end_km | geo_fresh_discharge | motion_state | ais_gaps_ge_60min | flag_fresh_discharge_match | flag_ais_gap_in_release_window | flag_identity_anomaly | flag_window_at_48h_boundary | sar_label | sar_dist_to_slick_km | flag_ais_silent_seen_by_sar_near_slick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | consistent | lead only | SHELIA BORDELON | 367655260 | 0.538 | 0.0028 | 0.516 | 4 | 0.333 | 48 | 0.192 | 0.333 | 0.0102 | False | 1 | 2 | forward+backward | 1.34 | 0.995 | moving | 3 | True | True | False | False | ais_silent_at_t0 | 0.818 | True |
| 2 | consistent | lead only | OVERSEAS CASCADE | 368589000 | 0.409 | 0.00435 | 0.688 | 2 | 0.667 | 44 | 0.0714 | 0.222 | 0.0746 | False | 39 | 40 | forward+backward | 286 | 0.143 | moving | 1 | False | False | False | False | nan | nan | False |
| 3 | consistent | lead only | BIG FOOT | 367550980 | 0 | 0.00968 | 0.387 | 1 | 0.667 | 40 | 0 | 0 | 0 | False | 0 | 1 | backward only | 22.3 | 0 | stationary | 6 | False | False | False | False | nan | nan | False |
| 4 | weak | lead only | PELICAN STATE | 367353110 | 0.216 | 0.000759 | 0.282 | 13 | 0 | 42 | 0.114 | 0.259 | 0.307 | False | 42 | 43 | forward+backward | 281 | 0.211 | moving | 0 | False | False | False | False | nan | nan | False |
| 5 | weak | lead only | KERRY EXPRESS | 548704000 | 0.163 | 0.0011 | 0.252 | 7 | 0 | 37 | 0.0448 | 0.148 | 0.0638 | False | 36 | 37 | forward+backward | 280 | 0.188 | moving | 3 | False | False | False | False | nan | nan | False |
| 6 | weak | lead only | MAGNOLIA STATE | 338302000 | 0.116 | 0.00182 | 0.696 | 9 | 0 | 42 | 0.0139 | 0.0741 | 0.0017 | False | 41 | 42 | forward+backward | 284 | 0.118 | moving | 0 | False | False | False | False | nan | nan | False |
| 7 | weak | lead only | DEEP RUNNER | 367643110 | 0.0825 | 0.00137 | 0.52 | 7 | 0 | 44 | 0.00918 | 0.037 | 0.019 | False | 43 | 44 | forward+backward | 196 | 0.0213 | moving | 2 | False | False | False | False | nan | nan | False |
| 8 | weak | lead only | C-CONSTRUCTOR | 368196410 | 0.0789 | 0.00127 | 0.137 | 8 | 0.333 | 48 | 0.00913 | 0.0741 | 0.0056 | False | 46 | 48 | forward+backward | 14 | 0 | loitering | 2 | False | False | False | True | ais_stationary | 14 | False |

| cue | lo_h | hi_h | detail |
|---|---|---|---|
| width_diffusion | 0.0206 | 0.206 | W=162 m, K 1.0-10.0 m2/s (biased old: E007 strands 2-3x too wide) |
| okubo_1971 | 2.33 | 2.33 | sigma=W/4=40 m |
| persistence_prior | 2 | 12 | typical ship-discharge slick lifetime |
| forward_best_window:SHELIA BORDELON | 1 | 2 | F=0.19, FSS1km=0.01 |
| backward_peak_age:SHELIA BORDELON | 48 | 48 | median over members |
| forward_best_window:OVERSEAS CASCADE | 39 | 40 | F=0.07, FSS1km=0.07 |
| backward_peak_age:OVERSEAS CASCADE | 44 | 44 | median over members |
| backward_peak_age:BIG FOOT | 40 | 40 | median over members |
| sar_darkness | 6 | 48 | damping 2.1 dB, edge sharpness 0.58 -> weak/weathered-looking (heuristic interval; over-wide E007 polygons bias damping LOW) |
| optical_baoac | nan | nan | no coincident optical imagery |

## Forecast T0 → T0+72 h (OpenOil with weathering, GSHHG coastline)

Sea temperature 22.0 °C is an ASSUMPTION (no SST in forcing). Oil type unknown from SAR: medium crude central, light/heavy as sensitivity.

| cluster_id | p_beached_72h | earliest_beaching_h |
|---|---|---|
| 1 | 0 | nan |
| 2 | 0 | nan |
| 3 | 0 | nan |
| 7 | 0 | nan |
| 12 | 0 | nan |
| 15 | 0 | nan |
| 16 | 0 | nan |
| 17 | 0 | nan |
| 18 | 0 | nan |
| 19 | 0 | nan |
| 21 | 0 | nan |
| 22 | 0 | nan |

| oil_type | lead_h | frac_oil | frac_evaporated | frac_dispersed |
|---|---|---|---|---|
| GENERIC HEAVY CRUDE | 24 | 0.243 | 0.144 | 0.612 |
| GENERIC HEAVY CRUDE | 48 | 0.202 | 0.144 | 0.654 |
| GENERIC HEAVY CRUDE | 72 | 0.193 | 0.144 | 0.663 |
| GENERIC LIGHT CRUDE | 24 | 0.0615 | 0.323 | 0.616 |
| GENERIC LIGHT CRUDE | 48 | 0.0263 | 0.323 | 0.651 |
| GENERIC LIGHT CRUDE | 72 | 0.019 | 0.323 | 0.658 |
| GENERIC MEDIUM CRUDE | 24 | 0.174 | 0.26 | 0.566 |
| GENERIC MEDIUM CRUDE | 48 | 0.144 | 0.26 | 0.596 |
| GENERIC MEDIUM CRUDE | 72 | 0.136 | 0.26 | 0.604 |
