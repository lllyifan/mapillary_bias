# IO Manifest (static scan, strong resolver)

## scripts\ICE_IA.py

**Reads**

- L86: `rf_model_ClusterType.joblib` (.load())
- L87: `val_set_ClusterType.csv` (.read_csv())

**Writes**

- L161: `ICE_{...}.png` (.savefig())
- L161: `ICE_{...}.png` (.savefig())
- L195: `SpatialICE_{...}.png` (.savefig())
- L195: `SpatialICE_{...}.png` (.savefig())

## scripts\ICE_IP.py

**Reads**

- L86: `rf_model_ClusterType.joblib` (.load())
- L87: `val_set_ClusterType.csv` (.read_csv())

**Writes**

- L161: `ICE_{...}.png` (.savefig())
- L161: `ICE_{...}.png` (.savefig())
- L195: `SpatialICE_{...}.png` (.savefig())
- L195: `SpatialICE_{...}.png` (.savefig())

## scripts\OCI_IA.py

**Reads**

- L67: `LSOA_joined_remodeling_noShapeArea.shp` (.read_file())
- L68: `LSOA_joined_remodeling_rate.csv` (.read_csv())
- L78: `IA.col.txt` (open())

**Writes**

- L142: `OCI_results.csv` (.to_csv())

## scripts\OCI_IP.py

**Reads**

- L65: `LSOA_joined_remodeling_noShapeArea.shp` (.read_file())
- L66: `LSOA_joined_remodeling_rate.csv` (.read_csv())
- L80: `IP.col.txt` (open())

**Writes**

- L142: `OCI_results.csv` (.to_csv())

## scripts\RF_IA.py

**Reads**

- L42: `matched_lsoas_from_clusters_IA.csv` (.read_csv())

**Writes**

- L111: `rf_model_ClusterType.joblib` (.dump())
- L164: `predictions_val_ClusterType.csv` (.to_csv())
- L170: `train_set_ClusterType.csv` (.to_csv())
- L171: `val_set_ClusterType.csv` (.to_csv())

## scripts\RF_IP.py

**Reads**

- L42: `matched_lsoas_from_clusters_IP.csv` (.read_csv())

**Writes**

- L111: `rf_model_ClusterType.joblib` (.dump())
- L164: `predictions_val_ClusterType.csv` (.to_csv())
- L170: `train_set_ClusterType.csv` (.to_csv())
- L171: `val_set_ClusterType.csv` (.to_csv())

## scripts\SHAP_IA.py

**Reads**

- L45: `rf_model_ClusterType.joblib` (.load())
- L46: `predictions_val_ClusterType.csv` (.read_csv())

**Writes**

- L68: `shap_summary_bar_low.png` (.savefig())
- L68: `shap_summary_bar_low.png` (.savefig())
- L75: `shap_values_low.npy` (.save())
- L76: `shap_X_val.csv` (.to_csv())

## scripts\SHAP_INTERACTION_IA.py

**Reads**

- L48: `rf_model_ClusterType.joblib` (.load())
- L49: `val_set_ClusterType.csv` (.read_csv())

**Writes**

- L128: `shap_interaction_signed_sum_heatmap_low.png` (.savefig())
- L128: `shap_interaction_signed_sum_heatmap_low.png` (.savefig())
- L136: `shap_inter_raw_low.npy` (.save())
- L137: `shap_signed_sum_low.csv` (.to_csv())
- L142: `shap_mask_low.npy` (.save())
- L143: `shap_inter_all.joblib` (.dump())

## scripts\SHAP_INTERACTION_IP.py

**Reads**

- L48: `rf_model_ClusterType.joblib` (.load())
- L49: `val_set_ClusterType.csv` (.read_csv())

**Writes**

- L128: `shap_interaction_signed_sum_heatmap_low.png` (.savefig())
- L128: `shap_interaction_signed_sum_heatmap_low.png` (.savefig())
- L136: `shap_inter_raw_low.npy` (.save())
- L137: `shap_signed_sum_low.csv` (.to_csv())
- L142: `shap_mask_low.npy` (.save())
- L143: `shap_inter_all.joblib` (.dump())

## scripts\SHAP_IP.py

**Reads**

- L45: `rf_model_ClusterType.joblib` (.load())
- L46: `predictions_val_ClusterType.csv` (.read_csv())

**Writes**

- L68: `shap_summary_bar_low.png` (.savefig())
- L68: `shap_summary_bar_low.png` (.savefig())
- L75: `shap_values_low.npy` (.save())
- L76: `shap_X_val.csv` (.to_csv())

## scripts\fig_2_A.py

**Reads**

- L64: `LSOA_joined_remodeling.shp` (.read_file())
- L65: `regions_wales_england.geojson` (.read_file())
- L66: `cities9_point.shp` (.read_file())

**Writes**

- L227: `fig_2_A.pdf` (.savefig())
- L227: `fig_2_A.pdf` (.savefig())

## scripts\fig_2_B.py

**Reads**

- L63: `LSOA_joined_remodeling_with_centroid.shp` (.read_file())

**Writes**

- L153: `fig_2_B.pdf` (.savefig())
- L153: `fig_2_B.pdf` (.savefig())

## scripts\fig_3_A_left.py

**Reads**

- L56: `LSOA_joined_remodeling_noShapeArea.shp` (.read_file())
- L57: `regions_wales_england.geojson` (.read_file())
- L65: `IP.col.txt` (open())

**Writes**

- L337: `fig_3_A_left.pdf` (.savefig())
- L337: `fig_3_A_left.pdf` (.savefig())
- L339: `fig_3_A_left.tiff` (.savefig())
- L339: `fig_3_A_left.tiff` (.savefig())

## scripts\fig_3_A_right.py

**Reads**

- L30: `LSOA_joined_remodeling_noShapeArea.shp` (.read_file())
- L31: `regions_wales_england.geojson` (.read_file())
- L36: `IP.col.txt` (open())

**Writes**

- L128: `fig_3_A_right.pdf` (.savefig())
- L128: `fig_3_A_right.pdf` (.savefig())

## scripts\fig_3_B_left.py

**Reads**

- L56: `LSOA_joined_remodeling_noShapeArea.shp` (.read_file())
- L57: `regions_wales_england.geojson` (.read_file())
- L65: `IA.col.txt` (open())

**Writes**

- L337: `fig_3_B_left.pdf` (.savefig())
- L337: `fig_3_B_left.pdf` (.savefig())
- L339: `fig_3_B_left.tiff` (.savefig())
- L339: `fig_3_B_left.tiff` (.savefig())

## scripts\fig_3_B_right.py

**Reads**

- L30: `LSOA_joined_remodeling_noShapeArea.shp` (.read_file())
- L31: `regions_wales_england.geojson` (.read_file())
- L36: `IA.col.txt` (open())

**Writes**

- L128: `fig_3_B_right.pdf` (.savefig())
- L128: `fig_3_B_right.pdf` (.savefig())

## scripts\fig_4_A.py

**Reads**

- L24: `OCI_results.csv` (.read_csv())

**Writes**

- L106: `fig_4_A.pdf` (.savefig())
- L106: `fig_4_A.pdf` (.savefig())

## scripts\fig_4_B.py

**Reads**

- L24: `OCI_results.csv` (.read_csv())

**Writes**

- L106: `fig_4_B.pdf` (.savefig())
- L106: `fig_4_B.pdf` (.savefig())

## scripts\fig_5_A_left.py

**Reads**

- L69: `shap_values_low.npy` (.load())
- L70: `shap_X_val.csv` (.read_csv())

**Writes**

- L233: `fig_5_A_left.pdf` (.savefig())
- L233: `fig_5_A_left.pdf` (.savefig())

## scripts\fig_5_A_right.py

**Reads**

- L61: `shap_values_low.npy` (.load())

**Writes**

- L169: `fig_5_A_right.pdf` (.savefig())
- L169: `fig_5_A_right.pdf` (.savefig())

## scripts\fig_5_B_left.py

**Reads**

- L67: `shap_values_low.npy` (.load())
- L68: `shap_X_val.csv` (.read_csv())

**Writes**

- L222: `fig_5_B_left.pdf` (.savefig())
- L222: `fig_5_B_left.pdf` (.savefig())

## scripts\fig_5_B_right.py

**Reads**

- L61: `shap_values_low.npy` (.load())

**Writes**

- L169: `fig_5_B_right.pdf` (.savefig())
- L169: `fig_5_B_right.pdf` (.savefig())

## scripts\fig_6_A.py

**Reads**

- L79: `shap_X_val.csv` (.read_csv())
- L80: `shap_values_low.npy` (.load())
- L278: `lowess_{...}_ci.npz` (.load())

**Writes**

- L122: `lowess_{...}_ci.npz` (.savez())
- L382: `fig_6_A.pdf` (.savefig())
- L382: `fig_6_A.pdf` (.savefig())

## scripts\fig_6_B.py

**Reads**

- L78: `shap_X_val.csv` (.read_csv())
- L79: `shap_values_low.npy` (.load())
- L263: `shap_X_val.csv` (.read_csv())
- L264: `shap_values_low.npy` (.load())
- L273: `lowess_{...}_ci.npz` (.load())

**Writes**

- L118: `lowess_{...}_ci.npz` (.savez())
- L373: `fig_6_B.pdf` (.savefig())
- L373: `fig_6_B.pdf` (.savefig())

## scripts\fig_7_A_left.py

**Reads**

- L73: `shap_signed_sum_low.csv` (.read_csv())

**Writes**

- L167: `fig_7_A_left.pdf` (.savefig())
- L167: `fig_7_A_left.pdf` (.savefig())

## scripts\fig_7_A_right.py

**Reads**

- L86: `shap_inter_raw_low.npy` (.load())
- L88: `shap_signed_sum_low.csv` (.read_csv())

**Writes**

- L165: `fig_7_A_right.pdf` (.savefig())
- L165: `fig_7_A_right.pdf` (.savefig())

## scripts\fig_7_B_left.py

**Reads**

- L77: `shap_signed_sum_low.csv` (.read_csv())

**Writes**

- L158: `fig_7_B_left.pdf` (.savefig())
- L158: `fig_7_B_left.pdf` (.savefig())

## scripts\fig_7_B_right.py

**Reads**

- L86: `shap_inter_raw_low.npy` (.load())
- L88: `shap_signed_sum_low.csv` (.read_csv())

**Writes**

- L159: `fig_7_B_right.pdf` (.savefig())
- L159: `fig_7_B_right.pdf` (.savefig())

## scripts\fig_8_A.py

**Reads**

- L97: `regions_wales_england.geojson` (.read_file())
- L154: `slopes_{...}_{...}.csv` (.read_csv())

**Writes**

- L222: `fig_8_IP_ICE_spatial_slopes_{...}.pdf` (.savefig())
- L222: `fig_8_IP_ICE_spatial_slopes_{...}.pdf` (.savefig())

## scripts\fig_8_B.py

**Reads**

- L97: `regions_wales_england.geojson` (.read_file())
- L154: `slopes_{...}_{...}.csv` (.read_csv())

**Writes**

- L222: `fig_8_IA_ICE_spatial_slopes_{...}.pdf` (.savefig())
- L222: `fig_8_IA_ICE_spatial_slopes_{...}.pdf` (.savefig())

