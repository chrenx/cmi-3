FEATS_MAP = {
    "biological_sex": {
        "male": 1,
        "female": 0,
    },
    "infancy_vac": {
        "wp": 1, 
        "ap": 0,
    },
    "ethnicity": {
        "unknown": 0, 
        "hispanic or latino": 1,
        "not hispanic or latino": 2,
    },
    "race": {
        "unknown or not reported": 0,
        "american indian/alaska native": 1,
        "native hawaiian or other pacific islander": 2,
        "black or african american": 3,
        "white": 4,
        "asian": 5,
        "more than one race": 6,
    }
}


FEATS_1_filename = {
    "gene_expr": "data_1_processed_gene_expr_merged.parquet",
    "ab_titer": "data_1_processed_ab_titer_merged.parquet",
    "cell_freq": "data_1_processed_cell_freq_merged.parquet",
    "cytokine_olink": "data_1_processed_cytokine_olink_merged.parquet",
    }

FEATS_2_filename = {
    "gene_expr": "data_2_processed_gene_expr_merged.parquet",
    "ab_titer": "data_2_processed_ab_titer_merged.parquet",
    "cell_freq": "data_2_processed_cell_freq_merged.parquet",
    "cytokine_olink": "data_2_processed_cytokine_olink_merged.parquet",
    "cytokine_legendplex": "data_2_processed_cytokine_legendplex_merged.parquet",
    "tcell_activ": "data_2_processed_tcell_activ_merged.parquet",
    "tcell_polar": "data_2_processed_tcell_polar_merged.parquet",
    }

# FEATS_1_filename = ["data_1_processed_gene_expr_merged.parquet",
#            "data_1_processed_ab_titer_merged.parquet",
#            "data_1_processed_cell_freq_merged.parquet",
#            "data_1_processed_cytokine_olink_merged.parquet"]

# FEATS_2_filename = ["data_2_processed_gene_expr_merged.parquet",
#            "data_2_processed_ab_titer_merged.parquet",
#            "data_2_processed_cell_freq_merged.parquet",
#            "data_2_processed_cytokine_olink_merged.parquet",
#            "data_2_processed_cytokine_legendplex_merged.parquet",
#            "data_2_processed_tcell_activ_merged.parquet",
#            "data_2_processed_tcell_polar_merged.parquet"]
