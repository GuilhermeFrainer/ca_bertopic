import polars as pl


def generate_latex_table(df: pl.DataFrame) -> str:
    renamed_df = df.with_columns(
        pl.col("model_name").str.replace_all("_", " ")
    ).drop([
        "outliers",
        "duration_seconds"
    ]).rename({
        "model_name": "Model",
        "n_topics": "Topics",
        "u_mass": "$U_{Mass}$",
        "c_v": "$c_v$",
        "c_npmi": "$c_{npmi}$",
        "irbo": "IRBO",
        "topic_diversity": "Diversity"
    })

    return renamed_df.to_pandas().to_latex(index=False, float_format="%.3f")

