import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import polars as pl
    from tritopic import TriTopic

    from src import evaluation

    return Path, TriTopic, evaluation, np, pl


@app.cell
def _(Path, np, pl):
    # Locate Yelp embeddings dataset
    data_dir = Path("data/processed")
    dataset_path = data_dir / "yelp_s10000_embeddings.parquet"
    if not dataset_path.exists():
        dataset_path = data_dir / "yelp_embeddings.parquet"

    print(f"Loading Yelp dataset from: {dataset_path}")
    df = pl.read_parquet(dataset_path)

    # Subsample for quick barebones exploration
    SAMPLE_SIZE = 1000
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(n=SAMPLE_SIZE, seed=42)
    else:
        df_sample = df

    documents = df_sample["clean_text"].to_list()
    embeddings = np.array(df_sample["clean_text_embedding"].to_list())

    # Extract metadata columns if present
    metadata_cols = ["stars", "user_average_stars", "business_stars", "state"]
    metadata_cols = [c for c in metadata_cols if c in df_sample.columns]
    metadata_df = df_sample.select(metadata_cols).to_pandas()

    print(
        f"Loaded {len(documents)} documents, embeddings shape: {embeddings.shape}, metadata shape: {metadata_df.shape}"
    )
    return documents, embeddings, metadata_df


@app.cell
def _(TriTopic, documents, embeddings, metadata_df):
    # Initialize and fit TriTopic model
    model = TriTopic(verbose=True)
    labels = model.fit_transform(
        documents=documents, embeddings=embeddings, metadata=metadata_df
    )

    topic_info = model.get_topic_info()
    print("\nDiscovered Topics:")
    print(topic_info)
    return model, topic_info


@app.cell
def _(model, topic_info):
    # Inspect representative documents for the top topic
    if topic_info is not None and not topic_info.empty:
        top_topic_id = topic_info["Topic"].iloc[0]
        rep_docs = model.get_representative_docs(topic_id=top_topic_id, n_docs=3)
        print(f"\nRepresentative documents for Topic {top_topic_id}:")
        for doc_idx, doc_text in rep_docs:
            print(f" - Doc [{doc_idx}]: {doc_text[:120]}...")
    return


@app.cell
def _(documents, evaluation, model):
    # 1. Format TriTopic top keywords into OCTIS structure
    topic_words = [
        topic.keywords[:10] for topic in model.topics_ if topic.topic_id != -1
    ]
    octis_output = evaluation.topic_words_to_octis(topic_words)

    # 2. Tokenize corpus texts for evaluation
    tokenized_texts = [doc.lower().split() for doc in documents]
    tokenized_texts = [t for t in tokenized_texts if len(t) > 0]

    # 3. Compute Coherence & Diversity metrics used in CA-BERTopic
    metrics = {}

    # Coherence metrics (c_v, u_mass, c_npmi)
    for measure in ["c_v", "u_mass", "c_npmi"]:
        metrics[measure] = evaluation.compute_coherence(
            model_output=octis_output, texts=tokenized_texts, measure=measure
        )

    # Diversity metrics (topic_diversity, irbo)
    for dm in ["topic_diversity", "irbo"]:
        metrics[dm] = evaluation.compute_diversity(dm, model_output=octis_output)

    print("\nTriTopic Evaluation Metrics (OCTIS):")
    for metric_name, score in metrics.items():
        print(f" - {metric_name:15s}: {score:.4f}")
    return


@app.cell
def _(model):
    t = model.topics_[0]
    len(t.keywords)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
