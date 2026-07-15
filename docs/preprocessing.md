# Text Preprocessing & Dataset Regeneration Guide

This document explains what the `clean_text` column is, how it is generated, why it is necessary for topic modeling and evaluation metrics, and how to regenerate all datasets in the project.

---

## 1. What is the `clean_text` Column?

The `clean_text` column is a preprocessed version of the raw document text. During topic modeling with BERTopic and evaluation with coherence metrics, using raw text (e.g., social media posts, parliamentary transcripts) introduces significant noise. 

In `src/processing.py`, the preprocessing pipeline standardizes the text inputs into several clean variants:
*   `clean_text`: The primary cleaned text (URLs and numbers removed, specific artifacts stripped).
*   `clean_text_lower`: Lowercased version of `clean_text`.
*   `clean_text_lower_punctless`: Lowercased version with all punctuation and excess whitespace removed.
*   `clean_text_stemmed`: Stemmed words (using the NLTK Snowball Stemmer) applied to the lowercased text.

---

## 2. Preprocessing Steps

The preprocessing pipeline executes the following lazy transformations via Polars in `src/processing.py` (see [src/processing.py](file:///D:/CA-BERTopic/src/processing.py#L281-L342)):

1.  **URL Removal (`remove_urls`)**: Removes hyperlinks matching `https?://\S+`, `www\.\S+`, and common Twitter residues like `t.co/\S+` or `httpstco\S+`.
2.  **Number Removal (`remove_numbers`)**: Strips digits (`\d+`) from the text, preventing arbitrary numbers from cluttering the topic vocabulary.
3.  **Dataset-Specific Artifact Removal (`remove_artifacts`)**: Strips noise strings defined in `ARTIFACTS_TO_REMOVE` (e.g., specific campaign tags, metadata markers, or platform-specific characters).
4.  **Case and Punctuation Normalization**: Standardizes casing and removes punctuation to create `clean_text_lower` and `clean_text_lower_punctless`.
5.  **Stemming (Optional)**: Word-stems the documents using NLTK's `SnowballStemmer` (primarily used for the ANES dataset).
6.  **Empty Row Filtering**: Filters out any documents where `clean_text` is empty (`""`) or consists only of whitespace.
7.  **Deduplication**: Deduplicates the dataset based on `clean_text` uniqueness. If a `date` column is present, it sorts chronologically and preserves the first occurrence.
8.  **Chunking**: Chunks long documents into smaller segments using overlapping sentences (with a tokenizer-aware token limit) to fit the context window of SentenceTransformer models.

---

## 3. Why Preprocessing & Filtering is Critical

### Preventing Metric Failures
Coherence metrics (like $C_V$, $C_{NPMI}$, and $U_{Mass}$) are evaluated using tools from the OCTIS framework. 
*   OCTIS requires a valid token list for each document. 
*   If a document contains **only** numbers, URLs, or specific artifacts (which are stripped during preprocessing), the resulting cleaned document becomes an empty string (`""`).
*   Empty strings result in an empty list of tokens after tokenization. 
*   Passing documents with **zero tokens** to OCTIS causes the coherence scorer to fail or output division-by-zero errors.

### The Trump Dataset Context
On the Trump tweets dataset, many posts consist entirely of a link, a retweet handle, or numbers (e.g., timestamps, poll percentages). If URLs and numbers are removed without dropping the resulting empty rows, the pipeline crashes during the coherence metrics evaluation phase. Dropping these empty rows (as implemented in [data.py](file:///D:/CA-BERTopic/src/data.py#L35-L43)) ensures robust execution.

---

## 4. Manual Dataset Regeneration Steps

Each dataset can be regenerated step-by-step using the following commands:

### A. General Datasets (`fed`, `gadarian`, `trump`)
1.  **Build** the dataset:
    ```bash
    uv run scripts/data_prep/build_datasets.py --dataset <dataset_name>
    ```
2.  **Preprocess** the dataset:
    ```bash
    uv run scripts/data_prep/preprocess_datasets.py --dataset <dataset_name>
    ```
3.  **Generate Embeddings** (creating `clean_text_embedding`):
    ```bash
    uv run scripts/data_prep/generate_embeddings.py --dataset <dataset_name> --columns clean_text
    ```
4.  **Generate R Representations** (creating BoW Parquet and STM RDS):
    ```bash
    Rscript scripts/r_scripts/build_bow.R --dataset <dataset_name>
    ```

### B. ANES (with Stemming)
Since ANES runs both standard and stemmed experiments, it requires generating both standard and stemmed text embeddings, and R representations:
1.  **Build**:
    ```bash
    uv run scripts/data_prep/build_datasets.py --dataset anes
    ```
2.  **Preprocess** (enabling stemming):
    ```bash
    uv run scripts/data_prep/preprocess_datasets.py --dataset anes --stem
    ```
3.  **Generate Embeddings**:
    ```bash
    uv run scripts/data_prep/generate_embeddings.py --dataset anes --columns clean_text clean_text_stemmed
    ```
4.  **Generate R Representations**:
    ```bash
    Rscript scripts/r_scripts/build_bow.R --dataset anes
    ```

### C. Yelp (Large Dataset)
To avoid converting raw Yelp NDJSON files directly (which is extremely slow), skip the raw JSON conversion:
1.  **Build**:
    ```bash
    uv run scripts/data_prep/build_datasets.py --dataset yelp --skip-convert
    ```
2.  **Preprocess**:
    ```bash
    uv run scripts/data_prep/preprocess_datasets.py --dataset yelp
    ```
3.  **Generate Embeddings**:
    ```bash
    uv run scripts/data_prep/generate_embeddings.py --dataset yelp --columns clean_text
    ```
4.  **Generate R Representations** (sampling 10k rows for STM):
    ```bash
    Rscript scripts/r_scripts/build_bow.R --dataset yelp --sample 10000
    ```
5.  **Align Sample (10k Sample)**:
    Since standard experiments run on a 10k sampled version of Yelp, execute the alignment script to sample and synchronize the document IDs:
    ```bash
    uv run python scripts/data_prep/align_yelp_sample.py
    ```

---

## 5. Automated Regeneration Script

We have provided a PowerShell script to automate the entire process from scratch:

```powershell
# Run for all datasets by default
.\scripts\pipelines\local_windows\build_all_datasets.ps1

# Run only for specific datasets
.\scripts\pipelines\local_windows\build_all_datasets.ps1 -Datasets "fed", "anes"
```

Refer to [build_all_datasets.ps1](../scripts/pipelines/local_windows/build_all_datasets.ps1) for execution and options.

---

## 6. What is `align_yelp_sample.py` and Why is it Needed?

The script [align_yelp_sample.py](../scripts/data_prep/align_yelp_sample.py) is a specialized dataset alignment utility. Its primary goal is to **subsample the massive Yelp dataset (16+ GB) down to a representative 10,000-document set while maintaining exact document alignment between BERTopic and STM**.

### The Core Problem: Chunked vs. Un-chunked Alignment
BERTopic and Structural Topic Models (STM) handle document token limits and inputs differently:
1.  **BERTopic (Chunked)**: During Python preprocessing, long documents are split into smaller chunks (under the token limit of the SentenceTransformer model) with overlapping sentences. Consequently, a single Yelp review can produce multiple chunks in `yelp_embeddings.parquet`.
2.  **STM (Un-chunked)**: Structural Topic Models operate on the bag-of-words representation of the *original, un-chunked reviews*.

To compare the performance of these models fairly, **they must run on the exact same subset of reviews**. If we randomly sampled 10,000 chunks for BERTopic, and 10,000 reviews for STM independently, their source texts would not match.

### How `align_yelp_sample.py` Resolves This
The alignment script synchronizes the datasets through three steps:

1.  **Sample Chunks from Embeddings**:
    It samples exactly 10,000 chunks from `yelp_embeddings.parquet` using a fixed seed (`36201624`). This creates the standard subset file `yelp_s10000_embeddings.parquet` used for BERTopic experiments.
2.  **Map Chunks to Source Document IDs**:
    It joins these 10,000 sampled chunk indices with `yelp_processed.parquet` to find the unique source document IDs (`id`) that those chunks originated from.
3.  **Reconstruct matching Un-chunked Sample**:
    It loads the raw processed Yelp dataset (`yelp_reviews.parquet`), cleans the text, filters for only the documents matching the source IDs found in Step 2, and saves it to `yelp_s10000_unchunked.parquet`.

This ensures that the un-chunked R bag-of-words and RDS representations (generated from `yelp_s10000_unchunked.parquet` by `build_bow.R`) contain the **exact same original reviews** as the chunked embeddings used by BERTopic.

