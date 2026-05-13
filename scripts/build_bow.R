library(arrow)
library(dplyr)
library(quanteda)
library(SnowballC)
library(optparse)
library(here)

main <- function() {
    option_list <- list(
        make_option(
            c("-d", "--dataset"),
            type = "character",
            help = "Dataset to use (e.g., 'fed', 'anes', 'gadarian', 'trump', 'yelp')"),
        make_option(
            c("--min_freq"),
            type = "integer",
            default = 1,
            help = "Minimum document frequency for terms (default: 1)"),
        make_option(
            c("--max_freq_pct"),
            type = "double",
            default = 1.0,
            help = "Maximum document frequency percentage for terms (default: 1.0)"),
        make_option(
            c("--no_stem"),
            action = "store_true",
            default = FALSE,
            help = "Disable stemming (default: FALSE)")
    )
    
    opt <- parse_args(OptionParser(option_list = option_list))
    
    if (is.null(opt$dataset)) {
        stop("Dataset name is required. Use --dataset <name>")
    }
    
    input_path <- here::here("data", "processed", paste0(opt$dataset, "_embeddings.parquet"))
    if (!file.exists(input_path)) {
        stop(paste("Input file not found:", input_path))
    }
    
    cat(sprintf("Processing dataset: %s\n", opt$dataset))
    
    # Set arrow option to avoid issues with dictionary/factor conversion
    options(arrow.use_factors = FALSE)
    
    # Selective loading: exclude embeddings
    cat("Loading data (excluding embeddings)...\n")
    tab <- arrow::read_parquet(input_path, as_data_frame = FALSE)
    all_cols <- names(tab)
    meta_cols <- all_cols[!grepl("embedding", all_cols)]
    
    # Select columns on the table
    tab <- tab[, meta_cols]
    
    # Identify dictionary columns to cast them to string (fixes conversion issues)
    schema <- tab$schema
    dict_cols <- meta_cols[sapply(meta_cols, function(x) inherits(schema[[x]]$type, "DictionaryType"))]
    
    if (length(dict_cols) > 0) {
        cat(sprintf("Casting dictionary columns to string: %s\n", paste(dict_cols, collapse=", ")))
        for (col in dict_cols) {
            # Use cast on the ChunkedArray
            tab[[col]] <- tab[[col]]$cast(arrow::utf8())
        }
    }
    
    data <- as.data.frame(tab)
    
    cat(sprintf("Loaded %d rows.\n", nrow(data)))
    
    # Tokenization and Cleaning
    cat("Tokenizing and cleaning text...\n")
    # We use 'clean_text' as the source for BoW
    if (!"clean_text" %in% names(data)) {
        stop("Column 'clean_text' not found in dataset.")
    }
    
    corp <- quanteda::corpus(data, text_field = "clean_text")
    
    toks <- quanteda::tokens(corp,
                             remove_punct = TRUE,
                             remove_symbols = TRUE,
                             remove_numbers = TRUE,
                             remove_url = TRUE,
                             remove_separators = TRUE)
    
    toks <- quanteda::tokens_tolower(toks)
    toks <- quanteda::tokens_remove(toks, quanteda::stopwords("en"))
    
    if (!opt$no_stem) {
        cat("Stemming tokens...\n")
        toks <- quanteda::tokens_wordstem(toks)
    }
    
    # Create DFM
    cat("Creating Document-Feature Matrix...\n")
    dfm_obj <- quanteda::dfm(toks)
    
    # Filtering
    if (opt$min_freq > 1 || opt$max_freq_pct < 1.0) {
        cat(sprintf("Filtering: min_freq=%d, max_freq_pct=%.2f\n", opt$min_freq, opt$max_freq_pct))
        dfm_obj <- quanteda::dfm_trim(dfm_obj,
                                      min_docfreq = opt$min_freq,
                                      max_docfreq = opt$max_freq_pct,
                                      docfreq_type = "prop")
    }
    
    # 1. Output RDS for STM
    cat("Preparing and saving STM data (RDS)...\n")
    # stm_data contains $documents, $vocab, and $meta (aligned with documents)
    # quanteda::convert automatically includes docvars in $meta
    stm_data <- quanteda::convert(dfm_obj, to = "stm")
    
    rds_output_path <- here::here("data", "processed", paste0(opt$dataset, "_stm_data.rds"))
    saveRDS(stm_data, file = rds_output_path)
    cat(sprintf("Saved RDS to: %s\n", rds_output_path))
    
    # 2. Output BoW Parquet
    cat("Preparing and saving BoW Parquet...\n")
    # Reconstruct bow_text from tokens or DFM
    kept_features <- quanteda::featnames(dfm_obj)
    
    # Filter tokens to keep only those in DFM
    toks_filtered <- quanteda::tokens_select(toks, pattern = kept_features, selection = "keep")
    
    # Collapse tokens back to string
    # sapply on tokens returns a named character vector
    bow_text_vec <- sapply(toks_filtered, paste, collapse = " ")
    
    # Ensure it's a character vector of the same length as data
    if (length(bow_text_vec) != nrow(data)) {
        stop(sprintf("Length mismatch: bow_text (%d) vs data (%d)", length(bow_text_vec), nrow(data)))
    }
    
    data$bow_text <- as.character(bow_text_vec)
    
    parquet_output_path <- here::here("data", "processed", paste0(opt$dataset, "_bow.parquet"))
    arrow::write_parquet(data, parquet_output_path)
    cat(sprintf("Saved Parquet to: %s\n", parquet_output_path))
    
    cat("Done!\n")
}

if (sys.nframe() == 0) {
    main()
}
