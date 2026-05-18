library(stm)
library(arrow)
library(optparse)
library(here)
library(jsonlite)

main <- function() {
    option_list <- list(
        make_option(c("-r", "--rds_path"), type = "character", help = "Path to stm_data.rds"),
        make_option(c("-k", "--k"), type = "integer", help = "Number of topics"),
        make_option(c("-i", "--indices_path"), type = "character", default = NULL, help = "Path to JSON sample indices"),
        make_option(c("-o", "--output_dir"), type = "character", help = "Directory to save outputs"),
        make_option(c("-s", "--seed"), type = "integer", default = 42, help = "Random seed"),
        make_option(c("--model_path"), type = "character", help = "Path to save the full model RDS")
    )
    
    opt <- parse_args(OptionParser(option_list = option_list))
    
    if (!dir.exists(opt$output_dir)) {
        dir.create(opt$output_dir, recursive = TRUE)
    }
    
    set.seed(opt$seed)
    
    cat("Loading data...\n")
    stm_data <- readRDS(opt$rds_path)
    
    # 1. Apply sampling if indices provided
    if (!is.null(opt$indices_path) && file.exists(opt$indices_path)) {
        cat("Applying sampling...\n")
        indices <- jsonlite::fromJSON(opt$indices_path)
        
        # Filter based on the 'index' column in meta
        if (!"index" %in% names(stm_data$meta)) {
            stop("Column 'index' not found in stm_data$meta. Cannot apply sampling.")
        }
        
        keep_mask <- stm_data$meta$index %in% indices
        cat(sprintf("Keeping %d out of %d documents after sampling.\n", sum(keep_mask), length(keep_mask)))
        
        stm_data$meta <- stm_data$meta[keep_mask, ]
        stm_data$documents <- stm_data$documents[keep_mask]
    }
    
    # Ensure there are documents left
    if (length(stm_data$documents) == 0) {
        stop("No documents left to train the model.")
    }
    
    cat(sprintf("Training STM model with K=%d...\n", opt$k))
    
    start_time <- Sys.time()
    
    model <- stm(
        documents = stm_data$documents,
        vocab = stm_data$vocab,
        K = opt$k,
        data = stm_data$meta,
        init.type = "Spectral",
        seed = opt$seed,
        verbose = TRUE
    )
    
    end_time <- Sys.time()
    duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
    cat(sprintf("Training finished in %.2f seconds.\n", duration))
    
    # 2. Save Model
    cat("Saving model RDS...\n")
    saveRDS(model, file = opt$model_path)
    
    # 3. Export for Python Evaluation
    cat("Exporting beta, theta and vocab...\n")
    
    # Beta (Topic-Word)
    # model$beta$logbeta is a list of K x V matrices.
    # Without content covariates, it has one element.
    beta <- exp(model$beta$logbeta[[1]])
    write_parquet(as.data.frame(beta), file.path(opt$output_dir, "beta.parquet"))
    
    # Theta (Document-Topic)
    theta <- as.data.frame(model$theta)
    write_parquet(theta, file.path(opt$output_dir, "theta.parquet"))
    
    # Vocab
    vocab_con <- file(file.path(opt$output_dir, "vocab.txt"), open = "wt", encoding = "UTF-8")
    writeLines(model$vocab, vocab_con)
    close(vocab_con)
    
    # Write duration to a small file for python to read
    duration_con <- file(file.path(opt$output_dir, "duration.txt"), open = "wt", encoding = "UTF-8")
    writeLines(as.character(duration), duration_con)
    close(duration_con)
    
    cat("Done!\n")
}

if (sys.nframe() == 0) {
    main()
}
