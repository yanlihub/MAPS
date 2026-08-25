#!/usr/bin/env Rscript
# Generate a synthetic pool with synthpop's sequential CART synthesis.
#
# synthpop is a non-neural, statistically motivated synthesiser from the official statistics
# community. syn(m = k) produces k synthetic datasets of size nrow(data), which together
# form the k*N pool MAPS expects.

# Drop or edit this if synthpop is installed in the default R library path.
.libPaths(c("~/R/library", .libPaths()))
suppressPackageStartupMessages(library(synthpop))

args <- commandArgs(trailingOnly = TRUE)
in_csv  <- args[1]
out_csv <- args[2]
m       <- as.integer(args[3])
seed    <- if (length(args) >= 4) as.integer(args[4]) else 0
method  <- if (length(args) >= 5) args[5] else "cart"

cat(sprintf("[synthpop] reading %s\n", in_csv))
d <- read.csv(in_csv, stringsAsFactors = TRUE)
cat(sprintf("[synthpop] %d records, %d variables; m = %d, method = %s\n",
            nrow(d), ncol(d), m, method))

# Character columns must be factors for CART synthesis.
for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- as.factor(d[[nm]])

t0 <- Sys.time()
# print.flag exposes per-variable progress. Without it a multi-hour run gives no signal at
# all, which makes it impossible to tell a slow run from a stuck one or to estimate a finish.
syn_out <- syn(d, m = m, method = method, seed = seed, print.flag = TRUE)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

pool <- if (m == 1) syn_out$syn else do.call(rbind, syn_out$syn)
cat(sprintf("[synthpop] generated %d records in %.1f s\n", nrow(pool), elapsed))

write.csv(pool, out_csv, row.names = FALSE)
writeLines(
  sprintf('{"n_pool": %d, "m": %d, "method": "%s", "seed": %d, "wall_seconds": %.2f}',
          nrow(pool), m, method, seed, elapsed),
  sub("\\.csv$", "_meta.json", out_csv)
)
cat(sprintf("[synthpop] wrote %s\n", out_csv))
