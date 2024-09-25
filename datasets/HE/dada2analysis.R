library(dada2); packageVersion("dada2")
library(set);
path <- "seqs/" # CHANGE ME to the directory containing the fastq files after unzipping.
list.files(path)

# Forward and reverse fastq filenames have format: SAMPLENAME_R1_001.fastq and SAMPLENAME_R2_001.fastq
fnFs <- sort(list.files(path, pattern=".fastq", full.names = TRUE))
# fnRs <- sort(list.files(path, pattern="_R2_001.fastq", full.names = TRUE))
# Extract sample names, assuming filenames have format: SAMPLENAME_XXX.fastq
filtpath <- file.path(path, "filtered")
# sample.names <- sapply(strsplit(basename(fnFs), "_"), `[`, 1)

plotQualityProfile(fnFs[1:2])

# plotQualityProfile(fnRs[1:2])

# Place filtered files in filtered/ subdirectory
out <- filterAndTrim(file.path(path, fns), file.path(filtpath, fns), truncLen=240,maxEE=1,truncQ=11,rm.phix=TRUE,compress=TRUE,verbose=TRUE,
              multithread=TRUE)
# filtFs <- file.path(path, "filtered", paste0(sample.names, "_F_filt.fastq.gz"))
# filtRs <- file.path(path, "filtered", paste0(sample.names, "_R_filt.fastq.gz"))

# names(filtFs) <- sample.names
# names(filtRs) <- sample.names
# 
# out <- filterAndTrim(fnFs, filtFs, fnRs, filtRs, truncLen=c(240,160),
#                      maxN=0, maxEE=c(2,2), truncQ=2, rm.phix=TRUE,
#                      compress=TRUE, multithread=TRUE) # On Windows set multithread=FALSE
head(out)

errF <- learnErrors(file.path(filtpath, fns), multithread=TRUE)

# errR <- learnErrors(filtRs, multithread=TRUE)

plotErrors(errF, nominalQ=TRUE)

dadaFs <- dada(file.path(filtpath, fns), err=errF, multithread=TRUE)

# dadaRs <- dada(filtRs, err=errR, multithread=TRUE)

# mergers <- mergePairs(dadaFs, filtFs, dadaRs, filtRs, verbose=TRUE)

##################################################################################
head(mergers)
seqtab <- makeSequenceTable(dadaFs)
dim(seqtab)

seqtab.nochim <- removeBimeraDenovo(seqtab, method="consensus", multithread=TRUE, verbose=TRUE)
dim(seqtab.nochim)
sum(seqtab.nochim)/sum(seqtab)

getN <- function(x) sum(getUniques(x))
track <- cbind(out, sapply(dadaFs, getN), sapply(dadaRs, getN), sapply(mergers, getN), rowSums(seqtab.nochim))
tracktots <- colSums(track)
# If processing a single sample, remove the sapply calls: e.g. replace sapply(dadaFs, getN) with getN(dadaFs)
colnames(track) <- c("input", "filtered", "denoisedF", "denoisedR", "merged", "nonchim")
rownames(track) <- sample.names
head(track)

write.table(t(seqtab.nochim), 'seqtab-nochim.txt',append = FALSE, sep = " ", 
             row.names = TRUE, col.names = TRUE)
write.csv(t(seqtab.nochim), 'seqtab-nochim.csv')
totals <- colSums(track)
one <- -(totalsl["filtered"] - totals["input"])/totals["input"]
two <- -(totals["denoisedF"] - totals["filtered"])/totals["filtered"]
three <- -(totals["denoisedR"] - totals["filtered"])/totals["filtered"]
four <- -(totals["merged"] - totals["denoisedF"])/totals["denoisedF"]
five <- -(totals["nonchim"]- totals["merged"])/totals["merged"]
perc_change <- c(0, one, two, three, four, five)
newdat <- rbind(totals, perc_change, track)
write.csv(newdat, 'tracking.csv')