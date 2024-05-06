"""
  Deep-learning biomarker for Lung Health stats_analysis - Plotting KM in NLST
"""
.libPaths()
# Get current library paths
current_lib_paths <- .libPaths()
# Add custom directory to library paths
new_lib_paths <- c("/home/ahmed/R/x86_64-pc-linux-gnu-library/4.1/languageserver/libs", current_lib_paths)
# Set the new library paths
.libPaths(new_lib_paths)



library("Matrix", lib.loc="/home/ahmed/R/x86_64-pc-linux-gnu-library/4.1/languageserver/libs")
library("survival", lib.loc="/home/ahmed/R/x86_64-pc-linux-gnu-library/4.1/languageserver/libs")
library(ggplot2)
library(lubridate) 
library(tidyr)
library(ggsurvfit)
library(gtsummary)
library("survminer", lib.loc="/home/ahmed/R/x86_64-pc-linux-gnu-library/4.1/languageserver/libs")


install.packages("fontconfig", lib="/home/ahmed/R/x86_64-pc-linux-gnu-library/4.1/languageserver/libs", dependencies=TRUE)


###########
# library(httpgd)
# install.packages("Matrix", repos = "http://R-Forge.R-project.org", lib="/home/ahmed/R/x86_64-pc-linux-gnu-library/4.1/languageserver/libs", dependencies=TRUE)
# install.packages("survminer", lib="/home/ahmed/R/x86_64-pc-linux-gnu-library/4.1/languageserver/libs", dependencies=TRUE)
# library("survminer", lib.loc="/home/ahmed/R/x86_64-pc-linux-gnu-library/4.1/languageserver/libs")
#################


# load data
data <- read.csv('/mnt/data6/DeepPY/src_main/stats_analysis/smokeAI_NLST_test.csv')

# AI lung health 5 risk groups
data$DeepPY <- factor(data$Deep_py)
data$DeepPY

# # Get overall survival
data$death[which(data$death>1)] = 0

# Get cancer death
data$ci[which(data$ci!='1')] = 0
data$ci[which(data$ci=='1')] = 1
data$ci = as.numeric(data$ci)


# KM OS AI lung health

fit <- survfit(Surv(fup, death) ~ DeepPY, data = data)
my_colors <- c("skyblue3", "blue3", "red1", "tomato", "darkred")

res <- ggsurvplot(fit, data = data, risk.table = TRUE,  conf.int = FALSE, censor = FALSE,
                  ylim = c(0.5, 1), xlim = c(0, 4760), xscale = "d_y", break.x.by = 365.3,
                  legend.labs = c("Very Low", "Low", "Moderate", "High", "Very High"),
                  legend.title = "Survival Analysis of AI lung health risk groups in NLST test cohort", legend = c(0.37, 0.2),
                  tables.theme = theme_cleantable(),
                  risk.table.height = 0.25, xlab = 'Time (Years)', ylab = 'Death Free Survival',
                  palette = my_colors) 

# Modify legend and plot aesthetics
res$plot <- res$plot + theme(legend.key.width = unit(8, "mm"),
                             legend.key.height = unit(5, "mm"),
                             legend.text = element_text(size = 12))

print(res)

