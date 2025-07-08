"""
Deep-learning biomarker for Lung Health stats_analysis - Odds ratio analysis

# The code and data of this repository are intended to promote transparent and reproducible research
# of the paper -- Deep Learning-based Lung Health Quantification on Computed Tomography in Adults

# THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


"""
Example usage at the end of this file
"""


# load libraries
library(meta)
require(survival)
library(survminer) 
library(dplyr)
library(grid)
library(gridExtra)
library(meta)
library(naniar)
library(ggplot2)
library(scales)
library(ggpubr)
library(metafor)
library(grid)
############################
# Odds ratio function
#############################
plot_logistic_meta_analysis_flexible_subgroups_predictors_FlexibleCategories <- function(adjustments, subset_labels, pdf_file, binarized_predictors, subgroup_data = NULL, reference = TRUE, subgroup = TRUE, xlim = c(0.1, 1.5)) {
  if (is.null(subgroup_data)) {
    stop("Please provide the 'subgroup_data' list with dataframes for each adjustment within each subgroup")
  }
  
  if (!is.list(binarized_predictors) || length(binarized_predictors) != length(adjustments)) {
    stop("The 'binarized_predictors' should be a list with the same length as 'adjustments'")
  }
  
  pooled_results <- data.frame()
  
  for (subset_label in subset_labels) {
    for (adj_name in names(adjustments)) {
      if (!subset_label %in% names(subgroup_data) || !adj_name %in% names(subgroup_data[[subset_label]])) {
        next
      }
      
      predictor_var <- binarized_predictors[[adj_name]]
      current_data <- subgroup_data[[subset_label]][[adj_name]]
      
      # Ensure predictor variable is treated as factor or numeric as appropriate
      if (!is.numeric(current_data[[predictor_var]])) {
        current_data[[predictor_var]] <- as.factor(current_data[[predictor_var]])
      }
      
      formula <- adjustments[[adj_name]]
      complete_data <- na.omit(current_data[, all.vars(formula)])
      N_complete_data <- nrow(complete_data)
      
      N_low <- sum(complete_data[[predictor_var]] == "0")
      
      # Add reference category for the current subset and adjustment, conditional on the reference argument
      if (reference) {
        reference_row <- data.frame(
          Cohort = paste0(subset_label, " - ", adj_name, " - REFERENCE"),
          OR = 1,
          Lower = 0,
          Upper = 0,
          logOR = 0,
          SE = 0.0001,
          p = 1,
          Cohort_N_P = paste0(subset_label, " - ", adj_name, " - REFERENCE"),
          N_Cases = N_low,
          Subgroup = paste0(subset_label)
        )
        pooled_results <- rbind(pooled_results, reference_row)
      }
      
      pooled_logistic_model <- glm(formula, data = complete_data, family = binomial)
      pooled_logistic_summary <- summary(pooled_logistic_model)
      print(pooled_logistic_summary)
      
      # Handle different predictor categories
      if (is.factor(complete_data[[predictor_var]])) {
        levels_predictor <- levels(complete_data[[predictor_var]])
        for (level in levels_predictor[-1]) { # Exclude the reference level
          num_cases <- sum(complete_data[[predictor_var]] == level)
          comparison_label <- paste0(subset_label, " - ", adj_name, " - ", level)
          
          coef_Thy <- pooled_logistic_summary$coefficients[paste0(predictor_var, level), "Estimate"]
          se_coef_Thy <- pooled_logistic_summary$coefficients[paste0(predictor_var, level), "Std. Error"]
          expcoef_Thy <- exp(coef_Thy)
          lower_expcoef_Thy <- exp(coef_Thy - 1.96 * se_coef_Thy)
          upper_expcoef_Thy <- exp(coef_Thy + 1.96 * se_coef_Thy)
          p_Thy <- pooled_logistic_summary$coefficients[paste0(predictor_var, level), "Pr(>|z|)"]
          
          temp_results <- data.frame(
            Cohort = comparison_label,
            OR = expcoef_Thy,
            Lower = lower_expcoef_Thy,
            Upper = upper_expcoef_Thy,
            logOR = coef_Thy,
            SE = se_coef_Thy,
            p = p_Thy,
            N_Cases = num_cases,
            Subgroup = paste0(subset_label)
          )
          temp_results$Cohort_N_P <- ifelse(
            temp_results$p < 0.001, 
            paste0(temp_results$Cohort, " (p < 0.001)"), 
            paste0(temp_results$Cohort, " (p = ", round(temp_results$p, 3), ")")
          )
          pooled_results <- rbind(pooled_results, temp_results)
        }
      } else {
        # Continuous or binary predictor
        coef_Thy <- pooled_logistic_summary$coefficients[predictor_var, "Estimate"]
        se_coef_Thy <- pooled_logistic_summary$coefficients[predictor_var, "Std. Error"]
        expcoef_Thy <- exp(coef_Thy)
        lower_expcoef_Thy <- exp(coef_Thy - 1.96 * se_coef_Thy)
        upper_expcoef_Thy <- exp(coef_Thy + 1.96 * se_coef_Thy)
        p_Thy <- pooled_logistic_summary$coefficients[predictor_var, "Pr(>|z|)"]
        
        temp_results <- data.frame(
          Cohort = paste0(subset_label, " - ", adj_name),
          OR = expcoef_Thy,
          Lower = lower_expcoef_Thy,
          Upper = upper_expcoef_Thy,
          logOR = coef_Thy,
          SE = se_coef_Thy,
          p = p_Thy,
          N_Cases = N_complete_data,
          Subgroup = paste0(subset_label)
        )
        temp_results$Cohort_N_P <- ifelse(
          temp_results$p < 0.001, 
          paste0(temp_results$Cohort, " (p < 0.001)"), 
          paste0(temp_results$Cohort, " (p = ", round(temp_results$p, 3), ")")
        )
        pooled_results <- rbind(pooled_results, temp_results)
      }
    }
  }
  
  pooled_results$Subgroup <- as.factor(pooled_results$Subgroup)
  
  if (subgroup) {
    pooled_meta_analysis <- metagen(
      TE = log(pooled_results$OR),
      seTE = pooled_results$SE,
      studlab = pooled_results$Cohort_N_P,
      pval = pooled_results$p,
      sm = "OR",
      common = FALSE,
      random = FALSE,
      n.e = pooled_results$N_Cases,
      byvar = pooled_results$Subgroup
    )
  } else {
    pooled_meta_analysis <- metagen(
      TE = log(pooled_results$OR),
      seTE = pooled_results$SE,
      studlab = pooled_results$Cohort_N_P,
      pval = pooled_results$p,
      sm = "OR",
      common = FALSE,
      random = FALSE,
      n.e = pooled_results$N_Cases
      # Notice the removal of the 'byvar' parameter here
    )
  }
  
  pdf(pdf_file, width = 14, height = 10)
  xlab_text <- "Odds Ratio (95% CI)"
  
  forest(
    pooled_meta_analysis,
    leftcols = c("studlab", "n.e"),
    leftlabs = c("Subgroups", "No. Cases"),
    xlab = xlab_text,
    xlim = xlim,
    atransf = exp,
    col.square = "grey",
    col.inside = "black",
    weight.study = "same",
    byvar = pooled_results$Subgroup
  )
  grid.text("Logistic Regression Analysis", 0.15, .87, gp=gpar(cex=2))
  dev.off()
}

# Example usage for oevrall survival in NLST
# read data
df_NLST <- read.csv('/mnt/data/NLST_test.csv')

# odds ratio metastaic cancer NLST - Figure 3f
# keep only participants who developed cancer
df_NLST_cancer <- df_NLST[!is.na(df_NLST$STAGE_METASTATIC), ]

adjustments <- list(
  LungCa_metastatic = as.formula(STAGE_METASTATIC ~ lunghealth_cat + cigsmok + pkyr + BMI + sct_long_dia + diagstro + diaghear + diaghype + canclung + diagdiab + diagadas + diagcopd + diagemph + diagfibr + diagpneu + diagchas + diagasbe + diagbron + diagchro + diagsarc + diagsili + diagtube + strata(FEMALE, age_binned_NLST))
)

subset_labels <- c("NLST")
binarized_predictors <- list(
  LungCa_metastatic = "lunghealth_cat"
)

subgroup_data <- list(
  NLST = list(
    LungCa_metastatic = df_NLST_cancer

  )
)

pdf_file <- "mnt/outcome_analysis/Figures_LH/oddsratio_METASTIC_inparticipantswhodevelopcancer.pdf"
plot_logistic_meta_analysis_flexible_subgroups_predictors_FlexibleCategories(adjustments, subset_labels, pdf_file, binarized_predictors, subgroup_data, reference=TRUE, xlim = c(0.5, 15))
