"""
AI-derived Lung Score stats_analysis - linear model association analysis

# The code and data of this repository are intended to promote transparent and reproducible research
# of the paper -- AI–based Radiographic Lung Score Associates with Clinical Outcomes in Adults: a model development and validation study

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

############################
# Linear model association function
#############################
plot_linear_meta_analysis_standardized_scale_metagen_subgroups <- function(adjustments, predictors, df_list, pdf_file, outcome_variable = "lunghealth_cont", xlim = c(-0.1, 0.1), strata_variable = NULL, scale_factor = 1, adjust_p = FALSE, p_adjust_method = "holm", subgroups = NULL) {
  
  # 1. Prepare the data
  pooled_results <- data.frame()
  
  if (is.null(subgroups)) {
    subgroups <- setNames(rep("Other", length(predictors)), predictors)
  }
  
  # 2. Fit the linear regression model and extract necessary statistics for each predictor
  for (subset_label in names(df_list)) {
    current_data <- df_list[[subset_label]]
    
    for (predictor in predictors) {
      # Convert the predictor to numeric
      current_data[[predictor]] <- as.numeric(current_data[[predictor]])
      
      # Standardize the predictor
      current_data[[predictor]] <- scale(current_data[[predictor]], center = TRUE, scale = TRUE)
      
      formula_part <- paste(outcome_variable, "~", predictor, adjustments, sep = " ")
      if (!is.null(strata_variable)) {
        formula_part <- paste(formula_part, "+ strata(", strata_variable, ")", sep = "")
      }
      formula <- as.formula(formula_part)
      
      # Only consider rows without missing values for the current adjustment and predictor
      complete_data <- na.omit(current_data[, all.vars(formula)])
      
      linear_model <- lm(formula, data = complete_data)
      linear_summary <- summary(linear_model)
      
      coef_val <- linear_summary$coefficients[predictor, "Estimate"] * scale_factor
      se_coef_val <- linear_summary$coefficients[predictor, "Std. Error"] * scale_factor
      p_val <- 2 * (1 - pt(abs(coef_val / se_coef_val), linear_summary$df[2]))
      
      temp_results <- data.frame(
        Cohort = paste0(subset_label, " - ", predictor, " (p", ifelse(p_val < 0.001, "<0.001", sprintf("=%.3f", p_val)), ")"),
        N = nrow(complete_data),
        Beta = coef_val,
        SE = se_coef_val,
        p = p_val,
        Subgroup = subgroups[predictor]
      )
      
      pooled_results <- rbind(pooled_results, temp_results)
    }
  }
  
  # 3. Adjust p-values if specified
  if (adjust_p) {
    pooled_results$p <- p.adjust(pooled_results$p, method = p_adjust_method)
  }
  
  # 4. Perform Meta-Analysis with Updated Data
  pooled_results$Subgroup <- as.factor(pooled_results$Subgroup)
  pooled_meta_analysis <- metagen(
    TE = pooled_results$Beta,  
    seTE = pooled_results$SE,  
    studlab = pooled_results$Cohort,  
    sm = "MD",  # Mean difference
    common = FALSE,  # Fixed-effects model
    random = FALSE,  # Do not use random-effects model
    n.e = pooled_results$N,
    n.c = pooled_results$N,
    byvar = pooled_results$Subgroup
  )
  
  # 5. Plot the Forest Plot
  pdf(pdf_file, width = 12, height = 12)
  xlab_text <- "<- Worse Lung Health | Better Lung Health ->"
  forest(pooled_meta_analysis, xlim = xlim, xlab = xlab_text, leftcols = c("studlab", "n.e"), leftlabs = c("Variables\n and Subgroups", "Count"), rightcols = c("effect", "ci"), print.byvar = FALSE, subgroup = TRUE, digits = 2,   weight.study = "same",
         col.square = "grey",
         col.inside = "black")
  dev.off()
}



# Example usage for oevrall survival in NLST
# read data
df_NLST <- read.csv('/mnt/data/NLST_test.csv')

# linear model association NLST - Figure 5f
# adjustments in addition to gender and age 
adjustments <- "+ cigsmok"
# predictors 
predictors <- c(
  "Scaled_BMI",
  "Scaled_Pack-years", "Scaled_Smokeday", "Scaled_Smokeyear", 
  "Scaled_acrin_drinknum_curr_scaled", "Scaled_acrin_drinknum_form_scaled", "Scaled_lss_alcohol_num_scaled"
)

df_list <- list("NLST" = df_NLST)

subgroups <- c(
  "Scaled_BMI"="Body Composition", 
  "Scaled_Pack-years"="Smoking", "Scaled_Smokeday"="Smoking", "Scaled_Smokeyear"="Smoking",
  "Scaled_acrin_drinknum_curr_scaled"="Alcohol", "Scaled_acrin_drinknum_form_scaled"="Alcohol", 
  "Scaled_lss_alcohol_num_scaled"="Alcohol"
)

plot_linear_meta_analysis_standardized_scale_metagen_subgroups(
  adjustments, 
  predictors, 
  df_list, 
  "mnt/outcome_analysis/Figures_LH//Meta_Sugrouped_LinReg_NLST_scaled.pdf",
  "lunghealth_cont", 
  strata_variable = "FEMALE, age_binned_NLST",
  scale_factor = 1,
  adjust_p = FALSE, 
  p_adjust_method = "BH", 
  subgroups = subgroups,
  xlim = c(-0.06, 0.04)
)

