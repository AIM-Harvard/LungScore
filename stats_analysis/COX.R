"""
AI-derived Lung Score stats_analysis - COX analysis

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
# COX analysis adjusted function
#############################
plot_cox_meta_analysis_fivesplit_reference <- function(time_col, event_col, adjustments, subset_labels, df_list, pdf_file, lunghealth_variable, xlim = c(0.5, 1.5)) {
  # 1. Prepare the data
  pooled_results <- data.frame()
  
  # 2. Fit the Cox model and extract necessary statistics for each adjustment and subset
  for (subset_label in subset_labels) {
    if (subset_label %in% names(df_list)) {
      current_data <- df_list[[subset_label]]
    } else {
      next
    }
    
    # lunghealth_cat
    current_data[[lungage_variable]] <- as.factor(current_data[[lungage_variable]])

    for (adj_name in names(adjustments)) {
      formula <- adjustments[[adj_name]]
      complete_data <- na.omit(current_data[, all.vars(formula)])
      
      # Calculate N_low for the current subset and adjustment
      N_low <- sum(complete_data[[lungage_variable]] == "0")
      
      # Add reference category (lungage_cat = 0) for the current subset and adjustment
      reference_row <- data.frame(
        Cohort = "REFERENCE",
        HR = 1,
        Lower = 0,
        Upper = 0,
        logHR = 1,
        SE = 0.0001,
        p = 1,
        Cohort_N_P = "very high",            
        N_Cases = N_low
      )
      pooled_results <- rbind(pooled_results, reference_row)
      
      # Cox model fitting and result extraction
      pooled_cox_model <- coxph(formula, data = complete_data)
      pooled_cox_summary <- summary(pooled_cox_model)
      print(pooled_cox_summary)
      
      # risk groups labels
      label_names <- c("high", "moderate", "low", "very low")
      
      
      
      for (i in 1:4) {
        lungage_val <- paste0(lungage_variable, i)
        # Number of cases for the current comparison
        num_cases <- sum(complete_data[[lungage_variable]] == as.character(i))
        comparison_label <- label_names[i]  # Use the label from the predefined list
        
        
        # Extracting necessary statistics
        coef_Thy <- pooled_cox_summary$coefficients[lungage_val,"coef"]
        se_coef_Thy <- pooled_cox_summary$coefficients[lungage_val,"se(coef)"]
        expcoef_Thy <- exp(coef_Thy)
        lower_expcoef_Thy <- exp(coef_Thy - 1.96 * se_coef_Thy)
        upper_expcoef_Thy <- exp(coef_Thy + 1.96 * se_coef_Thy)
        p_Thy <- pooled_cox_summary$coefficients[lungage_val,"Pr(>|z|)"]
        
        # Adding results to the pooled results
        temp_results <- data.frame(
          Cohort = comparison_label,
          HR = expcoef_Thy,
          Lower = lower_expcoef_Thy,
          Upper = upper_expcoef_Thy,
          logHR = coef_Thy,
          SE = se_coef_Thy,
          p = p_Thy,
          N_Cases = num_cases
        )
        
        
        temp_results$Cohort_N_P <- temp_results$Cohort
        pooled_results <- rbind(pooled_results, temp_results)
      }
    }
  }
  
  # 3. Perform Meta-Analysis with Updated Data
  pooled_meta_analysis <- metagen(
    TE = log(pooled_results$HR),  
    seTE = pooled_results$SE,  
    studlab = pooled_results$Cohort_N_P,  
    pval = pooled_results$p,  
    sm = "HR",
    common = F,
    random= F,
    n.e = pooled_results$N_Cases
  )
  
  # 4. Plot the Forest Plot
  pdf_file <- paste0(pdf_file)
  pdf(pdf_file, width = 14, height = 10)
  xlab_text <- ""
  forest(
    pooled_meta_analysis, 
    leftcols = c("studlab", "n.e"),
    leftlabs = c("Lung Health", "No."),
    xlab = xlab_text, 
    xlim = xlim, 
    atransf=exp, weight.study = "same", col.square = "grey", col.inside="black"
  )

  
  # calculate degree of freedom
  # change function data name  "NLST OR FHS"
  p_value_catdf_reduced <- cal_4df_pvalue_NLST_adj(
    data = current_data,  # Your dataset
    time_col = time_col,  # Time-to-event column
    event_col = event_col,  # Event column
    group_col = "lunghealth_cate"  # Grouping variable
  )

  p_value_catdf_reduced <- ifelse(p_value_catdf_reduced < 0.001, "p < 0.001", sprintf("p = %.3f", p_value_catdf_reduced))
  
  
  # Add 4df p-value annotation just below the CI interval based on anova test
  grid.text(paste(p_value_catdf_reduced), 
            x = 0.647, y = 0.41, gp=gpar(cex=1.1))  # Adjust the position and font size as needed
  
  dev.off()
}


# Define a function to calculate 4df p-values in NLST adjusted
cal_4df_pvalue_NLST_adj <- function(data, time_col, event_col, group_col) {
  
  # Fit the full Cox model including the group variable
  formula <- as.formula(paste("Surv(", time_col, ",", event_col, ") ~", 
                               group_col, "+ pkyr_scaled + cigsmok + BMI + sct_long_dia_min6 + diagstro + diaghear + diaghype + canclung + diagdiab + diagadas + diagcopd + diagemph + diagfibr + diagpneu + diagchas + diagasbe + diagbron + diagchro + diagsarc + diagsili + diagtube + strata(age_binned_NLST, FEMALE)"))
  COX_model_full <- coxph(formula, data = data)
  
  # Fit the reduced model excluding the group variable
  formula_reduced <- as.formula(paste("Surv(", time_col, ",", event_col, ") ~ pkyr_scaled + cigsmok + BMI + sct_long_dia_min6 + diagstro + diaghear + diaghype + canclung + diagdiab + diagadas + diagcopd + diagemph + diagfibr + diagpneu + diagchas + diagasbe + diagbron + diagchro + diagsarc + diagsili + diagtube   + strata(age_binned_NLST, FEMALE)"))
  COX_model_reduced <- coxph(formula_reduced, data = data)
  
  # Perform a likelihood ratio test between the full and reduced models
  LRT <- anova(COX_model_full, COX_model_reduced, test = "LRT")
  
  # Extract the 2df p-value
  p_value <- LRT[2, 4]  
  
  return(p_value)
}


####################################################


# Example usage for oevrall survival in NLST
# read data
df_NLST <- read.csv('/mnt/data/NLST_test.csv')

# OS NLST  COX - Figure 2d
adjustments <- list(
  Strata_Sex_Age_Adj_Smok = as.formula(Surv(OSYEARS_12y, DEATH_12y) ~ lunghealth_cat + cigsmok + pkyr + BMI + sct_long_dia_min6 + diagstro + diaghear + diaghype + canclung + diagdiab + diagadas + diagcopd + diagemph + diagfibr + diagpneu + diagchas + diagasbe + diagbron + diagchro + diagsarc + diagsili + diagtube + strata(age_binned_NLST, FEMALE))
)
subset_labels <- c("NLST")                                                 
df_list <- list("NLST" = df_NLST)

time_col <- "OSYEARS_12y"
event_col <- "DEATH_12y"

plot_cox_meta_analysis_fivesplit_reference(time_col, event_col, adjustments, subset_labels, df_list, "mnt/outcome_analysis/Figures_LH/OS_COX_NLST.pdf", "lunghealth_cat", xlim = c(0.5, 5))

