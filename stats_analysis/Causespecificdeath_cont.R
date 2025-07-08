"""
Deep-learning biomarker for Lung Health stats_analysis - cause specific diseases death COX analysis (using continous score)

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

###################################
# cause specific diseases death COX analysis function
####################################
plot_cox_meta_analysis_twosplit <- function(adjustments, subset_labels, df_list, pdf_file, ssl_variable = "lunghealth_cont", cont=FALSE, xlim=c(0.5, 2)) {
  
  # 1. Prepare the data
  #data_adjust <- df_pooled
  
  # Create a dataframe to hold the results
  pooled_results <- data.frame()
  
  # 2. Fit the Cox model and extract necessary statistics for each adjustment
  for (subset_label in subset_labels) {
    if (subset_label %in% names(df_list)) {
      current_data <- df_list[[subset_label]]
    } else {
      next
    }
    
    if (cont == TRUE) {
      current_data[[ssl_variable]] <- as.numeric(current_data[[ssl_variable]])
    } else {
      current_data[[ssl_variable]] <- as.factor(current_data[[ssl_variable]])
    }
    
    for (adj_name in names(adjustments)) {
      formula <- adjustments[[adj_name]]
      
      # Only consider rows without missing values for the current adjustment
      complete_data <- na.omit(current_data[, all.vars(formula)])
      num_cases <- nrow(complete_data)
      
      pooled_cox_model <- coxph(formula, data = complete_data)
      pooled_cox_summary <- summary(pooled_cox_model)
      
      coef_Thy <- pooled_cox_summary$coefficients[ssl_variable,"coef"]
      
      coef_Thy <- pooled_cox_summary$coefficients[ssl_variable,"coef"]
      se_coef_Thy <- pooled_cox_summary$coefficients[ssl_variable,"se(coef)"]
      expcoef_Thy <- exp(coef_Thy)
      lower_expcoef_Thy <- exp(coef_Thy - 1.96 * se_coef_Thy)
      upper_expcoef_Thy <- exp(coef_Thy + 1.96 * se_coef_Thy)
      p_Thy <- pooled_cox_summary$coefficients[ssl_variable,"Pr(>|z|)"]
      
      
      temp_results <- data.frame(
        Cohort = paste0(adj_name, " Diseases "),
        HR = expcoef_Thy,
        Lower = lower_expcoef_Thy,
        Upper = upper_expcoef_Thy,
        logHR = coef_Thy,
        SE = se_coef_Thy,
        p = p_Thy,
        No. = num_cases
      )
      
      temp_results$Cohort_N_P <- ifelse(
        temp_results$p < 0.001, 
        paste0(temp_results$Cohort, " (p < 0.001)"), 
        paste0(temp_results$Cohort, " (p = ", round(temp_results$p, 3), ")")
      )
      
      #temp_results$Cohort_N_P <- paste0(temp_results$Cohort, " (p = ", round(temp_results$p, 3), ")")
      pooled_results <- rbind(pooled_results, temp_results)
    }
  }
  
  ### # Reciproke the HR
  ### # 1. Recalculate Hazard Ratios and Confidence Intervals
  ### pooled_results$HR <- round(1 / pooled_results$HR, 3)
  ### pooled_results$Lower <- round(1 / pooled_results$Upper, 3)
  ### pooled_results$Upper <- round(1 / pooled_results$Lower, 3)
  
  # 2. Perform Meta-Analysis with Updated Data
  pooled_meta_analysis <- metagen(
    TE = log(pooled_results$HR),  
    seTE = pooled_results$SE,  
    studlab = pooled_results$Cohort_N_P,  
    pval = pooled_results$p,  
    sm = "HR",
    n.e = pooled_results$No.,
    common = F,
    random= F
  )
  
  # 3. Plot the Forest Plot
  pdf_file <- paste0(pdf_file)
  pdf(pdf_file, width = 14, height = 5)
  xlab_text <- "     <- Decrease risk | Increase risk ->"
  forest(pooled_meta_analysis, xlab = xlab_text, leftcols = c("studlab", "n.e"), leftlabs=c("Continously decreasing lung health adjusted", "No."), xlim = xlim, weight.study = "same", col.square = "grey", col.inside="black")
  grid.text("Cox Prop Hazards", 0.15, .87, gp=gpar(cex=2))
  dev.off()
}

####################################################


# Example usage for oevrall survival in NLST
# read data
df_NLST <- read.csv('/mnt/data/NLST_test.csv')

# cause specific diseases death COX NLST using continous score - Extended Figure 3c
df_NLST$lunghealth_cont_scaled <- scale(df_NLST$lunghealth_cont, center = TRUE, scale = TRUE)

adjustments <- list(
  Neoplasm = as.formula(Surv(ICD_OSYEARS_12y, NEOPLASM_DEATH_12y) ~ lunghealth_cont_scaled + cigsmok + pkyr_scaled + BMI + sct_long_dia_min6 + diagstro + diaghear + diaghype + canclung + diagdiab + diagadas + diagcopd + diagemph + diagfibr + diagpneu + diagchas + diagasbe + diagbron + diagchro + diagsarc + diagsili + diagtube + strata(FEMALE, age_binned_NLST)),
  Metabolic = as.formula(Surv(ICD_OSYEARS_12y, METABOLIC_DEATH_12y) ~ lunghealth_cont_scaled + cigsmok + pkyr_scaled + BMI + sct_long_dia_min6 + diagstro + diaghear + diaghype + canclung + diagdiab + diagadas + diagcopd + diagemph + diagfibr + diagpneu + diagchas + diagasbe + diagbron + diagchro + diagsarc + diagsili + diagtube + strata(FEMALE, age_binned_NLST)),
  Digestive = as.formula(Surv(ICD_OSYEARS_12y, DIGESTIVE_DEATH_12y) ~ lunghealth_cont_scaled + cigsmok + pkyr_scaled + BMI + sct_long_dia_min6 + diagstro + diaghear + diaghype + canclung + diagdiab + diagadas + diagcopd + diagemph + diagfibr + diagpneu + diagchas + diagasbe + diagbron + diagchro + diagsarc + diagsili + diagtube + strata(FEMALE, age_binned_NLST))
)

subset_labels <- c("NLST")
df_list <- list("NLST" = df_NLST)

plot_cox_meta_analysis_twosplit(adjustments, subset_labels, df_list, "mnt/outcome_analysis/Figures_LH/cause-specific_diseases_NLST_cont_adjusted.pdf", "lunghealth_cont_scaled", cont=TRUE, xlim = c(0.5, 2.5))
