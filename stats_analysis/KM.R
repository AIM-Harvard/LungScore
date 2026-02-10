"""
AI-derived Lung Score stats_analysis - KM analysis

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
# KM Plot function
#############################
plot_KM_unzoomed_fivesplit <- function(df, time_col, event_col, group_col, title, xlab, ylab, pdf_file, cohort_name, xlim, ylim = c(0, 1), YEAR = 12) {
  
  # Prepare the data frame for plotting
  df_for_plot <- df
  df_for_plot$surv_obj <- with(df, Surv(time = df[[time_col]], event = df[[event_col]]))
  df_for_plot$group <- df[[group_col]]
  print(df_for_plot$group)

  # Fit survival curves
  fit_surv <- survfit(surv_obj ~ group, data = df_for_plot)
  
  
  ### SURVIVAL AT specific YEAR USING KAPLAN MEIER ESTIMATOR
  surv_prob <- summary(fit_surv, times = YEAR)$surv
  # extract mortality probabilities at 5 years
  mortality_prob <- 1 - surv_prob
  
  
  
  # Calculate the total number of included patients
  total_patients <- nrow(df)
  
  # Cox Proportional Hazards Model to get the HR and 95% CI
  cox_model <- coxph(surv_obj ~ group, data = df_for_plot)
  print(summary(cox_model))
  hr <- exp(coef(cox_model))
  ci <- exp(confint(cox_model))

  
  pval <- summary(cox_model)$coefficients[,5]
  
  # Reciprocal of the Hazard Ratios and Confidence Intervals
  hr_group1 <- hr[1]
  hr_group2 <- hr[2]
  hr_group3 <- hr[3]
  hr_group4 <- hr[4]
 
  
  # for 4 risk groups
  ci_group1 <- c(ci[1], ci[4]) 
  ci_group2 <- c(ci[2], ci[5])
  ci_group3 <- c(ci[3], ci[6])
  ci_group4 <- c(ci[4], ci[7])
  
  # Formatting p-values
  p_text_group1 <- ifelse(pval[1] < 0.001, "<0.001", sprintf("=%.3f", pval[1]))
  p_text_group2 <- ifelse(pval[2] < 0.001, "<0.001", sprintf("=%.3f", pval[2]))
  p_text_group3 <- ifelse(pval[3] < 0.001, "<0.001", sprintf("=%.3f", pval[3]))
  p_text_group4 <- ifelse(pval[4] < 0.001, "<0.001", sprintf("=%.3f", pval[4]))
  
  
  # Creating summary text REVERSED
  hr_text_group1 <- sprintf("%.2f (%.2f-%.2f)", hr_group1, ci_group1[1], ci_group1[2])
  hr_text_group2 <- sprintf("%.2f (%.2f-%.2f)", hr_group2, ci_group2[1], ci_group2[2])
  hr_text_group3 <- sprintf("%.2f (%.2f-%.2f)", hr_group3, ci_group3[1], ci_group3[2])
  hr_text_group4 <- sprintf("%.2f (%.2f-%.2f)", hr_group4, ci_group4[1], ci_group4[2])
  
  
  # Generate the plot
  g <- plot_survival_unzoomed(time_col, event_col, fit_surv, df_for_plot, cohort_name, hr_text_group1, hr_text_group2, hr_text_group3, hr_text_group4, total_patients, title, xlab, ylab, group_col, xlim, ylim, surv_prob_1, surv_prob_2, surv_prob_3, surv_prob_4, surv_prob_5, mortality_prob_1, mortality_prob_2, mortality_prob_3, mortality_prob_4, mortality_prob_5, YEAR)
  
  # Save plot as PDF without blank page
  pdf(pdf_file, width = 6, height = 5, onefile = TRUE)
  print(g) 
  dev.off()
  
  # Perform log-rank test using the original group column
  surv_diff <- survdiff(surv_obj ~ group, data = df_for_plot)
  print(surv_diff)
}

plot_survival_unzoomed <- function(time_col, event_col, fit, data, cohort_name, hr_text_group1, hr_text_group2, hr_text_group3, hr_text_group4, total_patients, title, xlab, ylab, group_col, xlim, ylim, surv_prob_1, surv_prob_2, surv_prob_3, surv_prob_4, surv_prob_5, mortality_prob_1, mortality_prob_2, mortality_prob_3, mortality_prob_4, mortality_prob_5, YEAR) {
  
  # calculate degree of freedom
  # KM are always unadjusted here
  # change function (NLST or FHS), data, time, and event accordingly
  p_value_catdf_reduced <- cal_4df_pvalue_unadj(
    data = data,  # Your dataset
    time_col = time_col,  # Time-to-event column
    event_col = event_col,  # Event column
    group_col = "lunghealth_cat"  # Grouping variable
  )
  
  p_value_catdf_reduced <- ifelse(p_value_catdf_reduced < 0.001, "< 0.001", sprintf("= %.3f", p_value_catdf_reduced))
  p_value_catdf_reduced <- paste("p ", p_value_catdf_reduced)
  
  
  g <- ggsurvplot(
    fit,
    data = data,
    risk.table = TRUE,
    #risk.events = TRUE,
    risk.table.title = "At risk",
    risk.table.height = 0.2,
    fontsize = 3.5,
    conf.int = FALSE,
    legend = c(0.9, 0.2),
    legend.title = "Lung Health",
    
    legend.labs = c("very high", "high", "moderate", "low", "very low"),
    palette = c("#1b7837","#5aae61", "#a6dba0", "#c2a5cf", "#762a83"),
    
    
    title = paste(title, cohort_name),
    xlab = xlab,
    ylab = ylab,
    tables.theme = theme_cleantable() + theme(plot.title = element_text(size = 12)),
    censor = FALSE,
    xlim = xlim,
    ylim = c(0, 1)
  )
  
  g$plot <- g$plot + 
    scale_x_continuous(expand = c(0,0)) +
    scale_y_continuous(expand = c(0,0), labels = scales::percent_format(scale = 100)) +
    # change ylim
    scale_y_continuous(limits = ylim, expand = c(0,0), labels = scales::percent_format(scale = 100)) +
    theme(legend.position = "none") +
    
    # very high
    annotate("text", x = 0.1, y = (0.25 * (1-ylim[[1]]) + ylim[[1]]), label = paste("high:"), hjust = "left", size = 3.5) + 
    annotate("text", x = 2, y = (0.25 * (1-ylim[[1]]) + ylim[[1]]), label = paste("Ref"), hjust = "left", size = 3.5)

    # high group
    annotate("text", x = 0.1, y = (0.2  * (1-ylim[[1]]) + ylim[[1]]), label = paste("high:"), hjust = "left", size = 3.5) +
    annotate("text", x = 2, y = (0.2  * (1-ylim[[1]]) + ylim[[1]]), label = hr_text_group1, hjust = "left", size = 3.5) +
    
    # moderate group
    annotate("text", x = 0.1, y = (0.15  * (1-ylim[[1]]) + ylim[[1]]), label = paste("moderate:"), hjust = "left", size = 3.5) +
    annotate("text", x = 2, y = (0.15 * (1-ylim[[1]]) + ylim[[1]]), label = hr_text_group2, hjust = "left", size = 3.5) +
    
    # low group
    annotate("text", x = 0.1, y = (0.1  * (1-ylim[[1]]) + ylim[[1]]), label = paste("low:"), hjust = "left", size = 3.5) +
    annotate("text", x = 2, y = (0.1  * (1-ylim[[1]]) + ylim[[1]]), label = hr_text_group3, hjust = "left", size = 3.5) +
    
    # very low group
    annotate("text", x = 0.1, y = (0.05  * (1-ylim[[1]]) + ylim[[1]]), label = paste("very low:"), hjust = "left", size = 3.5) +
    annotate("text", x = 2, y = (0.05 * (1-ylim[[1]]) + ylim[[1]]), label = hr_text_group4, hjust = "left", size = 3.5) +
    
    # 4 df p-value
    annotate("text", x = 5.2, y = (0.20 * (1-ylim[[1]]) + ylim[[1]]), label = p_value_catdf_reduced, hjust = "left", size = 3.5) +
    
    
    # no of patients and HZ 95 ci 
    annotate("text", x = 0.1, y = (0.35 * (1-ylim[[1]]) + ylim[[1]]), label = paste("n=", total_patients), hjust = "left", size = 3.5) + 
    annotate("text", x = 2, y = (0.3  * (1-ylim[[1]]) + ylim[[1]]), label = paste("HR (95% CI)"), hjust = "left", size = 3.5) +
    
  
  return(g)
}


# Define a function to calculate 4 degree of freedom (4df) p-values in FHS or NLST unadjusted
cal_4df_pvalue_unadj <- function(data, time_col, event_col, group_col) {
  
  # Fit the full Cox model including the group variable (no adjustment)
  formula <- as.formula(paste("Surv(", time_col, ",", event_col, ") ~", group_col))
  COX_model_full <- coxph(formula, data = data)
  
  # Fit the reduced model excluding the group variable (intercept-only model)
  formula_reduced <- as.formula(paste("Surv(", time_col, ",", event_col, ") ~ 1"))
  COX_model_reduced <- coxph(formula_reduced, data = data)
  
  # Perform a likelihood ratio test between the full and reduced models
  LRT <- anova(COX_model_full, COX_model_reduced, test = "LRT")
  
  # Extract the 4df p-value
  p_value <- LRT[2, 4]  
  
  return(p_value)
}


####################################################

# Example usage for oevrall survival in NLST
# read data
df_NLST <- read.csv('/mnt/data/NLST_test.csv')

# OS NLST - Figure 2b
plot_KM_unzoomed_fivesplit(
  df = df_NLST, 
  time_col = "OSYEARS_12y", 
  event_col = "DEATH_12y", 
  group_col = "lunghealth_cat",  # 5 risk groups 
  title = "Overall Survival in", 
  xlab = "Years since enrollment", 
  ylab = "Percentage Surviving", 
  pdf_file = "mnt/outcome_analysis/Figures_LH/OS_KM_NLST.pdf", 
  cohort_name = "NLST",
  xlim = c(0, 12), 
  ylim = c(0.4,1)
)
