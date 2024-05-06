"""
  Deep-learning biomarker for Lung Health stats_analysis - Cox analysis in NLST
"""
library(survival)
library(lubridate) 
library(ggplot2)
library(ggsurvfit)
library(gtsummary)

data <- read.csv('/mnt/data6/DeepPY/src_main/stats_analysis/smokeAI_NLST_test.csv')

# AI lung health 5 risk groups
data$DeepPY <- factor(data$Deep_py)
data$DeepPY

# Get Overall survival events
data$death[which(data$death>1)] <- 0

# Get lung Cancer death events
data$ci[which(data$ci!='1')] = 0
data$ci[which(data$ci=='1')] = 1
data$ci <- as.numeric(data$ci)

# Nodules present =1 , else=0
data$sct_long_dia_min6 <- ifelse(is.na(data$nodsize), 0, 1)

# concordance index
res.cox <- coxph(Surv(fup, death) ~ lungdamage, data = data)
print(res.cox$concordance)

##########################################

# COX AI lung health overall survival unadjusted
res.cox <- coxph(Surv(fup, death) ~ DeepPY, data = data)
print("COX Overall Survival unadjusted AI Lung Health")
print(res.cox)

# COX AI lung health overall survival adjusted for age, gender, pack years and nodules
res.cox <- coxph(Surv(fup, death) ~ DeepPY + gen + age + packyear + sct_long_dia_min6, data = data)
print("COX Overall Survival adjusted specifics AI Lung Health")
print(res.cox)

# COX AI lung health overall survival adjusted for everything
res.cox <- coxph(Surv(fup, death) ~ DeepPY + gen + age + sct_long_dia_min6 + packyear +  cigsmok + stro_hear_hype_canc + bmi + copd + diab  + fibr + pneu + chro + bron + adass + asbe + emph + sili + sarc + tube, data = data)
print("COX Overall Survival adjusted everything AI Lung Health")
print(res.cox)

# COX AI lung health lung cancer death adjusted for everything
res.cox <- coxph(Surv(fup, ci) ~ DeepPY + gen + age + sct_long_dia_min6 + packyear +  cigsmok + stro_hear_hype_canc + bmi + copd + diab  + fibr + pneu + chro + bron + adass + asbe + emph + sili + sarc + tube, data = data)
print("COX Lung Cancer Death AI Lung Health")
print(res.cox)

# COX AI lung health cancer incidence adjusted for everything
res.cox <- coxph(Surv(candx_fup_days, candx_fup_stat) ~ DeepPY + gen + age + sct_long_dia_min6 + packyear +  cigsmok + stro_hear_hype_canc + bmi + copd + diab  + fibr + pneu + chro + bron + adass + asbe + emph + sili + sarc + tube, data = data)
print("COX Cancer incidence AI Lung Health")
print(res.cox)

# COX AI lung health Respirotary disease death adjusted for everything
res.cox <- coxph(Surv(fup, j) ~ DeepPY + gen + age + sct_long_dia_min6 + packyear +  cigsmok + stro_hear_hype_canc + bmi + copd + diab  + fibr + pneu + chro + bron + adass + asbe + emph + sili + sarc + tube, data = data)
print("COX Respiratory Disease Death AI Lung Health")
print(res.cox)

# COX AI lung health CVD death adjusted for everything-related
res.cox <- coxph(Surv(fup, cvd) ~ DeepPY + gen + age + diab + stro + hear + hype + cigsmok + bmi, data = data)
print("COX CVD Death AI Lung Health")
print(res.cox)