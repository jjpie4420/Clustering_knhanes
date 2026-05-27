# install.packages(c("survey", "dplyr", "readr", "tidyr"))

library(survey)
library(dplyr)
library(readr)
library(tidyr)

input <- "D:/precision_nutrition/FFQ/processed/analysis_cohort_energy_adjusted.csv"

out_overall <- "D:/precision_nutrition/FFQ/results/27_survey_table1_overall.csv"
out_targets <- "D:/precision_nutrition/FFQ/results/27_survey_target_prevalence.csv"
out_by_sex <- "D:/precision_nutrition/FFQ/results/27_survey_target_by_sex.csv"

df <- read_csv(input, show_col_types = FALSE)

# KNHANES complex survey variables
weight_col <- "wt_ntr"
strata_col <- "kstrata"
psu_col <- "psu"

needed <- c(weight_col, strata_col, psu_col)
missing_needed <- setdiff(needed, names(df))

if (length(missing_needed) > 0) {
  stop(paste("Missing survey design variables:", paste(missing_needed, collapse = ", ")))
}

df <- df %>%
  filter(
    !is.na(.data[[weight_col]]),
    !is.na(.data[[strata_col]]),
    !is.na(.data[[psu_col]]),
    .data[[weight_col]] > 0
  )

options(survey.lonely.psu = "adjust")

design <- svydesign(
  id = as.formula(paste0("~", psu_col)),
  strata = as.formula(paste0("~", strata_col)),
  weights = as.formula(paste0("~", weight_col)),
  data = df,
  nest = TRUE
)

continuous_vars <- c(
  "age", "HE_BMI", "HE_wc", "HE_sbp", "HE_dbp",
  "HE_glu", "HE_HbA1c", "HE_chol", "HE_HDL_st2", "HE_TG",
  "FQ_EN", "RC_EN", "FQ_PROT", "RC_PROT", "FQ_FAT", "RC_FAT",
  "FQ_CHO", "RC_CHO", "FQ_TDF", "RC_TDF", "FQ_NA", "RC_NA"
)

categorical_vars <- c(
  "sex", "incm", "edu", "educ", "sm_presnt", "dr_month", "pa_aerobic"
)

targets <- c(
  "label_diabetes",
  "label_diabetes_untreated",
  "label_prediabetes_clean",
  "label_hypertension",
  "label_hypertension_untreated",
  "label_prehypertension_clean",
  "label_dyslipidemia",
  "label_dyslipidemia_untreated",
  "label_obesity"
)

continuous_vars <- intersect(continuous_vars, names(df))
categorical_vars <- intersect(categorical_vars, names(df))
targets <- intersect(targets, names(df))

# ------------------------------------------------------------
# Overall Table 1
# ------------------------------------------------------------

overall_cont <- lapply(continuous_vars, function(v) {
  f <- as.formula(paste0("~", v))
  m <- svymean(f, design, na.rm = TRUE)
  
  data.frame(
    variable = v,
    type = "continuous",
    level = NA,
    weighted_mean = as.numeric(coef(m)),
    se = as.numeric(SE(m)),
    ci_lower = as.numeric(confint(m)[, 1]),
    ci_upper = as.numeric(confint(m)[, 2])
  )
}) %>% bind_rows()

overall_cat <- lapply(categorical_vars, function(v) {
  f <- as.formula(paste0("~factor(", v, ")"))
  p <- svymean(f, design, na.rm = TRUE)
  
  data.frame(
    variable = v,
    type = "categorical",
    level = names(coef(p)),
    weighted_percent = as.numeric(coef(p)) * 100,
    se_percent = as.numeric(SE(p)) * 100,
    ci_lower_percent = as.numeric(confint(p)[, 1]) * 100,
    ci_upper_percent = as.numeric(confint(p)[, 2]) * 100
  )
}) %>% bind_rows()

overall_table <- bind_rows(overall_cont, overall_cat)

write_csv(overall_table, out_overall)

# ------------------------------------------------------------
# Weighted phenotype prevalence
# ------------------------------------------------------------

target_prev <- lapply(targets, function(v) {
  df[[v]] <- ifelse(df[[v]] %in% c(0, 1), df[[v]], NA)
  
  design_tmp <- update(design, tmp_target = df[[v]])
  m <- svymean(~tmp_target, design_tmp, na.rm = TRUE)
  
  data.frame(
    target = v,
    n_unweighted = sum(!is.na(df[[v]])),
    n_positive_unweighted = sum(df[[v]] == 1, na.rm = TRUE),
    prevalence_unweighted = mean(df[[v]], na.rm = TRUE),
    prevalence_weighted = as.numeric(coef(m)),
    se = as.numeric(SE(m)),
    ci_lower = as.numeric(confint(m)[, 1]),
    ci_upper = as.numeric(confint(m)[, 2]),
    prevalence_weighted_percent = as.numeric(coef(m)) * 100
  )
}) %>% bind_rows()

write_csv(target_prev, out_targets)

# ------------------------------------------------------------
# Weighted phenotype prevalence by sex
# ------------------------------------------------------------

by_sex <- lapply(targets, function(v) {
  df[[v]] <- ifelse(df[[v]] %in% c(0, 1), df[[v]], NA)
  
  design_tmp <- update(design, tmp_target = df[[v]])
  
  res <- svyby(
    ~tmp_target,
    ~sex,
    design_tmp,
    svymean,
    na.rm = TRUE,
    vartype = c("se", "ci")
  )
  
  data.frame(
    target = v,
    sex = res$sex,
    prevalence_weighted = res$tmp_target,
    se = res$se,
    ci_lower = res$ci_l,
    ci_upper = res$ci_u,
    prevalence_weighted_percent = res$tmp_target * 100
  )
}) %>% bind_rows()

write_csv(by_sex, out_by_sex)

cat("\n[SAVED]\n")
cat(out_overall, "\n")
cat(out_targets, "\n")
cat(out_by_sex, "\n")