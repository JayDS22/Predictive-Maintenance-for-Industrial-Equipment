#!/usr/bin/env Rscript
# Survival analysis on the turbofan dataset.
#
# Inputs : data/turbofan_synthetic.csv
# Outputs: reports/survival_summary.txt
#          reports/km_curve.png
#          reports/cox_hazard_ratios.csv
#
# Usage:
#   Rscript src/survival_analysis.R [data_path] [out_dir]

suppressPackageStartupMessages({
  required <- c("survival", "survminer", "dplyr", "readr", "ggplot2")
  missing <- setdiff(required, rownames(installed.packages()))
  if (length(missing) > 0) {
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
  invisible(lapply(required, library, character.only = TRUE))
})

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[1] else "data/turbofan_synthetic.csv"
out_dir   <- if (length(args) >= 2) args[2] else "reports"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("Loading %s...\n", data_path))
df <- read_csv(data_path, show_col_types = FALSE)

# One survival row per unit: cycles-to-failure plus averaged sensor covariates.
# All units run to failure in synthetic data, so event = 1 throughout.
unit_summary <- df %>%
  group_by(unit_id) %>%
  summarise(
    life_cycles = max(cycle),
    avg_sensor_3 = mean(sensor_3),
    avg_sensor_4 = mean(sensor_4),
    avg_sensor_9 = mean(sensor_9),
    avg_sensor_14 = mean(sensor_14),
    .groups = "drop"
  ) %>%
  mutate(event = 1L)

km <- survfit(Surv(life_cycles, event) ~ 1, data = unit_summary)
km_plot <- ggsurvplot(
  km, data = unit_summary,
  conf.int = TRUE, risk.table = TRUE,
  xlab = "Cycles", ylab = "Survival probability",
  title = "Kaplan-Meier survival, synthetic turbofan fleet",
  ggtheme = theme_minimal()
)
ggsave(file.path(out_dir, "km_curve.png"), plot = km_plot$plot, width = 8, height = 5, dpi = 150)

cox <- coxph(
  Surv(life_cycles, event) ~ avg_sensor_3 + avg_sensor_4 + avg_sensor_9 + avg_sensor_14,
  data = unit_summary
)

cox_summary <- summary(cox)
hr_df <- data.frame(
  covariate = rownames(cox_summary$conf.int),
  hazard_ratio = cox_summary$conf.int[, "exp(coef)"],
  lower_95 = cox_summary$conf.int[, "lower .95"],
  upper_95 = cox_summary$conf.int[, "upper .95"],
  p_value = cox_summary$coefficients[, "Pr(>|z|)"]
)
write_csv(hr_df, file.path(out_dir, "cox_hazard_ratios.csv"))

c_index <- cox_summary$concordance[["C"]]

sink(file.path(out_dir, "survival_summary.txt"))
cat("== Survival analysis summary ==\n")
cat(sprintf("Units analysed: %d\n", nrow(unit_summary)))
cat(sprintf("Median life cycles: %.1f\n", median(unit_summary$life_cycles)))
cat(sprintf("Cox concordance index (C-index): %.3f\n\n", c_index))
print(cox_summary)
sink()

cat(sprintf("Cox C-index = %.3f, hazard ratios written to %s/cox_hazard_ratios.csv\n",
            c_index, out_dir))
