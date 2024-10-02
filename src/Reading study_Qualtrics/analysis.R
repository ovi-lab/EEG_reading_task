# Load necessary libraries with error handling
required_packages <- c("dplyr", "ggplot2", "tidyr", "reshape2", "car", "effsize", "ez")
for(pkg in required_packages){
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Read the dataset
data <- read.csv('~/Desktop/Reading study_Qualtrics/post-study-questionaire.csv')

# Step 1: Remove specified participant IDs and ensure unique p_id
exclude_pids <- c(14, 5, 13, 16, 17, 20, 20.5, 31)
data_cleaned <- data %>%
  filter(!p_id %in% exclude_pids) %>%
  distinct(p_id, .keep_all = TRUE)

# Step 2: Impute missing values for numeric columns with mean
numeric_cols <- sapply(data_cleaned, is.numeric)
data_cleaned <- data_cleaned %>%
  mutate(across(which(numeric_cols), ~ifelse(is.na(.), mean(., na.rm=TRUE), .)))

# Step 3: Remove outliers using z-score method on numeric data
numeric_data <- select(data_cleaned, -p_id) %>% select_if(is.numeric)
z_scores <- as.data.frame(scale(numeric_data))
data_no_outliers <- data_cleaned[apply(abs(z_scores) < 3, 1, all), ]

# Step 4: Conduct paired t-tests and calculate Cohen's d
paired_columns <- grep("^nd_", names(data_no_outliers), value=TRUE)
anova_results <- list()
t_test_results <- list()

# Function to perform analysis
perform_analysis <- function(nd_col, d_col, data) {
  nd_data <- data[[nd_col]]
  d_data <- data[[d_col]]
  
  # Paired t-test
  t_test <- t.test(nd_data, d_data, paired=TRUE)
  
  # Calculate Cohen's d using 'effsize'
  cohen_d <- cohen.d(nd_data, d_data, paired = TRUE)$estimate
  
  # Prepare long data for ANOVA
  long_data <- data.frame(
    p_id = data$p_id,
    condition = rep(c("Non-distractor", "Distractor"), each = nrow(data)),
    measure = c(nd_data, d_data)
  )
  
  # Perform Repeated Measures ANOVA using ezANOVA
  anova_result <- ezANOVA(
    data = long_data,
    dv = measure,
    wid = p_id,
    within = condition,
    detailed = TRUE
  )
  
  list(
    t_stat = t_test$statistic,
    p_value = t_test$p.value,
    cohen_d = cohen_d,
    anova = anova_result
  )
}

# Loop through paired columns and perform analysis
for (col in paired_columns) {
  d_col <- sub("^nd_", "d_", col)
  
  # Check if corresponding d_col exists
  if(!d_col %in% names(data_no_outliers)){
    warning(paste("Corresponding distractor column", d_col, "not found for", col))
    next
  }
  
  results <- perform_analysis(col, d_col, data_no_outliers)
  
  anova_results[[col]] <- results$anova
  t_test_results[[col]] <- list(
    t_stat = results$t_stat,
    p_value = results$p_value,
    cohen_d = results$cohen_d
  )
}

# Print T-test results
print("Paired t-Test Results:")
print(t_test_results)

# Print ANOVA results
print("Repeated Measures ANOVA Results:")
print(anova_results)

# Initialize a data frame to store t-test results
t_test_results_df <- data.frame(
  Measure = character(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

# Loop through paired columns and perform analysis
for (col in paired_columns) {
  d_col <- sub("^nd_", "d_", col)
  
  # Check if corresponding d_col exists
  if(!d_col %in% names(data_no_outliers)){
    warning(paste("Corresponding distractor column", d_col, "not found for", col))
    next
  }
  
  results <- perform_analysis(col, d_col, data_no_outliers)
  
  anova_results[[col]] <- results$anova
  t_test_results[[col]] <- list(
    t_stat = results$t_stat,
    p_value = results$p_value,
    cohen_d = results$cohen_d
  )
  
  # Store the p-values and measure names in the data frame
  measure_name <- sub("^(nd|d)_", "", col)
  t_test_results_df <- rbind(t_test_results_df, data.frame(
    Measure = measure_name,
    p_value = results$p_value,
    stringsAsFactors = FALSE
  ))
}

# Add significance levels based on p-values
t_test_results_df <- t_test_results_df %>%
  mutate(
    signif_label = case_when(
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ ""
    )
  )


# Step 5: Plot all measures in bar chart with 0.95 confidence intervals using generic terms

# Reshape data to long format
plot_long <- data_no_outliers %>%
  pivot_longer(
    cols = starts_with("nd_") | starts_with("d_"),
    names_to = c("condition", "Measure"),
    names_pattern = "(nd|d)_(.*)",
    values_to = "value"
  ) %>%
  mutate(
    Condition = case_when(
      condition == "nd" ~ "Non-distractor",
      condition == "d" ~ "Distractor",
      TRUE ~ condition
    )
  ) %>%
  select(-condition)

# Calculate summary statistics
plot_summary <- plot_long %>%
  group_by(Measure, Condition) %>%
  summarise(
    Mean = mean(value, na.rm = TRUE),
    SD = sd(value, na.rm = TRUE),
    N = n(),
    SE = SD / sqrt(N),
    CI = 1.96 * SE,
    .groups = 'drop'
  )

# Merge t-test results with plot summary
plot_summary <- plot_summary %>%
  left_join(t_test_results_df, by = "Measure")

# Calculate positions for significance annotations
annotations <- plot_summary %>%
  group_by(Measure) %>%
  summarise(
    y_position = max(Mean + CI) + 0.2,
    signif_label = unique(signif_label)
  ) %>%
  filter(signif_label != "")  # Keep only significant results

# Adjust Measure to factor to maintain order
plot_summary$Measure <- factor(plot_summary$Measure, levels = unique(plot_summary$Measure))
annotations$Measure <- factor(annotations$Measure, levels = levels(plot_summary$Measure))

# Create the plot
ggplot(plot_summary, aes(x = Measure, y = Mean, fill = Condition)) +
  geom_bar(
    stat = "identity",
    position = position_dodge(width = 0.9),
    width = 0.8
  ) +
  geom_errorbar(
    aes(ymin = Mean - CI, ymax = Mean + CI),
    width = 0.2,
    position = position_dodge(width = 0.9)
  ) +
  # Add significance lines
  geom_segment(
    data = annotations,
    aes(
      x = as.numeric(Measure) - 0.2,
      xend = as.numeric(Measure) + 0.2,
      y = y_position,
      yend = y_position
    ),
    inherit.aes = FALSE,
    color = "black"
  ) +
  # Add significance labels
  geom_text(
    data = annotations,
    aes(
      x = as.numeric(Measure),
      y = y_position + 0.05,
      label = signif_label
    ),
    inherit.aes = FALSE,
    color = "black",
    size = 5
  ) +
  labs(
    title = "Comparison of Measures Between Conditions with 95% CI",
    x = "Measure",
    y = "Mean Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  ) +
  scale_fill_brewer(palette = "Set1")
