# # Import necessary libraries
# import pandas as pd
# import numpy as np
# from scipy import stats
# import statsmodels.api as sm
# from statsmodels.stats.anova import AnovaRM
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# import pingouin as pg
# import os 

# # Suppress warnings for clean output
# warnings.filterwarnings('ignore')

# # ===========================
# # Step 1: Load and Clean Data
# # ===========================

# # Load the dataset 
# file_path = 'src\Reading study_Qualtrics\post-study-questionaire.csv'  # Ensure this path is correct
# data = pd.read_csv(file_path)

# # Clean column names by stripping extra spaces
# data.columns = data.columns.str.strip()

# # ===========================
# # Step 2: Remove Specific Participant IDs
# # ===========================

# participant_ids_to_remove = [14, 5, 13, 16, 17, 20, 20.5, 31]

# # Ensure 'p_id' is of numeric type for proper filtering
# data['p_id'] = pd.to_numeric(data['p_id'], errors='coerce')

# # Remove specified participant IDs
# data_cleaned = data[~data['p_id'].isin(participant_ids_to_remove)].copy()

# # Optional: Reset index after filtering
# data_cleaned.reset_index(drop=True, inplace=True)

# # ===========================
# # Step 3: Impute Missing Values
# # ===========================

# # Identify numeric columns for imputation
# numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns.tolist()

# # Impute missing values in numeric columns with the mean (rounded)
# data_cleaned[numeric_cols] = data_cleaned[numeric_cols].apply(
#     lambda x: x.fillna(round(x.mean())) if x.isnull().any() else x
# )

# # Optional: Verify no missing values remain
# # print(data_cleaned[numeric_cols].isnull().sum())

# # ===========================
# # Step 4: Remove Outliers Using the IQR Method
# # ===========================

# def remove_outliers_iqr(df, columns):
#     """
#     Removes rows with outliers in any of the specified columns based on the IQR method.
#     """
#     Q1 = df[columns].quantile(0.25)
#     Q3 = df[columns].quantile(0.75)
#     IQR = Q3 - Q1
#     # Define bounds
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     # Filter out outliers
#     filtered_df = df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)].copy()
#     return filtered_df

# # Define the columns to check for outliers
# columns_to_check = [
#     'nd_focus', 'nd_mental_demand', 'nd_physical_demand', 'nd_temp_demand', 
#     'nd_perf', 'nd_effort', 'nd_frustration', 
#     'd_focus', 'd_mental_demand', 'd_physical_demand', 'd_temp_demand', 
#     'd_perf', 'd_effort', 'd_frustration'
# ]

# # Ensure all columns exist in the DataFrame
# missing_cols = set(columns_to_check) - set(data_cleaned.columns)
# if missing_cols:
#     raise ValueError(f"The following columns are missing in the dataset: {missing_cols}")

# # Remove outliers
# data_no_outliers = remove_outliers_iqr(data_cleaned, columns_to_check)

# # Optional: Check the number of rows removed
# # print(f"Rows before outlier removal: {data_cleaned.shape[0]}, after: {data_no_outliers.shape[0]}")

# # ===========================
# # Step 5: Prepare Data for Repeated Measures ANOVA
# # ===========================

# # Melt the data to long format
# anova_data = pd.melt(
#     data_no_outliers,
#     id_vars=['p_id'],
#     value_vars=columns_to_check,
#     var_name='condition_measure',
#     value_name='score'
# )

# # Split 'condition_measure' into 'type' and 'measure'
# anova_data[['type', 'measure']] = anova_data['condition_measure'].str.extract(r'(d|nd)_(.*)')

# # Convert 'type' and 'measure' to categorical variables
# anova_data['type'] = anova_data['type'].map({'d': 'Distractor', 'nd': 'Non-Distractor'})
# anova_data['measure'] = anova_data['measure'].str.replace('_', ' ').str.title()

# # Optional: Preview the transformed data
# # print(anova_data.head())

# # ===========================
# # Step 6: Conduct Repeated Measures ANOVA
# # ===========================

# # Define unique measures
# unique_measures = anova_data['measure'].unique()

# # Initialize a list to store ANOVA results
# anova_results_list = []

# print("\n=== Detailed Repeated Measures ANOVA Results ===\n")

# for measure in unique_measures:
#     # Subset data for the current measure
#     measure_data = anova_data[anova_data['measure'] == measure]
    
#     # Perform Repeated Measures ANOVA for 'type' within each measure
#     # Since 'type' has only two levels, it's equivalent to a paired t-test
#     try:
#         aov = pg.rm_anova(
#             data=measure_data,
#             dv='score',
#             within=['type'],
#             subject='p_id',
#             detailed=True
#         )
#         # Add measure name to the results
#         aov['Measure'] = measure
#         # Append to the list
#         anova_results_list.append(aov)
        
#         # Print detailed results
#         print(f"--- ANOVA for Measure: {measure} ---")
#         print(aov.to_string(index=False))
#         print("\n")
        
#     except Exception as e:
#         print(f"ANOVA failed for measure '{measure}' with error: {e}")
#         # Proceed to paired t-tests if ANOVA fails
#         pass

# # Combine all ANOVA results into a single DataFrame
# if anova_results_list:
#     anova_results_df = pd.concat(anova_results_list, ignore_index=True)
#     # Print column names to verify
#     print("Columns in ANOVA Results DataFrame:", anova_results_df.columns.tolist())
#     # Adjust column selection based on actual columns
#     # Here, 'Source' instead of 'Effect', 'DF' instead of 'DFn', etc.
#     # Update this line to match the actual column names
#     # Original line causing error:
#     # anova_results_df = anova_results_df[['Measure', 'Source', 'DF', 'F', 'p-unc', 'np2']]
    
#     # Corrected line:
#     anova_results_df = anova_results_df[['Measure', 'Source', 'DF', 'F', 'p-unc', 'ng2']]
    
#     # Optional: Rename 'ng2' to 'np2' if desired
#     anova_results_df = anova_results_df.rename(columns={'ng2': 'np2'})
    
#     print("\n=== Summary of All Repeated Measures ANOVA ===")
#     print(anova_results_df.to_string(index=False))
    
#     # ===========================
#     # Step 7: Conduct Paired t-Tests and Calculate Effect Sizes (If Needed)
#     # ===========================
    
#     # Initialize a list to store t-test results
#     ttest_results_list = []
    
#     print("\n=== Paired t-Test Results ===\n")
    
#     for measure in unique_measures:
#         # Define column names
#         nd_col = f'nd_{measure.lower().replace(" ", "_")}'
#         d_col = f'd_{measure.lower().replace(" ", "_")}'
        
#         # Check if both columns exist
#         if nd_col not in data_no_outliers.columns or d_col not in data_no_outliers.columns:
#             print(f"Columns for measure '{measure}' are missing. Skipping t-test.")
#             continue
        
#         # Extract data
#         nd_scores = data_no_outliers[nd_col]
#         d_scores = data_no_outliers[d_col]
        
#         # Perform paired t-test
#         t_stat, p_val = stats.ttest_rel(d_scores, nd_scores)
        
#         # Calculate Cohen's d using Pingouin
#         cohen_d = pg.compute_effsize(d_scores, nd_scores, paired=True)
        
#         # Store results in a dictionary
#         ttest_results_list.append({
#             'Measure': measure,
#             't_stat': t_stat,
#             'p_val': p_val,
#             'cohen_d': cohen_d
#         })
        
#         # Print detailed t-test results
#         print(f"--- Paired t-Test for Measure: {measure} ---")
#         print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}, Cohen's d: {cohen_d:.4f}\n")
    
#     # Convert t-test results to DataFrame
#     ttest_results_df = pd.DataFrame(ttest_results_list)
    
#     # Optional: Apply multiple comparisons correction (e.g., Bonferroni)
#     from statsmodels.stats.multitest import multipletests
#     ttest_results_df['p_adj_bonferroni'] = multipletests(ttest_results_df['p_val'], method='bonferroni')[1]
    
#     # Display t-test results
#     print("=== Summary of All Paired t-Tests ===")
#     print(ttest_results_df.to_string(index=False))
    
# # ===========================
# # Step 8: Visualization
# # ===========================

# # For visualization, we'll plot the mean scores with 95% confidence intervals for each measure and condition

# # Calculate summary statistics
# summary_stats = anova_data.groupby(['measure', 'type']).agg(
#     Mean=('score', 'mean'),
#     SEM=('score', lambda x: stats.sem(x, nan_policy='omit'))
# ).reset_index()

# # Calculate 95% Confidence Intervals
# summary_stats['CI'] = 1.96 * summary_stats['SEM']

# # Print summary statistics for verification
# print("\n=== Summary Statistics for Visualization ===")
# print(summary_stats.to_string(index=False))

# # Set up the plot
# plt.figure(figsize=(12, 8))
# sns.set(style="whitegrid")

# # Create a barplot without error bars
# ax = sns.barplot(
#     x='measure',
#     y='Mean',
#     hue='type',
#     data=summary_stats,
#     palette='Set1',
#     ci=None,  # We'll add custom error bars
#     capsize=0.05,
#     errwidth=1
# )

# # Loop over each bar and corresponding summary statistic
# for bar, (_, row) in zip(ax.patches, summary_stats.iterrows()):
#     # Extract the CI value
#     ci = row['CI']
#     # Get the center of the bar
#     x = bar.get_x() + bar.get_width() / 2
#     y = bar.get_height()
#     # Plot the error bar
#     ax.errorbar(
#         x=x,
#         y=y,
#         yerr=ci,
#         fmt='none',
#         c='black',
#         capsize=5
#     )

# # Enhance plot aesthetics
# plt.title('Comparison of Non-Distractor vs Distractor Conditions (with 95% CI)')
# plt.xlabel('Measure')
# plt.ylabel('Mean Score')
# plt.xticks(rotation=45)
# plt.legend(title='Condition')
# plt.tight_layout()
# plt.show()


# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pingouin as pg
import os

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 200 # 200 

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# ===========================
# Step 1: Load and Clean Data
# ===========================

# Load the dataset
file_path = 'src\Reading study_Qualtrics\post-study-questionaire.csv'  # Ensure this path is correct
data = pd.read_csv(file_path)

# Clean column names by stripping extra spaces
data.columns = data.columns.str.strip()

# ===========================
# Step 2: Remove Specific Participant IDs
# ===========================

participant_ids_to_remove = [14, 5, 13, 16, 17, 20, 20.5, 31]

# Ensure 'p_id' is of numeric type for proper filtering
data['p_id'] = pd.to_numeric(data['p_id'], errors='coerce')

# Remove specified participant IDs
data_cleaned = data[~data['p_id'].isin(participant_ids_to_remove)].copy()

# Optional: Reset index after filtering
data_cleaned.reset_index(drop=True, inplace=True)

# ===========================
# Step 3: Impute Missing Values
# ===========================

# Identify numeric columns for imputation
numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns.tolist()

# Impute missing values in numeric columns with the mean (rounded)
data_cleaned[numeric_cols] = data_cleaned[numeric_cols].apply(
    lambda x: x.fillna(round(x.mean())) if x.isnull().any() else x
)

# Optional: Verify no missing values remain
# print(data_cleaned[numeric_cols].isnull().sum())

# ===========================
# Step 4: Remove Outliers Using the IQR Method
# ===========================

def remove_outliers_iqr(df, columns):
    """
    Removes rows with outliers in any of the specified columns based on the IQR method.
    """
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filter out outliers
    filtered_df = df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)].copy()
    return filtered_df

# Define the columns to check for outliers
columns_to_check = [
    'nd_focus', 'nd_mental_demand', 'nd_physical_demand', 'nd_temp_demand',
    'nd_perf', 'nd_effort', 'nd_frustration',
    'd_focus', 'd_mental_demand', 'd_physical_demand', 'd_temp_demand',
    'd_perf', 'd_effort', 'd_frustration'
]

# Ensure all columns exist in the DataFrame
missing_cols = set(columns_to_check) - set(data_cleaned.columns)
if missing_cols:
    raise ValueError(f"The following columns are missing in the dataset: {missing_cols}")

# Remove outliers
data_no_outliers = remove_outliers_iqr(data_cleaned, columns_to_check)

# Optional: Check the number of rows removed
# print(f"Rows before outlier removal: {data_cleaned.shape[0]}, after: {data_no_outliers.shape[0]}")

# ===========================
# Step 5: Prepare Data for Repeated Measures ANOVA
# ===========================

# Melt the data to long format
stat_data = pd.melt(
    data_no_outliers,
    id_vars=['p_id'],
    value_vars=columns_to_check,
    var_name='condition_measure',
    value_name='score'
)

# Split 'condition_measure' into 'type' and 'measure'
stat_data[['type', 'measure']] = stat_data['condition_measure'].str.extract(r'(d|nd)_(.*)')

# Convert 'type' and 'measure' to categorical variables
stat_data['type'] = stat_data['type'].map({'d': 'Distractor', 'nd': 'Non-Distractor'})
stat_data['measure'] = stat_data['measure'].str.replace('_', ' ').str.title()

# Optional: Preview the transformed data
# print(anova_data.head())

# ===========================
# Step 6: Conduct Repeated Measures ANOVA (Optional)
# ===========================

# Since we will be performing t-tests or Wilcoxon tests based on normality,
# we can skip the ANOVA step or keep it for comparison purposes.

# ===========================
# Step 7: Statistical Tests Based on Normality and Effect Sizes
# ===========================

# Define unique measures
unique_measures = stat_data['measure'].unique()

# Initialize a list to store test results
test_results_list = []

print("\n=== Statistical Test Results ===\n")

for measure in unique_measures:
    # Define column names
    nd_col = f'nd_{measure.lower().replace(" ", "_")}'
    d_col = f'd_{measure.lower().replace(" ", "_")}'
    
    # Check if both columns exist
    if nd_col not in data_no_outliers.columns or d_col not in data_no_outliers.columns:
        print(f"Columns for measure '{measure}' are missing. Skipping tests.")
        continue
    
    # Extract data
    nd_scores = data_no_outliers[nd_col]
    d_scores = data_no_outliers[d_col]
    
    # Compute differences
    diff_scores = d_scores - nd_scores
    
    # Perform Shapiro-Wilk test for normality on differences
    shapiro_stat, shapiro_p = stats.shapiro(diff_scores)
    
    # Decide which test to use
    if shapiro_p > 0.05:
        # Data is normally distributed, use paired t-test
        test_name = 'Paired t-test'
        test_res = pg.ttest(d_scores, nd_scores, paired=True)
        t_stat = test_res['T'].values[0]
        p_val = test_res['p-val'].values[0]
        effect_size = test_res['cohen-d'].values[0]
        effect_size_name = "Cohen's d"
    else:
        # Data is not normally distributed, use Wilcoxon signed-rank test
        test_name = 'Wilcoxon signed-rank test'
        test_res = pg.wilcoxon(d_scores, nd_scores)
        t_stat = test_res['W-val'].values[0]
        p_val = test_res['p-val'].values[0]
        effect_size = test_res['RBC'].values[0]  # Rank-biserial correlation
        effect_size_name = 'Rank-biserial correlation'
    
    # Store results in a dictionary
    test_results_list.append({
        'Measure': measure,
        'Test': test_name,
        'Test Statistic': t_stat,
        'p-value': p_val,
        'Effect Size': effect_size,
        'Effect Size Name': effect_size_name,
        'Shapiro p-value': shapiro_p
    })
    
    # Print detailed test results
    print(f"--- {test_name} for Measure: {measure} ---")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"Test Statistic: {t_stat:.4f}, p-value: {p_val:.4f}, {effect_size_name}: {effect_size:.4f}\n")

# Convert test results to DataFrame
test_results_df = pd.DataFrame(test_results_list)

# Adjust p-values for multiple comparisons
from statsmodels.stats.multitest import multipletests

# You can choose the correction method, e.g., 'bonferroni', 'fdr_bh' (Benjamini-Hochberg)
correction_method = 'bonferroni'

adjusted_pvals = multipletests(test_results_df['p-value'], method=correction_method)
test_results_df['p_adj'] = adjusted_pvals[1]

# Apply significance levels based on adjusted p-values
def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

test_results_df['significance'] = test_results_df['p_adj'].apply(get_significance_stars)

# Display test results
print("=== Summary of Statistical Tests with Multiple Comparisons Correction ===")
print(test_results_df[['Measure', 'Test', 'Test Statistic', 'p-value', 'p_adj', 'Effect Size', 'Effect Size Name']].to_string(index=False))


# ===========================
# Step 8: Visualization with Significance Annotations
# ===========================

# Set up the plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create the barplot
ax = sns.barplot(
    x='measure',
    y='score',
    hue='type',
    data=stat_data,
    ci=95  # 95% confidence intervals
)

# Adjust y-axis limits to make space for annotations
ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Annotate significance
from itertools import product

# Create a mapping of measures to x-axis positions
measure_names = stat_data['measure'].unique()
measure_positions = {measure: idx for idx, measure in enumerate(measure_names)}

# Define offsets for each hue level
hue_levels = stat_data['type'].unique()
hue_offsets = {hue: idx - 1 for idx, hue in enumerate(hue_levels)}

# Bar width based on number of hue levels
bar_width = 0.8 / len(hue_levels)

for _, row in test_results_df.iterrows():
    measure = row['Measure']
    significance = row['significance']

    # Skip if not significant
    if significance == 'ns':
        continue

    # Get x positions for both conditions
    x_base = measure_positions[measure]
    x1 = x_base + hue_offsets[hue_levels[0]] * bar_width + bar_width / 2 -0.15
    x2 = x_base + hue_offsets[hue_levels[1]] * bar_width + bar_width / 2 +0.15

    # Get y positions (top of the error bars)
    y1 = stat_data[(stat_data['measure'] == measure) & (stat_data['type'] == hue_levels[0])]['score'].mean()
    y1_err = stats.sem(stat_data[(stat_data['measure'] == measure) & (stat_data['type'] == hue_levels[0])]['score']) * 1.96
    y2 = stat_data[(stat_data['measure'] == measure) & (stat_data['type'] == hue_levels[1])]['score'].mean()
    y2_err = stats.sem(stat_data[(stat_data['measure'] == measure) & (stat_data['type'] == hue_levels[1])]['score']) * 1.96

    max_y = max(y1 + y1_err, y2 + y2_err)

    # Height for the significance line
    h = ax.get_ylim()[1] * 0.02
    y = max_y + h + 0.2

    # Draw the significance line
    ax.plot([x1 , x1, x2, x2], [y - h, y, y, y - h], lw=1.5, c='k')

    # Add the significance text
    ax.text((x1 + x2) / 2, y + h, significance, ha='center', va='bottom', color='k', fontsize=12)

# Enhance plot aesthetics
plt.title('Comparison of Non-Distractor vs Distractor Conditions')
plt.xlabel('Measure')
plt.ylabel('Mean Score')
plt.legend(title='Condition')
plt.tight_layout()
plt.show()



mean_std_summary = stat_data.groupby(['measure', 'type']).agg(
    Mean=('score', 'mean'),
    Std=('score', 'std')
).reset_index()

# Display the summary table
print("=== Mean and Standard Deviation for Each Measure and Condition ===")
print(mean_std_summary.to_string(index=False))
