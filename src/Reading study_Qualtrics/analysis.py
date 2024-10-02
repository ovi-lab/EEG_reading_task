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

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# ===========================
# Step 1: Load and Clean Data
# ===========================

# Load the dataset
file_path = 'post-study-questionaire.csv'  # Ensure this path is correct
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
anova_data = pd.melt(
    data_no_outliers,
    id_vars=['p_id'],
    value_vars=columns_to_check,
    var_name='condition_measure',
    value_name='score'
)

# Split 'condition_measure' into 'type' and 'measure'
anova_data[['type', 'measure']] = anova_data['condition_measure'].str.extract(r'(d|nd)_(.*)')

# Convert 'type' and 'measure' to categorical variables
anova_data['type'] = anova_data['type'].map({'d': 'Distractor', 'nd': 'Non-Distractor'})
anova_data['measure'] = anova_data['measure'].str.replace('_', ' ').str.title()

# Optional: Preview the transformed data
# print(anova_data.head())

# ===========================
# Step 6: Conduct Repeated Measures ANOVA
# ===========================

# Define unique measures
unique_measures = anova_data['measure'].unique()

# Initialize a list to store ANOVA results
anova_results_list = []

print("\n=== Detailed Repeated Measures ANOVA Results ===\n")

for measure in unique_measures:
    # Subset data for the current measure
    measure_data = anova_data[anova_data['measure'] == measure]
    
    # Perform Repeated Measures ANOVA for 'type' within each measure
    # Since 'type' has only two levels, it's equivalent to a paired t-test
    try:
        aov = pg.rm_anova(
            data=measure_data,
            dv='score',
            within=['type'],
            subject='p_id',
            detailed=True
        )
        # Add measure name to the results
        aov['Measure'] = measure
        # Append to the list
        anova_results_list.append(aov)
        
        # Print detailed results
        print(f"--- ANOVA for Measure: {measure} ---")
        print(aov.to_string(index=False))
        print("\n")
        
    except Exception as e:
        print(f"ANOVA failed for measure '{measure}' with error: {e}")
        # Proceed to paired t-tests if ANOVA fails
        pass

# Combine all ANOVA results into a single DataFrame
if anova_results_list:
    anova_results_df = pd.concat(anova_results_list, ignore_index=True)
    # Print column names to verify
    print("Columns in ANOVA Results DataFrame:", anova_results_df.columns.tolist())
    # Adjust column selection based on actual columns
    # Here, 'Source' instead of 'Effect', 'DF' instead of 'DFn', etc.
    # Update this line to match the actual column names
    # Original line causing error:
    # anova_results_df = anova_results_df[['Measure', 'Source', 'DF', 'F', 'p-unc', 'np2']]
    
    # Corrected line:
    anova_results_df = anova_results_df[['Measure', 'Source', 'DF', 'F', 'p-unc', 'ng2']]
    
    # Optional: Rename 'ng2' to 'np2' if desired
    anova_results_df = anova_results_df.rename(columns={'ng2': 'np2'})
    
    print("\n=== Summary of All Repeated Measures ANOVA ===")
    print(anova_results_df.to_string(index=False))
    
    # ===========================
    # Step 7: Conduct Paired t-Tests and Calculate Effect Sizes (If Needed)
    # ===========================
    
    # Initialize a list to store t-test results
    ttest_results_list = []
    
    print("\n=== Paired t-Test Results ===\n")
    
    for measure in unique_measures:
        # Define column names
        nd_col = f'nd_{measure.lower().replace(" ", "_")}'
        d_col = f'd_{measure.lower().replace(" ", "_")}'
        
        # Check if both columns exist
        if nd_col not in data_no_outliers.columns or d_col not in data_no_outliers.columns:
            print(f"Columns for measure '{measure}' are missing. Skipping t-test.")
            continue
        
        # Extract data
        nd_scores = data_no_outliers[nd_col]
        d_scores = data_no_outliers[d_col]
        
        # Perform paired t-test
        t_stat, p_val = stats.ttest_rel(d_scores, nd_scores)
        
        # Calculate Cohen's d using Pingouin
        cohen_d = pg.compute_effsize(d_scores, nd_scores, paired=True)
        
        # Store results in a dictionary
        ttest_results_list.append({
            'Measure': measure,
            't_stat': t_stat,
            'p_val': p_val,
            'cohen_d': cohen_d
        })
        
        # Print detailed t-test results
        print(f"--- Paired t-Test for Measure: {measure} ---")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}, Cohen's d: {cohen_d:.4f}\n")
    
    # Convert t-test results to DataFrame
    ttest_results_df = pd.DataFrame(ttest_results_list)
    
    # Optional: Apply multiple comparisons correction (e.g., Bonferroni)
    from statsmodels.stats.multitest import multipletests
    ttest_results_df['p_adj_bonferroni'] = multipletests(ttest_results_df['p_val'], method='bonferroni')[1]
    
    # Display t-test results
    print("=== Summary of All Paired t-Tests ===")
    print(ttest_results_df.to_string(index=False))
    
# ===========================
# Step 8: Visualization
# ===========================

# For visualization, we'll plot the mean scores with 95% confidence intervals for each measure and condition

# Calculate summary statistics
summary_stats = anova_data.groupby(['measure', 'type']).agg(
    Mean=('score', 'mean'),
    SEM=('score', lambda x: stats.sem(x, nan_policy='omit'))
).reset_index()

# Calculate 95% Confidence Intervals
summary_stats['CI'] = 1.96 * summary_stats['SEM']

# Print summary statistics for verification
print("\n=== Summary Statistics for Visualization ===")
print(summary_stats.to_string(index=False))

# Set up the plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create a barplot without error bars
ax = sns.barplot(
    x='measure',
    y='Mean',
    hue='type',
    data=summary_stats,
    palette='Set1',
    ci=None,  # We'll add custom error bars
    capsize=0.05,
    errwidth=1
)

# Add error bars manually by iterating over each bar
for i, bar in enumerate(ax.patches):
    # Determine the corresponding row in summary_stats
    # Seaborn plots bars in the order of hue within each x-category
    # Calculate measure and type based on the index
    measure_index = i // 2  # Assuming two conditions per measure
    condition_type = 'Non-Distractor' if i % 2 == 0 else 'Distractor'
    
    # Retrieve the corresponding measure
    measure = unique_measures[measure_index]
    
    # Retrieve the corresponding CI value
    ci = summary_stats[
        (summary_stats['measure'] == measure) & 
        (summary_stats['type'] == condition_type)
    ]['CI'].values[0]
    
    # Get the center of the bar
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    
    # Plot the error bar
    ax.errorbar(
        x=x,
        y=y,
        yerr=ci,
        fmt='none',
        c='black',
        capsize=5
    )

# Enhance plot aesthetics
plt.title('Comparison of Non-Distractor vs Distractor Conditions (with 95% CI)')
plt.xlabel('Measure')
plt.ylabel('Mean Score')
plt.xticks(rotation=45)
plt.legend(title='Condition')
plt.tight_layout()
plt.show()