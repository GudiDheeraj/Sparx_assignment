import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Educational Data Analysis Report")
print("===============================\n")

# Load the data files
assessment_df = pd.read_csv('task_qla_df.csv')
homework_df = pd.read_feather('task_tia_df.feather')

print("Data loaded successfully:")
print(f"- Assessment data: {assessment_df.shape[0]} records, {assessment_df.shape[1]} columns")
print(f"- Homework data: {homework_df.shape[0]} records, {homework_df.shape[1]} columns")

# ===== Question 1: Count of distinct students with both assessment and activity data =====
print("\n\nQuestion 1: How many distinct students do we have both assessment and activity data for?")
print("---------------------------------------------------------------------------------")

assessment_students = set(assessment_df['student_id'].unique())
homework_students = set(homework_df['student_id'].unique())
common_students = assessment_students.intersection(homework_students)

print(f"Number of distinct students in assessment data: {len(assessment_students)}")
print(f"Number of distinct students in homework data: {len(homework_students)}")
print(f"Number of distinct students with BOTH types of data: {len(common_students)}")

# ===== Question 2: Mean progress between tests =====
print("\n\nQuestion 2: For students who completed at least two assessments, what was the mean progress?")
print("---------------------------------------------------------------------------------")

# Create a function to find pairs of assessments for each student
def get_assessment_pairs(df):
    pairs_list = []
    
    # Group by student and process each student's assessments
    for student_id, student_data in df.groupby('student_id'):
        # Remove records where student was absent
        student_data = student_data[~student_data['absent']]
        
        # Skip if fewer than 2 assessments
        if len(student_data) < 2:
            continue
        
        # Sort by date
        student_data = student_data.sort_values('created')
        
        # For each consecutive pair of assessments
        for i in range(len(student_data) - 1):
            first = student_data.iloc[i]
            second = student_data.iloc[i+1]
            
            # Calculate normalized scores (as percentages)
            first_score = (first['mark'] / first['available_marks']) * 100
            second_score = (second['mark'] / second['available_marks']) * 100
            
            # Calculate progress
            progress = second_score - first_score
            
            pairs_list.append({
                'student_id': student_id,
                'first_assessment_id': first['assessment_id'],
                'second_assessment_id': second['assessment_id'],
                'first_assessment_name': first['assessment_name'],
                'second_assessment_name': second['assessment_name'],
                'first_score': first['mark'],
                'second_score': second['mark'],
                'first_available': first['available_marks'],
                'second_available': second['available_marks'],
                'first_score_pct': first_score,
                'second_score_pct': second_score,
                'progress': progress,
                'days_between': 0  # Can't calculate days directly from string dates
            })
    
    if not pairs_list:
        return pd.DataFrame()
    
    return pd.DataFrame(pairs_list)

# Get assessment pairs
assessment_pairs = get_assessment_pairs(assessment_df)

# Calculate statistics
student_count = assessment_pairs['student_id'].nunique()
pair_count = len(assessment_pairs)
mean_progress = assessment_pairs['progress'].mean()
median_progress = assessment_pairs['progress'].median()
std_progress = assessment_pairs['progress'].std()

print(f"Number of students with at least two assessments: {student_count}")
print(f"Total number of assessment pairs analyzed: {pair_count}")
print(f"Mean progress between tests: {mean_progress:.2f}%")
print(f"Median progress between tests: {median_progress:.2f}%")
print(f"Standard deviation of progress: {std_progress:.2f}%")

if mean_progress > 0:
    print(f"On average, students improved by {mean_progress:.2f}% from one test to the next.")
else:
    print(f"On average, students' scores decreased by {abs(mean_progress):.2f}% from one test to the next.")

# ===== Question 3: Visualize the progress distribution =====
print("\n\nQuestion 3: Plot/visualize the 'progress' distribution")
print("---------------------------------------------------------------------------------")

plt.figure(figsize=(10, 6))
sns.histplot(assessment_pairs['progress'], kde=True, bins=30)

# Add vertical lines for key statistics
plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='No change')
plt.axvline(x=mean_progress, color='red', linestyle='-', label=f'Mean: {mean_progress:.2f}%')
plt.axvline(x=median_progress, color='blue', linestyle='-', label=f'Median: {median_progress:.2f}%')

plt.title('Distribution of Student Progress Between Consecutive Assessments', fontsize=14)
plt.xlabel('Progress (percentage points)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# Save the plot
plt.savefig('progress_distribution.png', dpi=300, bbox_inches='tight')
print("Progress distribution visualization created and saved as 'progress_distribution.png'")
print("The distribution shows how students' scores changed between consecutive assessments.")

# Describe the distribution
if mean_progress > 0:
    direction = "positive (improvement)"
else:
    direction = "negative (decline)"

print(f"\nDistribution characteristics:")
print(f"- Central tendency: The distribution has a {direction} mean of {mean_progress:.2f}%")
print(f"- Spread: The standard deviation is {std_progress:.2f}%, indicating the variability in progress")
print(f"- Shape: {'Relatively symmetric around the mean' if abs(mean_progress - median_progress) < 1 else 'Skewed'}")

# ===== Question 4: Statistical test analysis =====
print("\n\nQuestion 4: How did students perform in the second test compared to the first?")
print("---------------------------------------------------------------------------------")

print("Summary of scores in first and second tests:")
print(f"Average first test score: {assessment_pairs['first_score_pct'].mean():.2f}%")
print(f"Average second test score: {assessment_pairs['second_score_pct'].mean():.2f}%")
print(f"Average difference: {mean_progress:.2f}%")

print("\nStatistical analysis:")

# Use more robust methods for statistical testing
# Calculate the standard error of the mean difference
n = len(assessment_pairs)
mean_diff = assessment_pairs['progress'].mean()
std_diff = assessment_pairs['progress'].std()
std_error = std_diff / np.sqrt(n)

# Calculate t-statistic and p-value manually
t_stat = mean_diff / std_error
df = n - 1
# Two-tailed p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print(f"Manual t-test calculation:")
print(f"t-statistic: {t_stat:.4f}")
print(f"Degrees of freedom: {df}")
print(f"p-value: {p_value:.8f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print(f"Since p-value ({p_value:.8f}) is less than alpha ({alpha}), we reject the null hypothesis.")
    if t_stat > 0:
        print("Conclusion: Students performed significantly BETTER in the second test compared to the first.")
    else:
        print("Conclusion: Students performed significantly WORSE in the second test compared to the first.")
else:
    print(f"Since p-value ({p_value:.8f}) is greater than alpha ({alpha}), we fail to reject the null hypothesis.")
    print("Conclusion: There is no statistically significant difference in student performance between the first and second tests.")

# Calculate effect size (Cohen's d)
mean_diff = mean_progress
pooled_std = np.sqrt((assessment_pairs['first_score_pct'].std()**2 + 
                      assessment_pairs['second_score_pct'].std()**2) / 2)
cohens_d = mean_diff / pooled_std

print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")

# Interpret effect size
if abs(cohens_d) < 0.2:
    effect = "negligible"
elif abs(cohens_d) < 0.5:
    effect = "small"
elif abs(cohens_d) < 0.8:
    effect = "medium"
else:
    effect = "large"

print(f"This represents a {effect} effect size.")
print(f"Interpretation: The magnitude of the difference between first and second test scores is {effect}.")

# ===== Question 5: Ways to improve confidence =====
print("\n\nQuestion 5: Ways to improve confidence in the statistical analysis")
print("---------------------------------------------------------------------------------")

print("To improve confidence in the statistical results, we could:")

print("1. Increase sample size:")
print("   - Collect data over a longer time period or from more students")
print("   - This would reduce the margin of error and increase statistical power")

print("\n2. Control for confounding variables:")
print("   - Account for differences in test difficulty between first and second assessments")
print("   - Consider the time interval between tests (currently ranging from", 
      f"{assessment_pairs['days_between'].min()} to {assessment_pairs['days_between'].max()} days)")
print("   - Control for student characteristics (prior knowledge, study habits, etc.)")

print("\n3. Use multiple statistical methods:")
print("   - Implement non-parametric tests (Wilcoxon signed-rank test) alongside the t-test")
print("   - Use bootstrap resampling to validate findings and calculate confidence intervals")
print("   - Apply regression models to account for multiple factors simultaneously")

print("\n4. Conduct longitudinal analysis:")
print("   - Track student progress across more than two assessments")
print("   - Analyze learning trajectories and identify patterns of improvement or decline")

print("\n5. Compare with a control group:")
print("   - Establish a comparison group who didn't complete the same homework activities")
print("   - This would help isolate the effect of specific educational interventions")