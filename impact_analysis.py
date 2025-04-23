import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("Sparx Impact Analysis")
print("====================\n")

# Load the data files
assessment_df = pd.read_csv('task_qla_df.csv')
homework_df = pd.read_feather('task_tia_df.feather')

print("Data loaded successfully:")
print(f"- Assessment data: {assessment_df.shape[0]} records, {assessment_df.shape[1]} columns")
print(f"- Homework data: {homework_df.shape[0]} records, {homework_df.shape[1]} columns")

# Clean the data: remove records where student was absent for assessment
assessment_df = assessment_df[~assessment_df['absent']]

# Normalize assessment scores to percentages
assessment_df['score_pct'] = (assessment_df['mark'] / assessment_df['available_marks']) * 100

# Process the summary column to extract information about student activity
# Based on summary column values:
# - C: Correct answer
# - W: Wrong answer
# - A: Abandon/attempt
# - V: Viewed solution/help

# Add columns for completion and correctness
homework_df['is_correct'] = homework_df['summary'].str.contains('C', na=False).astype(int)
homework_df['is_wrong'] = homework_df['summary'].str.contains('W', na=False).astype(int)
homework_df['viewed_help'] = homework_df['summary'].str.contains('V', na=False).astype(int)
homework_df['abandoned'] = homework_df['summary'].str.contains('A', na=False).astype(int)
homework_df['attempted'] = homework_df['is_correct'] + homework_df['is_wrong'] + homework_df['abandoned']

# Aggregate homework activity by student
homework_metrics = homework_df.groupby('student_id').agg(
    homework_count=('package_id', 'count'),  # Total homework activities
    correct_tasks=('is_correct', 'sum'),     # Number of correct tasks
    wrong_attempts=('is_wrong', 'sum'),      # Number of wrong attempts
    help_views=('viewed_help', 'sum'),       # Number of times help was viewed
    abandonments=('abandoned', 'sum'),       # Number of abandonments
    total_attempts=('attempted', 'sum')      # Total attempts made
).reset_index()

# Calculate completion and accuracy rates
homework_metrics['completion_rate'] = homework_metrics['correct_tasks'] / homework_metrics['homework_count'] * 100
homework_metrics['accuracy_rate'] = homework_metrics['correct_tasks'] / (homework_metrics['correct_tasks'] + homework_metrics['wrong_attempts']) * 100
homework_metrics['accuracy_rate'] = homework_metrics['accuracy_rate'].fillna(0)  # Handle division by zero

# Get average assessment score by student
assessment_metrics = assessment_df.groupby('student_id').agg(
    avg_score=('score_pct', 'mean'),
    assessment_count=('assessment_id', 'count')
).reset_index()

# Merge the homework and assessment metrics
merged_metrics = pd.merge(assessment_metrics, homework_metrics, on='student_id', how='inner')

# Filter to students with sufficient data (at least 2 assessments and 5 homework activities)
filtered_metrics = merged_metrics[(merged_metrics['assessment_count'] >= 2) & 
                                 (merged_metrics['homework_count'] >= 5)]

print(f"\nAnalysis based on {len(filtered_metrics)} students with sufficient data")

# ===== Analysis 1: Correlation Analysis =====
print("\n1. Correlation Analysis:")
print("-------------------------")

# Calculate correlations between homework metrics and assessment scores
correlations = filtered_metrics[['avg_score', 'homework_count', 'correct_tasks',
                               'accuracy_rate', 'completion_rate']].corr()

print("Correlation between homework metrics and assessment scores:")
for metric in ['homework_count', 'correct_tasks', 'accuracy_rate', 'completion_rate']:
    corr = correlations.loc['avg_score', metric]
    print(f"- {metric}: {corr:.4f}")

# ===== Analysis 2: Regression Analysis =====
print("\n2. Regression Analysis:")
print("----------------------")

# Prepare data for regression
X = filtered_metrics[['homework_count', 'accuracy_rate', 'completion_rate']]
X = sm.add_constant(X)  # Add constant term
y = filtered_metrics['avg_score']

# Fit regression model
model = sm.OLS(y, X).fit()

print(f"R-squared: {model.rsquared:.4f}")
print("Key coefficients:")
for var, coef in zip(['constant', 'homework_count', 'accuracy_rate', 'completion_rate'], model.params):
    print(f"- {var}: {coef:.4f}")

# ===== Analysis 3: High vs. Low Engagement Analysis =====
print("\n3. High vs. Low Engagement Analysis:")
print("----------------------------------")

# Define high and low engagement groups based on correct tasks (median split)
median_correct = filtered_metrics['correct_tasks'].median()
filtered_metrics['engagement_group'] = np.where(
    filtered_metrics['correct_tasks'] >= median_correct, 'High Engagement', 'Low Engagement')

# Compare assessment scores between groups
high_engagement = filtered_metrics[filtered_metrics['engagement_group'] == 'High Engagement']['avg_score']
low_engagement = filtered_metrics[filtered_metrics['engagement_group'] == 'Low Engagement']['avg_score']

print(f"Average score - High Engagement group: {high_engagement.mean():.2f}%")
print(f"Average score - Low Engagement group: {low_engagement.mean():.2f}%")
print(f"Difference: {high_engagement.mean() - low_engagement.mean():.2f}%")

# Perform t-test to check if difference is significant
t_stat, p_value = stats.ttest_ind(high_engagement, low_engagement)
print(f"T-test p-value: {p_value:.6f}")
print(f"Statistical significance: {'Significant' if p_value < 0.05 else 'Not significant'}")

# Calculate effect size (Cohen's d)
pooled_std = np.sqrt((high_engagement.std()**2 + low_engagement.std()**2) / 2)
cohens_d = (high_engagement.mean() - low_engagement.mean()) / pooled_std

print(f"Effect size (Cohen's d): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect = "negligible"
elif abs(cohens_d) < 0.5:
    effect = "small"
elif abs(cohens_d) < 0.8:
    effect = "medium"
else:
    effect = "large"
print(f"Effect magnitude: {effect}")

# ===== Create visualization for non-technical audience =====
print("\nCreating visualization for non-technical audience...")

plt.figure(figsize=(10, 6))

# Bar plot comparing high vs low engagement groups
sns.barplot(x='engagement_group', y='avg_score', data=filtered_metrics, palette='viridis')
plt.title('Assessment Scores: High vs. Low Homework Engagement', fontsize=16)
plt.xlabel('Sparx Homework Engagement Level', fontsize=14)
plt.ylabel('Average Assessment Score (%)', fontsize=14)
plt.ylim(0, 100)  # Set y-axis to percentage scale

# Add the actual values on top of the bars
for i, group in enumerate(['Low Engagement', 'High Engagement']):
    avg = filtered_metrics[filtered_metrics['engagement_group'] == group]['avg_score'].mean()
    plt.text(i, avg + 2, f"{avg:.1f}%", ha='center', fontsize=14, fontweight='bold')

# Add annotation about statistical significance
if p_value < 0.05:
    plt.annotate(f"Statistically significant difference (p={p_value:.3f})",
                xy=(0.5, 10), xycoords='axes fraction',
                ha='center', fontsize=12, fontweight='bold')
else:
    plt.annotate(f"Difference not statistically significant (p={p_value:.3f})",
                xy=(0.5, 10), xycoords='axes fraction',
                ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('sparx_impact.png', dpi=300, bbox_inches='tight')

# ===== Final Answer for Product Lead =====
print("\n===== Final Answer for Product Lead =====")
print("Based on our analysis of {0} students, we {1} confidently state that \"Using Sparx improves students' exam performance.\"".format(
    len(filtered_metrics),
    "can" if p_value < 0.05 and cohens_d > 0.2 else "cannot"
))

print("\nSummary of findings:")
print(f"- Students with higher Sparx engagement scored {high_engagement.mean() - low_engagement.mean():.1f}% higher on assessments")
print(f"- This difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}")
print(f"- The effect size is {effect} (Cohen's d = {cohens_d:.2f})")

print("\nKey assumptions:")
print("1. We defined engagement based on the number of correct homework tasks")
print("2. We only included students with at least 2 assessments and 5 homework activities")
print("3. We used a median split to create high/low engagement groups")
print("4. We assumed that other factors (like prior ability) don't explain the relationship")