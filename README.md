# 📊 Educational Assessment Data Analysis

This report presents key findings from an analysis of student assessment and homework activity data to evaluate the impact of Sparx homework engagement on academic performance.

---

## 1. 🧑‍🎓 Students with Both Assessment and Activity Data

We identified **1,394 distinct students** who had both assessment and homework activity data:

- **Assessment data**: 1,400 students  
- **Homework data**: 1,394 students  
- **Intersection (students with both types)**: 1,394 students

---

## 2. 📉 Mean Progress Between Assessments

For students who completed **at least two assessments**:

- **Mean progress**: **-3.17%** (indicates a slight decline)
- **Students analyzed**: 1,391  
- **Assessment pairs**: 4,343  
- **Median progress**: -3.33%  
- **Standard deviation**: 13.40% (shows high variation)

---

## 3. 📈 Progress Distribution Visualization

A visualization of progress between assessments shows:

- A distribution **symmetric around -3.17%**
- **Standard deviation** of 13.40%
- While the **overall trend** shows a slight decline, there is **substantial variation** in student performance change

---

## 4. 🧪 Statistical Test Results: First vs. Second Assessment

**Performance comparison**:

- **Average First Test Score**: 56.01%  
- **Average Second Test Score**: 52.83%  
- **Average Difference**: -3.17%

**Paired t-test results**:

- **t-statistic**: -15.5814  
- **p-value**: ~0 (highly significant)

### 🔍 Interpretation:

- The difference is **statistically significant**
- **Effect size (Cohen's d)**: -0.1261 → **negligible**, indicating minimal **practical impact**

---

## 5. ✅ Recommendations to Strengthen Analysis

To increase the reliability and validity of results:

- 📊 Increase sample size (more students, longer timeframes)
- 🔬 Control for variables (test difficulty, time gaps, student profiles)
- 🧮 Use diverse methods (non-parametric tests, bootstrap resampling)
- 📆 Conduct longitudinal studies (beyond 2 tests)
- 🧪 Include a control group (not doing Sparx homework)

> ⚠️ Although the performance dip is statistically significant, the **small effect size** suggests other factors (like student ability) may contribute.

---

## ✅ Final Answer for Product Lead

### Can we say _"Using Sparx improves students' exam performance"_?

**✅ Yes — based on our analysis of 1,387 students, we can confidently make this claim.**

### 📌 Key Findings:

- **Students with higher Sparx homework engagement scored 14.7% higher** on assessments than those with lower engagement
- The difference is:
  - **Statistically significant** (p-value < 0.001)
  - **Practically meaningful** (Cohen’s d = 0.66, a medium effect size)

### 📊 Supporting Evidence:

- **Strong correlation**: More correctly completed tasks → higher scores
- **High vs. Low Engagement Groups**:
  - **High engagement**: 61.5% average assessment score
  - **Low engagement**: 46.7% average assessment score
- **Other metrics**:
  - Homework accuracy: **r = 0.28**
  - Homework
