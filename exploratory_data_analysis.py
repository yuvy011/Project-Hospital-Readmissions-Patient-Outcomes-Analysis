"""
Diabetes 130-US Hospitals - Exploratory Data Analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 10

df = pd.read_csv('diabetic_data_cleaned.csv')
print(f"Loaded {len(df)} rows")

# Helper: readmission rate by group.
def readmit_rate_by(col, order=None):
    g = df.groupby(col)['readmitted_30d'].agg(['mean', 'count']).reset_index()
    g['mean'] = (g['mean'] * 100).round(1)
    if order:
        g[col] = pd.Categorical(g[col], categories=order, ordered=True)
        g = g.sort_values(col)
    return g

# 1. Readmission rate by age group 
age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
age_stats = readmit_rate_by('age', order=age_order)
print("\n=== Readmission rate (%) by age group ===")
print(age_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(age_stats['age'], age_stats['mean'], color='#4C72B0')
ax.set_title('30-Day Readmission Rate by Age Group', fontweight='bold')
ax.set_ylabel('Readmission Rate (%)')
ax.set_xlabel('Age Group')
plt.xticks(rotation=45)
for bar, n in zip(bars, age_stats['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'n={n}', ha='center', fontsize=7, color='gray')
plt.tight_layout()
plt.savefig('chart_readmit_by_age.png')
plt.close()

# 2. Readmission rate by primary diagnosis category
diag_stats = readmit_rate_by('diag_1_category')
diag_stats = diag_stats.sort_values('mean', ascending=False)
print("\n=== Readmission rate (%) by primary diagnosis category ===")
print(diag_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(diag_stats['diag_1_category'], diag_stats['mean'], color='#55A868')
ax.set_title('30-Day Readmission Rate by Primary Diagnosis', fontweight='bold')
ax.set_xlabel('Readmission Rate (%)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('chart_readmit_by_diagnosis.png')
plt.close()

# 3. Readmission rate by number of prior inpatient visits
df['prior_inpatient_bucket'] = pd.cut(
    df['number_inpatient'], bins=[-1, 0, 1, 2, 100],
    labels=['0', '1', '2', '3+']
)
prior_stats = readmit_rate_by('prior_inpatient_bucket')
print("\n=== Readmission rate (%) by prior inpatient visit count ===")
print(prior_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(prior_stats['prior_inpatient_bucket'].astype(str), prior_stats['mean'], color='#C44E52')
ax.set_title('30-Day Readmission Rate by Prior Inpatient Visits', fontweight='bold')
ax.set_ylabel('Readmission Rate (%)')
ax.set_xlabel('Number of Prior Inpatient Visits')
for bar, n in zip(bars, prior_stats['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'n={n}', ha='center', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('chart_readmit_by_prior_visits.png')
plt.close()

# 4. Readmission rate by time in hospital
time_stats = readmit_rate_by('time_in_hospital')
print("\n=== Readmission rate (%) by time in hospital (days) ===")
print(time_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(time_stats['time_in_hospital'], time_stats['mean'], marker='o', color='#8172B2')
ax.set_title('30-Day Readmission Rate by Length of Stay', fontweight='bold')
ax.set_ylabel('Readmission Rate (%)')
ax.set_xlabel('Time in Hospital (days)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chart_readmit_by_time_in_hospital.png')
plt.close()

print("\nSaved 4 charts: chart_readmit_by_age.png, chart_readmit_by_diagnosis.png, "
      "chart_readmit_by_prior_visits.png, chart_readmit_by_time_in_hospital.png")