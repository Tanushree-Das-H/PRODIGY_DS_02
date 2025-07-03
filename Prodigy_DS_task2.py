import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("train.csv")

# Clean data
df.drop('Cabin', axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop_duplicates(inplace=True)

# Prepare data for heatmap
df_corr = df.copy()
df_corr['Sex'] = df_corr['Sex'].map({'male': 0, 'female': 1})
df_corr['Embarked'] = df_corr['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df_corr.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 10))

# 1. Survival Count
sns.countplot(x='Survived', data=df, ax=axs[0, 0])
axs[0, 0].set_title('Survival Distribution', pad=10)
axs[0, 0].set_xticklabels(['Not Survived', 'Survived'])

# 2. Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df, ax=axs[0, 1])
axs[0, 1].set_title('Survival by Gender', pad=10)

# 3. Age Distribution
sns.histplot(df['Age'], bins=30, kde=True, ax=axs[1, 0])
axs[1, 0].set_title('Age Distribution of Passengers', pad=10)

# 4. Survival by Class
sns.countplot(x='Pclass', hue='Survived', data=df, ax=axs[1, 1])
axs[1, 1].set_title('Survival by Passenger Class', pad=10)

# 5. Fare vs Survival
sns.boxplot(x='Survived', y='Fare', data=df, ax=axs[2, 0])
axs[2, 0].set_title('Fare Paid vs Survival', pad=10)

# 6. Correlation Heatmap
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', ax=axs[2, 1])
axs[2, 1].set_title('Correlation Heatmap', pad=10)

# Improve spacing and layout
plt.tight_layout(pad=2.5)
plt.subplots_adjust(top=0.95)
plt.show()
