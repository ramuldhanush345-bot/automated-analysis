import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import os

print("Script started")

# Check if CSV file argument is provided
if len(sys.argv) < 2:
    print("Please provide a CSV file.")
    sys.exit()

file = sys.argv[1]

# Load dataset
df = pd.read_csv(file)

print("Dataset loaded successfully")
print("\nFirst 5 rows:")
print(df.head())

# Basic summary statistics
print("\nDataset Summary:")
summary = df.describe(include="all")
print(summary)

columns = df.columns.tolist()

missing_values = df.isnull().sum().to_dict()

summary_text = summary.to_string()

# Select numeric columns
numeric_df = df.select_dtypes(include=['number'])

# --- Correlation Heatmap ---
if numeric_df.shape[1] > 1:

    corr = numeric_df.corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")

    plt.title("Correlation Heatmap")
    plt.savefig("correlation.png")
    plt.close()

    print("Correlation chart saved as correlation.png")

else:
    print("Not enough numeric columns for correlation heatmap.")


# --- Distribution Plot ---
if numeric_df.shape[1] > 0:

    numeric_df.hist(figsize=(10,8))

    plt.suptitle("Distribution of Numeric Features")
    plt.savefig("distribution.png")
    plt.close()

    print("Distribution chart saved as distribution.png")

# --- Missing Values Chart ---
df.isnull().sum().plot(kind="bar", color="orange", figsize=(8,5))
plt.title("Missing Values per Column")
plt.ylabel("Count")
plt.savefig("missing.png")
plt.close()

print("Missing values chart saved as missing.png")


categorical_cols = df.select_dtypes(include=['object', 'string']).columns

if len(categorical_cols) > 0:
    col = categorical_cols[0]   # automatically pick first categorical column

    df[col].value_counts().head(10).plot(kind="bar", figsize=(8,5))

    plt.title(f"Top Categories in {col}")
    plt.xlabel(col)
    plt.ylabel("Count")

    plt.savefig("category_counts.png")
    plt.close()

    print("Category count chart saved as category_counts.png")

print("Generating README using LLM...")

prompt = f"""
You are a professional data analyst.

Dataset Columns:
{columns}

Missing Values:
{missing_values}

Summary Statistics:
{summary_text}

Write a markdown report with the following sections:

# Dataset Overview
Explain what the dataset likely contains.

# Analysis Performed
Explain the analyses performed such as correlation, distributions, and category counts.

# Key Insights
Explain important patterns discovered.

# Implications
Explain what decisions or conclusions can be drawn.
"""

token = os.environ.get("AIPROXY_TOKEN")

response = httpx.post(
    "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    },
    json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
)

try:
    result = response.json()

    if "choices" in result:
        report = result["choices"][0]["message"]["content"]
    else:
        print("API Error:", result)
        report = f"""
        # Dataset Overview
        The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.

        # Analysis Performed
        Summary statistics, missing values, correlation, and distributions were analyzed.

        # Key Insights
        The dataset shows variation across features and some missing values.

        # Implications
        The data can be used for further analysis and decision-making.
        """

except Exception as e:
    print("Error:", e)
    report = "LLM failed, fallback used."


with open("README.md", "w", encoding="utf-8") as f:
    f.write(report)

    f.write("\n\n## Visualizations\n")
    f.write("![Correlation](correlation.png)\n")
    f.write("![Distribution](distribution.png)\n")
    f.write("![Category](category_counts.png)\n")
    f.write("![Missing](missing.png)\n")

print("README.md generated successfully")