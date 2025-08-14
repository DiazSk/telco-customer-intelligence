# Save as scripts/visualize_results.py
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/processed/customer_segments.csv")

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Risk distribution
df["risk_segment"].value_counts().plot(
    kind="bar", ax=axes[0], color=["green", "yellow", "orange", "red"]
)
axes[0].set_title("Customer Risk Distribution")
axes[0].set_xlabel("Risk Segment")
axes[0].set_ylabel("Number of Customers")

# Churn probability distribution
axes[1].hist(df["churn_probability"], bins=30, edgecolor="black")
axes[1].axvline(0.2, color="red", linestyle="--", label="Intervention Threshold (20%)")
axes[1].set_title("Churn Probability Distribution")
axes[1].set_xlabel("Churn Probability")
axes[1].set_ylabel("Number of Customers")
axes[1].legend()

plt.tight_layout()
plt.savefig("docs/model_results.png", dpi=150, bbox_inches="tight")
print("Visualization saved to docs/model_results.png")
plt.show()
