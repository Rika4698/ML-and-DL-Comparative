# import matplotlib.pyplot as plt

# # Define model names and their corresponding accuracy values
# models = [
#     "Logistic Regression",
#     "Decision Tree",
#     "Support Vector Machine (Linear Kernel)",
#     "Support Vector Machine (RBF Kernel)",
#     "Random Forest",
#     "Recurrent Neural Network (RNN)"
# ]

# # Example accuracy values (replace these with your actual values)
# accuracy_values = [85.5, 80.0, 82.3, 84.1, 88.0, 90.0]  # Replace with your actual accuracy values

# # Create a linear graph
# plt.figure(figsize=(10, 6))
# plt.plot(models, accuracy_values, marker='o', linestyle='-', color='b')

# # Adding titles and labels
# plt.title('Model Accuracy Comparison', fontsize=16)
# plt.xlabel('Models', fontsize=14)
# plt.ylabel('Accuracy (%)', fontsize=14)
# plt.ylim(0, 100)  # Set y-axis limits to 0-100%
# plt.grid(True)

# # Annotate each point with its accuracy value
# for i, v in enumerate(accuracy_values):
#     plt.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10)

# # Show the plot
# plt.xticks(rotation=15)
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt

# Accuracy values for the models (assuming hypothetical or calculated values)
models = ['Recurrent Neural Network (RNN)','Random Forest','Support Vector Machine (RBF Kernel)','Support Vector Machine (Linear Kernel)','Decision Tree','Logistic Regression']
accuracy_values = [ 97.70,98.25,97.15,96.78,98.44,96.69,]  # Example accuracy values in percentage

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.barh(models, accuracy_values, color='skyblue')

# Add labels and title
plt.xlabel('Accuracy (%)')
# plt.ylabel('Model')
plt.title('Comparison between ML and DL results',fontsize=18,pad=24,fontweight='bold', color='grey',fontfamily="serif",loc='center')
plt.xlim(0, 100)
# Remove top and right spines (borders)
ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Annotating the bars with accuracy values
for index, value in enumerate(accuracy_values):
    plt.text(value + 1, index, f'{value}%', va='center')

# Display the plot
plt.tight_layout()
plt.show()