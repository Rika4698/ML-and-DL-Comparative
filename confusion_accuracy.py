# import matplotlib.pyplot as plt

# # Accuracy values for each algorithm
# accuracies = [98.88, 98.88, 98.92, 98.83, 98.83]
# algorithms = ['Logistic Regression', 'Decision Tree', 'SVM (Linear Kernel)', 'SVM (RBF Kernel)', 'Random Forest']

# # Create the pie chart without percentage labels
# plt.figure(figsize=(6, 6))
# plt.pie(accuracies, labels=algorithms, autopct=None, startangle=90, colors=['lightblue', 'cornflowerblue', 'gray', 'lightgreen', 'green'], 
#         wedgeprops={'edgecolor': 'black'})

# # Add a legend outside the plot (without percentages)
# plt.legend(algorithms, loc='upper left', bbox_to_anchor=(1, 0.8), title="Algorithms")

# # Display the plot
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt

# Confusion matrix accuracy values for each model (in percentage)
models = ['Logistic Regression', 'Decision Tree', 'SVM (Linear Kernel)', 'SVM (RBF Kernel)', 'Random Forest']
conf_matrix_accuracy = [0.96688, 0.98436, 0.96780, 0.97148, 0.98252]  # Replace these with actual values from your results

# Colors for the pie chart
colors = ['lightblue', 'cornflowerblue', 'gray', 'lightgreen', 'lightcoral']

# Create a pie chart
plt.figure(figsize=(5, 5))
plt.pie(conf_matrix_accuracy,  autopct=lambda p: '{:.5f}'.format(p * sum(conf_matrix_accuracy) / 100), startangle=90, colors=colors, 
        textprops={'fontsize': 12, "weight":"bold"})

plt.legend(models, loc='upper left', bbox_to_anchor=(-0.1, 0.1))
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')  

# Title
plt.title('Confusion Matrix Accuracy of ML Algorithms', fontsize=18,pad=24, fontweight='bold',color="grey",fontfamily="monospace")

# Show the pie chart
plt.show()