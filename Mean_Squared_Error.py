import matplotlib.pyplot as plt

# Confusion matrix accuracy values for each model (in percentage)
models = ['Logistic Regression', 'Decision Tree', 'SVM (Linear Kernel)', 'SVM (RBF Kernel)', 'Random Forest']
mse_values = [0.03312, 0.01564, 0.03220, 0.02852, 0.01748]  # Replace these with actual values from your results

# Colors for the pie chart
colors = ['lightblue', 'lavender', 'gold', 'lightgreen', 'lightcoral']

# Create a pie chart
plt.figure(figsize=(5, 5))
plt.pie(mse_values,  autopct=lambda p: '{:.5f}'.format(p * sum(mse_values) / 100), startangle=90, colors=colors, 
        textprops={'fontsize': 10, "weight":"bold"})

plt.legend(models, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1)
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')  

# Title
plt.title('Mean Squared Error (MSE) of ML Algorithms', fontsize=16,pad=14, fontweight='bold',color="grey",fontfamily="serif")

# Show the pie chart
plt.show()