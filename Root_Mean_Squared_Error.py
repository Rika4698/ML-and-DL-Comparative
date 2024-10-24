import matplotlib.pyplot as plt

# Confusion matrix accuracy values for each model (in percentage)
models = ['Logistic Regression', 'Decision Tree', 'SVM (Linear Kernel)', 'SVM (RBF Kernel)', 'Random Forest']
rmse_values = [0.18199, 0.12506, 0.17944, 0.16888, 0.13221]  # Replace these with actual values from your results

# Colors for the pie chart
colors = ['#90EE90', '#87CEEB', '#9370DB', '#E2A76F', '#FA8072']

# Create a pie chart
plt.figure(figsize=(5,5))
plt.pie(rmse_values,  autopct=lambda p: '{:.5f}'.format(p * sum(rmse_values) / 100), startangle=90, colors=colors, 
        textprops={'fontsize': 10, "weight":"bold"})


plt.legend(models, loc='center right', bbox_to_anchor=(1, -0.1))
# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')  

# Title
plt.title('Root Mean Squared Error (RMSE) of ML Algorithms', fontsize=16,pad=14, fontweight='bold',color="grey",fontfamily="serif")

# Show the pie chart
plt.show()