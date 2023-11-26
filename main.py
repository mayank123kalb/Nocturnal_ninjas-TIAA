# import pandas as pd
#
# df = pd.read_excel('investment_dataset_strategies.xlsx')
#
# # print(df['Investment Strategy'].unique())
#
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# #
# # # Assuming 'df' is your DataFrame
# # # If you haven't loaded your data yet, you can use your provided data like this:
# # # df = pd.DataFrame({
# # #     'Age': [...],
# # #     'Income': [...],
# # #     'Investment Strategy': [...],
# # #     'Financial Goals': [...],
# # #     'Investment Amount': [...]
# # # })
# #
# # # Count plot for 'Investment Strategy'
# # plt.figure(figsize=(10, 6))
# # sns.countplot(x='Investment Strategy', data=df)
# # plt.title('Count of Investment Strategies')
# # plt.show()
# #
# # # Box plot for numerical variables by 'Investment Strategy'
# # plt.figure(figsize=(12, 8))
# # sns.boxplot(x='Investment Strategy', y='Age', data=df)
# # plt.title('Age Distribution by Investment Strategy')
# # plt.show()
# #
# # plt.figure(figsize=(12, 8))
# # sns.boxplot(x='Investment Strategy', y='Income', data=df)
# # plt.title('Income Distribution by Investment Strategy')
# # plt.show()
# #
# # # Violin plot for numerical variables by 'Investment Strategy'
# # plt.figure(figsize=(12, 8))
# # sns.violinplot(x='Investment Strategy', y='Investment Amount', data=df)
# # plt.title('Investment Amount Distribution by Investment Strategy')
# # plt.show()
#
# #
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# #
# # # Assuming 'df' is your DataFrame
# # # If you haven't loaded your data yet, you can use your provided data like this:
# # # df = pd.DataFrame({
# # #     'Age': [...],
# # #     'Income': [...],
# # #     'Investment Strategy': [...],
# # #     'Financial Goals': [...],
# # #     'Investment Amount': [...]
# # # })
# #
# # # Create a cross-tabulation of 'Investment Strategy' and another categorical variable
# # cross_tab = pd.crosstab(df['Investment Strategy'], df['Financial Goals'])
# #
# # # Plot a heatmap
# # plt.figure(figsize=(12, 8))
# # sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlGnBu')
# # plt.title('Investment Strategy vs. Financial Goals')
# # plt.show()
#
#
# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# #
# # # Assuming 'df' is your DataFrame
# # # If you haven't loaded your data yet, you can use your provided data like this:
# # # df = pd.DataFrame({
# # #     'Age': [...],
# # #     'Income': [...],
# # #     'Investment Strategy': [...],
# # #     'Financial Goals': [...],
# # #     'Investment Amount': [...]
# # # })
# #
# # # Check the distribution of 'Investment Strategy'
# # class_distribution = df['Investment Strategy'].value_counts()
# #
# # # Plot a bar plot
# # plt.figure(figsize=(8, 6))
# # sns.barplot(x=class_distribution.index, y=class_distribution.values)
# # plt.title('Investment Strategy Distribution')
# # plt.xlabel('Investment Strategy')
# # plt.ylabel('Count')
# # plt.show()
# #
# # print("Class Distribution:")
# # print(class_distribution)
# import matplotlib.pyplot as plt
# #
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
#
# # Load the dataset
# df = pd.read_excel('investment_dataset_strategies.xlsx')
#
# # Separate features and target variable
# X = df.drop('Investment Strategy', axis=1)
# y = df['Investment Strategy']
#
# # Convert categorical classes to numeric labels using LabelEncoder
# label_encoder = LabelEncoder()
# y_numeric = label_encoder.fit_transform(y)
#
# # Convert categorical variables to numerical using one-hot encoding
# X_encoded = pd.get_dummies(X, columns=['Financial Goals'])
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_numeric, test_size=0.2, random_state=42)
#
# # Apply SMOTE for oversampling
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
#
# # Initialize the Random Forest model
# model_rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
# model_rf.fit(X_resampled, y_resampled)
# y_pred_rf = model_rf.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
# print("Random Forest Classification Report:")
# print(classification_report(y_test, y_pred_rf, zero_division=1))  # Set zero_division to 1 or 'warn'
#
# import joblib
#
# # Save the trained Random Forest model as a pickle file
# joblib.dump(model_rf, 'random_forest_model.pkl')
# print("Random Forest Model saved as random_forest_model.pkl")
#
# #
# # # Initialize the XGBoost model
# # model_xgb = XGBClassifier(random_state=42, n_estimators=100, max_depth=5)
# # model_xgb.fit(X_train, y_train)
# # y_pred_xgb = model_xgb.predict(X_test)
# # accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
# # print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")
# # print("XGBoost Classification Report:")
# # print(classification_report(y_test, y_pred_xgb, zero_division=1))  # Set zero_division to 1 or 'warn'
#
# # # Apply K-Means clustering
# # kmeans_model = KMeans(n_clusters=4, random_state=42)
# # clusters = kmeans_model.fit_predict(X_encoded)
# # pca = PCA(n_components=2)
# # X_pca = pca.fit_transform(X_encoded)
# # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
# # plt.title('K-Means Clustering')
# # plt.show()
#
# # # Apply SVM
# # svm_model = SVC(random_state=42)
# # svm_model.fit(X_train, y_train)
# # y_pred_svm = svm_model.predict(X_test)
# # accuracy_svm = accuracy_score(y_test, y_pred_svm)
# # print(f"SVM Accuracy: {accuracy_svm:.4f}")
# # print("SVM Classification Report:")
# # print(classification_report(y_test, y_pred_svm, zero_division=1))  # Set zero_division to 1 or 'warn'
#
#
# # from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# # from sklearn.svm import SVC
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score
# #
# # # Assuming you have X_train, X_test, y_train, y_test defined
# #
# # # Create individual classifiers
# # rf_classifier = RandomForestClassifier(random_state=42)
# # svm_classifier = SVC(probability=True, random_state=42)
# #
# # # Create a VotingClassifier
# # voting_clf = VotingClassifier(estimators=[
# #     ('rf', rf_classifier),
# #     ('svm', svm_classifier)
# # ], voting='soft')
# #
# # # Train the VotingClassifier
# # voting_clf.fit(X_train, y_train)
# #
# # # Make predictions
# # y_pred = voting_clf.predict(X_test)
# #
# # # Calculate accuracy
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f"Accuracy: {accuracy:.4f}")
#
#
# # import joblib
# #
# # # Save the trained model as a pickle file
# # joblib.dump(voting_clf, 'investment_strategy_model.pkl')
# #
# # print("Model saved as investment_strategy_model.pkl")
#

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_excel('investment_dataset_strategies.xlsx')

# Separate features and target variable
X = df.drop('Investment Strategy', axis=1)
y = df['Investment Strategy']

# Convert categorical classes to numeric labels using LabelEncoder
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)

# Convert categorical variables to numerical using one-hot encoding
X_encoded = pd.get_dummies(X, columns=['Financial Goals'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_numeric, test_size=0.2, random_state=42)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Initialize the Random Forest model
model_rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
model_rf.fit(X_resampled, y_resampled)

# Save the trained model as a pickle file
joblib.dump(model_rf, 'investment_strategy_model.pkl')

# Load the trained model
model = joblib.load('investment_strategy_model.pkl')

# Ensure that the user input has the same features as the training data
user_input = pd.DataFrame({
    'Age': [35],
    'Income': [80000],
    'Investment Amount': [50000],
    'Financial Goals_Home Ownership': [1],
    # Add other Financial Goals columns with values as 0
})

# Add missing columns in user_input that are present in X_encoded
missing_columns = set(X_encoded.columns) - set(user_input.columns)
for column in missing_columns:
    user_input[column] = 0

# Reorder columns to match the model's feature order
user_input = user_input[X_encoded.columns]

# Make a prediction using the loaded model
prediction = model.predict(user_input)

# Decode the numeric prediction back to the original class
predicted_class = label_encoder.inverse_transform(prediction)

# Print the predicted class
print(f"The predicted class is: {predicted_class[0]}")





