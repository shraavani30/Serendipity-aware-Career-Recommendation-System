# Import necessary libraries
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("stud.csv")

# Encode target variable (Courses)
label_encoder = LabelEncoder()
data['Courses_label'] = label_encoder.fit_transform(data['Courses'])

# Encode categorical features
categorical_columns = ['Drawing', 'Dancing', 'Singing', 'Sports', 'Video Game', 'Acting', 'Travelling', 'Gardening',
                       'Animals', 'Photography', 'Teaching', 'Exercise', 'Coding', 'Electricity Components',
                       'Mechanic Parts', 'Computer Parts', 'Researching', 'Architecture', 'Historic Collection',
                       'Botany', 'Zoology', 'Physics', 'Accounting', 'Economics', 'Sociology', 'Geography',
                       'Psycology', 'History', 'Science', 'Bussiness Education', 'Chemistry', 'Mathematics',
                       'Biology', 'Makeup', 'Designing', 'Content writing', 'Crafting', 'Literature', 'Reading',
                       'Cartooning', 'Debating', 'Asrtology', 'Hindi', 'French', 'English', 'Urdu', 'Other Language',
                       'Solving Puzzles', 'Gymnastics', 'Yoga', 'Engeeniering', 'Doctor', 'Pharmisist', 'Cycling',
                       'Knitting', 'Director', 'Journalism', 'Bussiness', 'Listening Music']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoders for later use if needed

# Prepare features (X) and target (Y)
X = data[categorical_columns]
Y = data['Courses_label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Completed. Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(model, 'model.pkl')

# Save the feature names for later use in Streamlit app
joblib.dump(categorical_columns, 'features.pkl')

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model, features list, and label encoder saved successfully.")
