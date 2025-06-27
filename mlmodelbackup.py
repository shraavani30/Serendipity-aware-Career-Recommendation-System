#import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
data= pd.read_csv("stud.csv")
data.shape
# in this we cant see every  column to solve this problem use
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
data.head(6)
data.info()
data.isnull().sum()
label_encoder = LabelEncoder()

# Assuming 'data' is your DataFrame and 'Courses' is a column in it
data['Courses_label'] = label_encoder.fit_transform(data['Courses'])


data['Courses_label'].value_counts()
y=data['Courses_label']
categorical_columns = ['Drawing','Dancing','Singing','Sports','Video Game','Acting','Travelling','Gardening','Animals','Photography','Teaching','Exercise','Coding','Electricity Components','Mechanic Parts','Computer Parts','Researching','Architecture','Historic Collection','Botany','Zoology','Physics','Accounting','Economics','Sociology','Geography','Psycology','History','Science','Bussiness Education','Chemistry','Mathematics','Biology','Makeup','Designing','Content writing','Crafting','Literature','Reading','Cartooning','Debating','Asrtology','Hindi','French','English','Urdu','Other Language','Solving Puzzles','Gymnastics','Yoga','Engeeniering','Doctor','Pharmisist','Cycling','Knitting','Director','Journalism','Bussiness','Listening Music']  # Replace with your categorical columns


label_encoders = {}

for col in categorical_columns:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])
    label_encoders[col] = label_encoder
data.head()
dataab=data.drop(['Courses'],axis=1)
dataab.head()
X=['Drawing','Dancing','Singing','Sports','Video Game','Acting','Travelling','Gardening','Animals','Photography','Teaching','Exercise','Coding','Electricity Components','Mechanic Parts','Computer Parts','Researching','Architecture','Historic Collection','Botany','Zoology','Physics','Accounting','Economics','Sociology','Geography','Psycology','History','Science','Bussiness Education','Chemistry','Mathematics','Biology','Makeup','Designing','Content writing','Crafting','Literature','Reading','Cartooning','Debating','Asrtology','Hindi','French','English','Urdu','Other Language','Solving Puzzles','Gymnastics','Yoga','Engeeniering','Doctor','Pharmisist','Cycling','Knitting','Director','Journalism','Bussiness','Listening Music']

# Create a new DataFrame 'X_df' by selecting columns from the 'dataab' DataFrame based on the 'X' list.
X_df = dataab[X]

# Create the target variable 'Y' by extracting the 'Courses_label' column from the 'dataab' DataFrame.
Y = dataab['Courses_label']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_df, Y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Generate a classification report
# Generate a classification report to provide more detailed evaluation metrics
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)
import joblib

# Assuming you have a trained model object named 'model'
# Save the model to a file
joblib.dump(model , 'model .pkl')
import joblib

# Load the model from the saved file
loaded_model = joblib.load('model .pkl')
user_input = {}
feature_names=['Drawing','Dancing','Singing','Sports','Video Game','Acting','Travelling','Gardening','Animals','Photography','Teaching','Exercise','Coding','Electricity Components','Mechanic Parts','Computer Parts','Researching','Architecture','Historic Collection','Botany','Zoology','Physics','Accounting','Economics','Sociology','Geography','Psycology','History','Science','Bussiness Education','Chemistry','Mathematics','Biology','Makeup','Designing','Content writing','Crafting','Literature','Reading','Cartooning','Debating','Asrtology','Hindi','French','English','Urdu','Other Language','Solving Puzzles','Gymnastics','Yoga','Engeeniering','Doctor','Pharmisist','Cycling','Knitting','Director','Journalism','Bussiness','Listening Music']

# Collect user input for each feature
for feature in feature_names:
    user_value = float(input(f"Enter value for {feature} (0 or 1): "))
    user_input[feature] = user_value

# Create a DataFrame from user input
user_data = pd.DataFrame([user_input])

# Ensure that the user input DataFrame has the same columns as your training data
# Add missing columns with zeros
missing_columns = set(X_train.columns) - set(user_data.columns)
for column in missing_columns:
    user_data[column] = 0  # Add missing columns and set them to 0

# Make a prediction using the model
prediction = model.predict(user_data)

# Define a mapping from numeric values to categories
numeric_to_category = {
    0: 'Animation, Graphics and Multimedia',
    1: 'B.Arch- Bachelor of Architecture',
    2: 'B.Com- Bachelor of Commerce',
    3: 'B.Ed.',
    4: 'B.Sc- Applied Geology',
    5: 'B.Sc- Nursing',
    6: 'B.Sc. Chemistry',
    7: 'B.Sc. Mathematics',
    8: 'B.Sc.- Information Technology',
    9: 'B.Sc.- Physics',
    10: 'B.Tech.-Civil Engineering',
    11: 'B.Tech.-Computer Science and Engineering',
    12: 'B.Tech.-Electrical and Electronics Engineering',
    13: 'B.Tech.-Electronics and Communication Engineering',
    14: 'B.Tech.-Mechanical Engineering',
    15: 'BA in Economics',
    16: 'BA in English',
    17: 'BA in Hindi',
    18: 'BA in History',
    19: 'BBA- Bachelor of Business Administration',
    20: 'BBS- Bachelor of Business Studies',
    21: 'BCA- Bachelor of Computer Applications',
    22: 'BDS- Bachelor of Dental Surgery',
    23: 'BEM- Bachelor of Event Management',
    24: 'BFD- Bachelor of Fashion Designing',
    25: 'BJMC- Bachelor of Journalism and Mass Communication',
    26: 'BPharma- Bachelor of Pharmacy',
    27: 'BTTM- Bachelor of Travel and Tourism Management',
    28: 'BVA- Bachelor of Visual Arts',
    29: 'CA- Chartered Accountancy',
    30: 'CS- Company Secretary',
    31: 'Civil Services',
    32: 'Diploma in Dramatic Arts',
    33: 'Integrated Law Course- BA + LL.B',
    34: 'MBBS'
   }

# Make a prediction using the model
prediction = model.predict(user_data)

# Extract the numeric prediction (assuming prediction is a NumPy array)
numeric_prediction = prediction[0]

# Convert the numeric prediction to a categorical label
if numeric_prediction in numeric_to_category:
    categorical_prediction = numeric_to_category[numeric_prediction]


# Print the categorical prediction
print("i suggest you to go with ", categorical_prediction)
