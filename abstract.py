#!/usr/bin/env python
# coding: utf-8

# In[18]:


import csv
import random
from faker import Faker
fake = Faker()
def generate_scores(student_type):
    if student_type == "topper":
        return [random.randint(85, 100) for _ in range(25)]
    elif student_type == "good_grader":
        return [random.randint(70, 85) for _ in range(25)]
    elif student_type == "average":
        return [random.randint(55, 70) for _ in range(25)]
    elif student_type == "below_average":
        return [random.randint(35, 55) for _ in range(25)]
    elif student_type == "failure":
        return [random.randint(0, 50) for _ in range(25)]

def generate_dataset(num_students):
    data = []
    for i in range(1, num_students + 1):
        student_type = random.choices(["topper", "good_grader", "average", "below_average", "failure"], 
                                      weights=[0.1, 0.25, 0.3, 0.2, 0.15])[0]
        
        
        
        
        row = [f"10VHSS{i}",fake.name()]
        row.extend(generate_scores(student_type))
        data.append(row)
    return data


def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Student ID", "Name", 
                         "Tamil_Midterm1", "Tamil_Midterm2", "Tamil_Half-Yearly", "Tamil_Quarterly", "Tamil_Final",
                         "English_Midterm1", "English_Midterm2", "English_Half-Yearly", "English_Quarterly", "English_Final",
                         "Maths_Midterm1", "Maths_Midterm2", "Maths_Half-Yearly", "Maths_Quarterly", "Maths_Final",
                         "Science_Midterm1", "Science_Midterm2", "Science_Half-Yearly", "Science_Quarterly", "Science_Final",
                         "Social_Midterm1", "Social_Midterm2", "Social_Half-Yearly", "Social_Quarterly", "Social_Final"])
        writer.writerows(data)

num_students = 500

dataset = generate_dataset(num_students)

write_to_csv("students_data.csv", dataset)


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


# In[3]:


data = pd.read_csv("students_data.csv")


# In[4]:


X = data.drop(columns=["Student ID", "Name"])


# In[13]:


y = data["Tamil_Final"]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


rf_model = RandomForestRegressor(n_estimators=20, random_state=42)


# In[16]:


rf_model.fit(X_train, y_train)


# In[17]:


predictions = rf_model.predict(X_test)


# In[ ]:


def get_input_marks():
    print("Enter marks for each subject:")
    marks = []
    subjects = X.columns  # Use the same columns as in X
    for subject in subjects:
        mark = float(input(f"{subject} Marks: "))
        marks.append(mark)
    return marks


# In[ ]:


def preprocess_input_marks(input_marks):
    return np.array(input_marks).reshape(1, -1)


# In[ ]:


input_marks = get_input_marks()


# In[39]:


input_marks_processed = preprocess_input_marks(input_marks)


# In[40]:


predicted_final_score = rf_model.predict(input_marks_processed)
print(f"Predicted Final Score: {predicted_final_score[0]:.2f}")


# In[41]:


import joblib
joblib.dump(predicted_final_score,"10th.pk1")


# In[42]:


from sklearn.metrics import mean_absolute_error


# In[43]:


model=RandomForestRegressor()
model.fit(X_train,y_train)


# In[44]:


y_pred=model.predict(X_test)


# In[45]:


mse=mean_squared_error(y_test,y_pred)
print(mse)


# In[ ]:




