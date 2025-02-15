#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class DataProcessor:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.data_input = None

    def load_data(self):
        self.data_input = pd.read_csv(self.csv_file_path)

    def clean_and_preprocess(self):
        # Handling Missing Values
        self.data_input.dropna(inplace=True)
        for column in self.data_input.columns:
            mean_value = self.data_input[column].mean()
            self.data_input[column].fillna(mean_value, inplace=True)

        # Handling Duplicates
        self.data_input.drop_duplicates(inplace=True)

    def separate_label_from_data(self, label_column):
        label_index = self.data_input.columns.get_loc(label_column)
        data_array = np.array(self.data_input)
        result_array_data = np.delete(data_array, label_index, axis=1)
        X = result_array_data
        y = data_array[:, label_index]

        return X, y

class ClassifierEvaluator(DataProcessor):
    def __init__(self, csv_file_path):
        super().__init__(csv_file_path)

    def train_and_evaluate_classifier(self, X_train, y_train, X_test, y_test, classifier):
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)

        return accuracy, precision, recall, f1, conf_matrix

    def plot_confusion_matrix(self, conf_matrix, class_labels, title):
        plt.figure()
        heatmap = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                              annot_kws={"weight": "bold", "size": 14},
                              xticklabels=class_labels, yticklabels=class_labels,square=True)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), weight='bold', fontsize=12)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), weight='bold', fontsize=12)
        plt.title(title, fontweight='bold', fontsize=12)
        plt.xlabel('Predicted Labels', fontweight='bold', fontsize=12)
        plt.ylabel('True Labels', fontweight='bold', fontsize=12)
        plt.show()

class AnemiaClassifier(ClassifierEvaluator):
    def __init__(self, csv_file_path):
        super().__init__(csv_file_path)

    def analyze_anemia_svm(self):
        self.load_data()
        self.clean_and_preprocess()

        # ############### Class: Anaemia
        anaemia_index = self.data_input.columns.get_loc('anaemia')
        X, y = self.separate_label_from_data('anaemia')

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # SVM Classifier
        svm_classifier = SVC(kernel='linear')
        accuracy_svm, precision_svm, recall_svm, f1_svm, conf_matrix_svm = self.train_and_evaluate_classifier(
        X_train, y_train, X_test, y_test, svm_classifier)
        
        # Plot Confusion Matrix
        class_labels_anaemia = ["Anemia: 0", "Anemia: 1"]
        self.plot_confusion_matrix(conf_matrix_svm, class_labels_anaemia, 'Anemia - SVM')
        
        return accuracy_svm, precision_svm, recall_svm, f1_svm
      
    def analyze_anemia_knn(self):
        self.load_data()
        self.clean_and_preprocess()
        # ############### Class: Anaemia
        anaemia_index = self.data_input.columns.get_loc('anaemia')
        X, y = self.separate_label_from_data('anaemia')

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #KNN
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        accuracy_knn, precision_knn, recall_knn, f1_knn, conf_matrix_knn = self.train_and_evaluate_classifier(
        X_train, y_train, X_test, y_test, knn_classifier)
        
        class_labels_anaemia = ["Anemia: 0", "Anemia: 1"]
        self.plot_confusion_matrix(conf_matrix_knn, class_labels_anaemia, 'Anemia - knn')
        
        return accuracy_knn, precision_knn, recall_knn, f1_knn
    
    def analyze_anemia_rf(self):
        self.load_data()
        self.clean_and_preprocess()
        # ############### Class: Anaemia
        anaemia_index = self.data_input.columns.get_loc('anaemia')
        X, y = self.separate_label_from_data('anaemia')

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #RandomForest
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        accuracy_rf, precision_rf, recall_rf, f1_rf, conf_matrix_rf = self.train_and_evaluate_classifier(
        X_train, y_train, X_test, y_test, rf_classifier)

        class_labels_anaemia = ["Anemia: 0", "Anemia: 1"]
        self.plot_confusion_matrix(conf_matrix_rf, class_labels_anaemia, 'Anemia - rf')

        return accuracy_rf, precision_rf, recall_rf, f1_rf
        

class HighBloodPressureClassifier(ClassifierEvaluator):
    def __init__(self, csv_file_path):
        super().__init__(csv_file_path)

    def analyze_high_blood_pressure_svm(self):
        self.load_data()
        self.clean_and_preprocess()

        # ############### Class:high_blood_pressure
        HBP_index = self.data_input.columns.get_loc('high_blood_pressure')
        X, y = self.separate_label_from_data('high_blood_pressure')

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # SVM Classifier
        svm_classifier = SVC(kernel='linear')
        accuracy_svm, precision_svm, recall_svm, f1_svm, conf_matrix_svm = self.train_and_evaluate_classifier(
            X_train, y_train, X_test, y_test, svm_classifier)
                
        # Plot Confusion Matrix
        class_labels_HBP = ["High_BP: 0", "High_BP: 1"]
        self.plot_confusion_matrix(conf_matrix_svm, class_labels_HBP, 'high_blood_pressure - SVM')
        
        return accuracy_svm, precision_svm, recall_svm, f1_svm
        
    def analyze_high_blood_pressure_knn(self):
        self.load_data()
        self.clean_and_preprocess()

        # ############### Class:high_blood_pressure
        HBP_index = self.data_input.columns.get_loc('high_blood_pressure')
        X, y = self.separate_label_from_data('high_blood_pressure')
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #KNN
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        accuracy_knn, precision_knn, recall_knn, f1_knn, conf_matrix_knn = self.train_and_evaluate_classifier(
        X_train, y_train, X_test, y_test, knn_classifier)
     
        class_labels_HBP = ["High_BP: 0", "High_BP: 1"]
        self.plot_confusion_matrix(conf_matrix_knn, class_labels_HBP, 'high_blood_pressure - knn')

        return accuracy_knn, precision_knn, recall_knn, f1_knn    
    
    def analyze_high_blood_pressure_rf(self):
        self.load_data()
        self.clean_and_preprocess()

        # ############### Class:high_blood_pressure
        HBP_index = self.data_input.columns.get_loc('high_blood_pressure')
        X, y = self.separate_label_from_data('high_blood_pressure')

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #RandomForest
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        accuracy_rf, precision_rf, recall_rf, f1_rf, conf_matrix_rf = self.train_and_evaluate_classifier(
        X_train, y_train, X_test, y_test, rf_classifier) 
        
        
        class_labels_HBP = ["High_BP: 0", "high_BP: 1"]
        self.plot_confusion_matrix(conf_matrix_rf,class_labels_HBP, 'High_BP RFC')

        return accuracy_rf, precision_rf, recall_rf, f1_rf
    
class  DeathEventClassifier(ClassifierEvaluator):
    def __init__(self, csv_file_path):
        super().__init__(csv_file_path)

    def analyze_DeathEvent_svm(self):
        self.load_data()
        self.clean_and_preprocess()

        # ############### Class: Anaemia
        DeathEvent_index = self.data_input.columns.get_loc('DEATH_EVENT')
        X, y = self.separate_label_from_data('DEATH_EVENT')

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # SVM Classifier
        svm_classifier = SVC(kernel='linear')
        accuracy_svm, precision_svm, recall_svm, f1_svm, conf_matrix_svm = self.train_and_evaluate_classifier(
            X_train, y_train, X_test, y_test, svm_classifier)
        
        # Plot Confusion Matrix
        class_labels_DeathEvent = ["DeathEvent: 0", "DeathEvent: 1"]
        self.plot_confusion_matrix(conf_matrix_svm, class_labels_DeathEvent, 'DeathEvent - SVM')
        
        return accuracy_svm, precision_svm, recall_svm, f1_svm
        
    def analyze_DeathEvent_knn(self):
        self.load_data()
        self.clean_and_preprocess()

        # ############### Class: Anaemia
        DeathEvent_index = self.data_input.columns.get_loc('DEATH_EVENT')
        X, y = self.separate_label_from_data('DEATH_EVENT')

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # KNN
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        accuracy_knn, precision_knn, recall_knn, f1_knn, conf_matrix_knn = self.train_and_evaluate_classifier(
        X_train, y_train, X_test, y_test, knn_classifier)
        
        class_labels_DeathEvent = ["DeathEvent: 0", "DeathEvent: 1"]
        self.plot_confusion_matrix(conf_matrix_knn, class_labels_DeathEvent, 'DeathEvent - knn')

        return accuracy_knn, precision_knn, recall_knn, f1_knn
    
    def analyze_DeathEvent_rf(self):
        self.load_data()
        self.clean_and_preprocess()

        # ############### Class: Anaemia
        DeathEvent_index = self.data_input.columns.get_loc('DEATH_EVENT')
        X, y = self.separate_label_from_data('DEATH_EVENT')

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # RandomForest
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        accuracy_rf, precision_rf, recall_rf, f1_rf, conf_matrix_rf = self.train_and_evaluate_classifier(
        X_train, y_train, X_test, y_test, rf_classifier)

        class_labels_DeathEvent = ["DeathEvent: 0", "DeathEvent: 1"]
        self.plot_confusion_matrix(conf_matrix_rf, class_labels_DeathEvent, 'DeathEvent - rf')

        return accuracy_rf, precision_rf, recall_rf, f1_rf

