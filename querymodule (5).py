
#query_module.py

def compute_average_age(death_records):
    
   # Compute the average age of individuals whose heart failure resulted in death.
   
    if not death_records:
        return None

    total_age = 0
    count = 0

    for record in death_records.values():
        age = record.get("age")
        if age is not None:
            total_age += float(age)
            count += 1

    if count == 0:
        return None

    average_age = total_age / count
    return average_age

def compute_modal_age(death_records):
    
    # Compute the modal age of individuals whose heart failure resulted in death.

     
    age_counts={}
    for record in death_records.values():
        age = record.get("age")
        if age in age_counts:
            age_counts[age] += 1
        else:
            age_counts[age] = 1

    modal_age = max(age_counts, key=age_counts.get)
    return modal_age

def compute_median_age(death_records):
    
    #Compute the median age of individuals whose heart failure resulted in death.
    
    if not death_records:
        return None

    ages = [float(record.get("age", 0)) for record in death_records.values() if record.get("age")]

    if not ages:
        return None

    sorted_ages = sorted(ages)
    n = len(sorted_ages)

    if n % 2 == 0:
        # If the number of elements is even, average the middle two values
        middle1 = sorted_ages[n // 2 - 1]
        middle2 = sorted_ages[n // 2]
        median_age = (middle1 + middle2) / 2
    else:
        # If the number of elements is odd, take the middle value
        median_age = sorted_ages[n // 2]

    return median_age
def compute_average_age_diabetes(data):
    
    #Compute the average age of individuals with diabetes.
   
    diabetes_records = {key: value for key, value in data.items() if value.get("diabetes") == "1"}

    if not diabetes_records:
        return None

    total_age = 0
    count = 0

    for record in diabetes_records.values():
        age = record.get("age")
        if age is not None:
            total_age += float(age)
            count += 1

    if count == 0:
        return None

    average_age = total_age / count
    return average_age


def compute_modal_age_diabetes(data):
    
    #Compute the modal age of individuals with diabetes.
     
    diabetes_records = {key: value for key, value in data.items() if value.get("diabetes") == "1"}

    if not diabetes_records:
        return None

    age_counts = {}

    for record in diabetes_records.values():
        age = record.get("age")
        if age is not None:
            if age in age_counts:
                age_counts[age] += 1
            else:
                age_counts[age] = 1

    if not age_counts:
        return None

    modal_age = max(age_counts, key=age_counts.get)
    return modal_age
def compute_average_age_diabetes(data):
    #Compute the average age of individuals with diabetes.
    diabetes_records = {key: value for key, value in data.items() if value.get("diabetes") == "1"}

    if not diabetes_records:
        return None

    total_age = 0
    count = 0

    for record in diabetes_records.values():
        age = record.get("age")
        if age is not None:
            total_age += float(age)
            count += 1

    if count == 0:
        return None

    average_age = total_age / count
    return average_age


def compute_modal_age_diabetes(data):
    
    #Compute the modal age of individuals with diabetes.

    diabetes_records = {key: value for key, value in data.items() if value.get("diabetes") == "1"}

    if not diabetes_records:
        return None

    age_counts = {}

    for record in diabetes_records.values():
        age = record.get("age")
        if age is not None:
            if age in age_counts:
                age_counts[age] += 1
            else:
                age_counts[age] = 1

    if not age_counts:
        return None

    modal_age = max(age_counts, key=age_counts.get)
    return modal_age


def compute_median_age_diabetes(data):
    
    #Compute the median age of individuals with diabetes.

    diabetes_records = {key: value for key, value in data.items() if value.get("diabetes") == "1"}

    if not diabetes_records:
        return None

    ages = [float(record.get("age", 0)) for record in diabetes_records.values() if record.get("age")]

    if not ages:
        return None

    sorted_ages = sorted(ages)
    n = len(sorted_ages)

    if n % 2 == 0:
        middle1 = sorted_ages[n // 2 - 1]
        middle2 = sorted_ages[n // 2]
        median_age = (middle1 + middle2) / 2
    else:
        median_age = sorted_ages[n // 2]

    return median_age
def compute_average_time_no_death(data):
    
    #Compute the average time taken for individuals whose heart failure did not result in death.

    no_death_records = {key: value for key, value in data.items() if value.get("DEATH_EVENT") == "0"}

    if not no_death_records:
        return None

    total_time = 0
    count = 0

    for record in no_death_records.values():
        time_taken = record.get("time")  
        if time_taken is not None:
            total_time += float(time_taken)
            count += 1

    if count == 0:
        return None

    average_time = total_time / count
    return average_time
def compute_stats_high_blood_pressure(data):
    
   #Compute median age, average age, and modal age for individuals with high blood pressure.

    high_bp_records = {key: value for key, value in data.items() if value.get("high_blood_pressure") == "1"}



    ages = [float(record.get("age", 0)) for record in high_bp_records.values() if record.get("age")]

    
     

    # Compute median age
    sorted_ages = sorted(ages)
    n = len(sorted_ages)

    if n % 2 == 0:
        middle1 = sorted_ages[n // 2 - 1]
        middle2 = sorted_ages[n // 2]
        median_age = (middle1 + middle2) / 2
    else:
        median_age = sorted_ages[n // 2]

    # Compute average age
    total_age = sum(ages)
    average_age = total_age / len(ages)

    # Compute modal age
    age_counts = {}
    for age in ages:
        if age in age_counts:
            age_counts[age] += 1
        else:
            age_counts[age] = 1

    modal_age = max(age_counts, key=age_counts.get)

    return median_age, average_age, modal_age
def diabetes_link_analysis(data):
    
   # diabetes is linked to smoking and high blood pressure.
   
    
    diabetes_records = {key: value for key, value in data.items() if value.get("diabetes") == "1"}

    if not diabetes_records:
        return False  # No individuals with diabetes found

    # Check for the presence of smoking and high blood pressure in individuals with diabetes
    for record in diabetes_records.values():
        if record.get("smoking") == "1" and record.get("high_blood_pressure") == "1":
            return True  # Diabetes is linked to smoking and high blood pressure

    return False  # No clear link found between diabetes, smoking, and high blood pressure
def compute_average_serum_sodium_diabetes(data):
    
    #Compute the average serum sodium of individuals with diabetes.

    diabetes_records = {key: value for key, value in data.items() if value.get("diabetes") == "1"}

     

    total_serum_sodium = 0
    count = 0

    for record in diabetes_records.values():
        serum_sodium = record.get("serum_sodium")  
        if serum_sodium is not None:
            total_serum_sodium += float(serum_sodium)
            count += 1

    if count == 0:
        return None  

    average_serum_sodium = total_serum_sodium / count
    return average_serum_sodium
def is_anemia_linked_to_smoking(data):
    
   # Determine if anemia is linked to smoking.
   
    anemia_records = {key: value for key, value in data.items() if value.get("anaemia") == "1"}

          

    # Check for the presence of smoking in individuals with anemia
    for record in anemia_records.values():
        if record.get("smoking") == "1":
            return True  # Anemia is linked to smoking

    return False  
def individuals_without_high_blood_pressure_died(data):
    #Return individuals without high blood pressure who died of heart failure.
   
    result_dict = {}

    for key, value in data.items():
        high_blood_pressure = value.get("high_blood_pressure")
        death_event = value.get("DEATH_EVENT")

        if high_blood_pressure == "0" and death_event == "1":
            result_dict[key] = value

    return result_dict
def compute_iqrs(data):
    
    #Compute and return the IQRs (Interquartile Ranges) for ejection fraction and serum creatinine.

    # Extract ejection fraction and serum creatinine data
    ejection_fraction_values = [float(value.get("ejection_fraction", 0)) for value in data.values()]
    serum_creatinine_values = [float(value.get("serum_creatinine", 0)) for value in data.values()]

    # Sort the data
    sorted_ef = sorted(ejection_fraction_values)
    sorted_scr = sorted(serum_creatinine_values)

    # Calculate quartiles
    q1_index_ef = int(0.25 * len(sorted_ef))
    q3_index_ef = int(0.75 * len(sorted_ef))
    q1_ef, q3_ef = sorted_ef[q1_index_ef], sorted_ef[q3_index_ef]

    q1_index_scr = int(0.25 * len(sorted_scr))
    q3_index_scr = int(0.75 * len(sorted_scr))
    q1_scr, q3_scr = sorted_scr[q1_index_scr], sorted_scr[q3_index_scr]

    # Calculate Interquartile Ranges (IQRs)
    iqr_ef = q3_ef - q1_ef
    iqr_scr = q3_scr - q1_scr

    return iqr_ef, iqr_scr
def compute_sample_variance(data):
    
    #Compute and return the sample variance for creatinine phosphokinase and serum sodium.
   
    # Extract creatinine phosphokinase and serum sodium data
    ck_values = [float(value.get("creatinine_phosphokinase", 0)) for value in data.values()]
    sodium_values = [float(value.get("serum_sodium", 0)) for value in data.values()]

    # Calculate means
    mean_ck = sum(ck_values) / len(ck_values)
    mean_sodium = sum(sodium_values) / len(sodium_values)

    # Calculate sample variances
    variance_ck = sum((x - mean_ck) ** 2 for x in ck_values) / (len(ck_values) - 1) if len(ck_values) > 1 else 0
    variance_sodium = sum((x - mean_sodium) ** 2 for x in sodium_values) / (len(sodium_values) - 1) if len(sodium_values) > 1 else 0

    return variance_ck, variance_sodium
def compute_sample_variance_and_save(data, output_file):
    
      
    # Extract creatinine phosphokinase and serum sodium data
    ck_values = [float(value.get("creatinine_phosphokinase", 0)) for value in data.values()]
    sodium_values = [float(value.get("serum_sodium", 0)) for value in data.values()]

    # Calculate means
    mean_ck = sum(ck_values) / len(ck_values)
    mean_sodium = sum(sodium_values) / len(sodium_values)

    # Calculate sample variances
    variance_ck = sum((x - mean_ck) ** 2 for x in ck_values) / (len(ck_values) - 1) if len(ck_values) > 1 else 0
    variance_sodium = sum((x - mean_sodium) ** 2 for x in sodium_values) / (len(sodium_values) - 1) if len(sodium_values) > 1 else 0

    # Save results to CSV
    with open(output_file, 'w', newline='') as csvfile:
        # Write the header
        csvfile.write("Statistic,Value\n")
        
        # Write the data
        csvfile.write(f"Sample Variance for Creatinine Phosphokinase,{variance_ck}\n")
        csvfile.write(f"Sample Variance for Serum Sodium,{variance_sodium}\n")

    return variance_ck, variance_sodium
