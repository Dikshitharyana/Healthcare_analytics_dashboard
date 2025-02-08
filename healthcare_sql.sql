CREATE database healthcare_db;
USE healthcare_db;

CREATE TABLE patient(
id INT auto_increment PRIMARY KEY,
age int,
gender enum('Male','Female'),
blood_pressure float,
cholesterol float,
diabetes tinyint, -- 1 for yes ,o for no 
heart_diseases tinyint, 
outcome tinyint
);

Show databases;
Use healthcare_db;
Show tables;

DROP TABLE IF EXISTS patient;

create table patient(
id INT auto_increment PRIMARY KEY,
gender ENUM('Male','Female'),
age INT,
hypertension tinyint, -- 1 for yes o for no
heart_diseases tinyint,
smoking_history enum('never','No_info','current'),
bmi float,
Hba1c_level float,
blood_glucose_level float,
diabetes tinyint
);

select * from patient limit 10;
select * from patient where age is null or bmi is null or Hba1c_level is null or blood_glucose_level is null;

select 
avg(age) as avg_age,
min(age) as min_age,
max(age) as max_age,
avg(bmi) as avg_bmi,
min(bmi) as min_bmi,
max(bmi) as max_bmi,
count(distinct gender) as unique_genders,
count(distinct smoking_history) as unique_smoking_statues
from patient;

SELECT COUNT(*) AS total_patients FROM patient;
SELECT gender, COUNT(*) FROM patient GROUP BY gender;
SELECT hypertension, COUNT(*) FROM patient GROUP BY hypertension;
SELECT diabetes, COUNT(*) FROM patient GROUP BY diabetes;
SELECT heart_diseases, COUNT(*) FROM patient GROUP BY heart_diseases;

SELECT age, COUNT(*) FROM patient WHERE diabetes = 1 group by age order by age;
SELECT smoking_history,COUNT(*) FROM patient WHERE heart_diseases = 1 group by smoking_history;
SELECT bmi, count(*) FROM patient WHERE hypertension = 1 group by bmi order by bmi desc;