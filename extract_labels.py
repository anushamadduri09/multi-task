import sys
import os

with open(sys.argv[1], 'r') as fin:
    female_count, male_count = 0, 0
    child_count, young_count, middle_count, old_count = 0, 0, 0, 0
    a_c, h_c, sa_c, su_c, f_c, d_c, n_c = 0, 0, 0, 0, 0, 0, 0 
    for file_name in fin:
        dir_name = "/".join(file_name.split("/")[0:-1])
        fin_labels = open(file_name.strip(), 'r')
        for line in fin_labels:
            line = line.strip().split()
            if len(line) < 17 or len(line) > 17:
                continue
            image_name  = line[2]
            gender = line[10]
            if gender == 'FEMALE':
                gender = 0
                female_count += 1
            else:
                gender = 1
                male_count += 1
            emotion = line[11]
            if emotion == 'ANGER':
                emotion = 0
                a_c += 1
            elif emotion == 'HAPPINESS':
                emotion = 1
                h_c += 1
            elif emotion == 'SADNESS':
                emotion = 2
                sa_c += 1
            elif emotion == 'SURPRISE':
                emotion = 3
                su_c += 1
            elif emotion == 'FEAR':
                emotion = 4
                f_c += 1
            elif emotion == 'DISGUST':
                emotion = 5
                d_c += 1
            elif emotion == 'NEUTRAL':
                emotion = 6
                n_c += 1
            age = line[14]
            if age == 'CHILD':
                age = 0
                child_count += 1
            elif age == 'YOUNG':
                age = 1
                young_count += 1
            elif age == 'MIDDLE':
                age = 2
                middle_count += 1
            elif age == 'OLD':
                age = 3
                old_count += 1
            print("{}/images/{} {} {} {}".format("/".join(dir_name.split("/")[-2:]), image_name, gender, age, emotion))
    #print(female_count, male_count, female_count+male_count)
    #print(child_count, young_count, middle_count, old_count, child_count+young_count+middle_count+old_count)
    #print(a_c, h_c, sa_c, su_c, f_c, d_c, n_c, a_c+h_c+sa_c+su_c+f_c+d_c+n_c)
