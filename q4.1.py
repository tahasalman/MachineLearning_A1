import csv
import statistics

def calculate_mean(data):
    sum = 0
    count = 0
    for e in data:
        if e:
            sum+=float(e)
            count+=1
    return sum/count

def calculate_median(data):
    refined_data = []
    for e in data:
        if e:
            refined_data.append(float(e))
    return statistics.median(refined_data)

def refine_data(data_path,saving_path):
    lines = None
    output = ""
    with open(data_path,'r') as f:
        lines=f.readlines()

    for line in lines:
        temp_list = line.split(',')
        for i in range(0,len(temp_list)):
            if temp_list[i] == "?":
                temp_list[i] = ""
        formatted_line = ','.join(temp_list)
        output += formatted_line

    with open(saving_path,'w') as f:
        f.write(output)
def update_missing_data_with_median(data_path,ignore_col_num=[]):
    mean_dict = {}
    cols = []
    with open(data_path, 'r+') as f:
        csv_data = csv.reader(f, delimiter=',')

        count = 0
        for row in csv_data:
            count_2 = 0
            for col in row:
                if count == 0:
                    cols.append([col])
                else:
                    cols[count_2].append(col)
                count_2 += 1
            count += 1

    for i in range(0, len(cols)):
        if i not in ignore_col_num:
            mean_dict[i] = calculate_median(cols[i])

    output = ""
    with open(data_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        temp_list = line.split(',')
        for i in range(0, len(temp_list)):
            if temp_list[i] == "":
                temp_list[i] = str(mean_dict[i])
        formatted_line = ','.join(temp_list)
        output += formatted_line

    return output


def update_missing_data_with_mean(data_path,ignore_col_num=[]):
    mean_dict = {}
    cols = []
    with open(data_path,'r+') as f:
        csv_data = csv.reader(f,delimiter=',')

        count = 0
        for row in csv_data:
            count_2 = 0
            for col in row:
                if count==0:
                    cols.append([col])
                else:
                    cols[count_2].append(col)
                count_2+=1
            count+=1

    for i in range(0,len(cols)):
        if i not in ignore_col_num:
            mean_dict[i] = calculate_mean(cols[i])

    output = ""
    with open(data_path,'r') as f:
        lines=f.readlines()

    for line in lines:
        temp_list = line.split(',')
        for i in range(0,len(temp_list)):
            if temp_list[i] == "":
                temp_list[i] = str(mean_dict[i])
        formatted_line = ','.join(temp_list)
        output += formatted_line

    return output


def remove_useless_cols(data_path,useless_cols):
    output = ""
    with open(data_path,'r') as f:
        lines=f.readlines()

    for line in lines:
        temp_list=line.split(',')
        formatted_list = []
        for i in range(0,len(temp_list)):
            if i not in useless_cols:
                formatted_list.append(temp_list[i])
        output+= (',').join(formatted_list)

    return output



def remove_missing_data_cols(data_path,removal_threshold=0.5):
    '''
    This method removes all variables that have instances greater that the removal threshold for missing data
    and updates the rest of the missing data with median.
    '''

    with open(data_path,'r') as f:
        lines=f.readlines()

    missing_data_cols = {}
    for i in range(0,len(lines[0].split(','))):
        missing_data_cols[i] = 0

    for line in lines:
        temp_list = line.split(',')
        for i in range(0,len(temp_list)):
            if temp_list[i] == "":
                missing_data_cols[i]+=1

    total_features = len(lines)
    removal_threshold = removal_threshold*total_features
    output = ""

    for line in lines:
        temp_list=line.split(',')
        formatted_list = []
        for i in range(0,len(temp_list)):
            if missing_data_cols[i] < removal_threshold:
                formatted_list.append(temp_list[i])
        output+= (',').join(formatted_list)

    return output

def remove_missing_data_rows(data_path):
    '''
    This method removes all rows that have any missing data
    '''
    output = ""
    with open(data_path,'r') as f:
        lines=f.readlines()

    for line in lines:
        add_to_output = True
        temp_list=line.split(',')
        for i in range(0,len(temp_list)):
            if temp_list[i] == "":
                add_to_output=False
                break
        if add_to_output:
            output+=line

    return output



if __name__ == "__main__":
    original_data_path = 'Datasets/CrimeData/crime_data.csv'
    refined_data_path = 'Datasets/CrimeData/crime_data_refined.csv'   #path to data where ? are replaced with empty cells
    update_path_mean = 'Datasets/CrimeData/crime_data_updated_mean.csv'   #path to data where all missing data is replaced with mean
    update_path_custom = 'Datasets/CrimeData/crime_data_updated_custom.csv'

    print("Replacing all ? with empty cells")
    refine_data(data_path=original_data_path,
                saving_path=refined_data_path)
    print("Updated file saved to {}\n".format(refined_data_path))

    print("Replacing all missing values with mean")
    refined_data_mean = update_missing_data_with_mean(refined_data_path,ignore_col_num=[3])
    with open(update_path_mean,'w') as f:
        f.write(refined_data_mean)
    print("Updated data saved to {}\n".format(update_path_mean))


    print("Removing non-predictor data columns")
    refined_data_custom = remove_useless_cols(data_path=refined_data_path,useless_cols=[0,1,2,3,4])
    with open(update_path_custom, 'w') as f:
        f.write(refined_data_custom)

    print("Replacing all missing values with median")
    refined_data_custom = update_missing_data_with_median(data_path=update_path_custom,ignore_col_num=[3])
    with open(update_path_custom,'w') as f:
        f.write(refined_data_custom)
    print("Updated data saved to {}\n".format(update_path_custom))

    #
    # removal_threshold = 0.2
    # print("Removing non-predictor data columns")
    # refined_data_custom = remove_useless_cols(data_path=refined_data_path,useless_cols=[0,1,2,3,4])
    # with open(update_path_custom, 'w') as f:
    #     f.write(refined_data_custom)
    # print("Removing data columns with missing data over {} of the total data".format(removal_threshold))
    # refined_data_custom = remove_missing_data_cols(data_path=update_path_custom,removal_threshold=removal_threshold)
    # with open(update_path_custom,'w') as f:
    #     f.write(refined_data_custom)
    # print("Removing all rows with missing data")
    # refined_data_custom = remove_missing_data_rows(update_path_custom)
    # with open(update_path_custom,'w') as f:
    #     f.write(refined_data_custom)
    # print("Updated data saved to {}".format(update_path_custom))