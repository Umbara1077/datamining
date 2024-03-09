#Collaborated with Gino Costanzo
import pandas as pd
import numpy as np

#this is the dataset from the example in class, use it for testing
data = [['Susan','F','Blue','O-','excellent',75,'N','N'],['Jim','M','Red','O+','good',65,'N','N'],
                ['Joe','M','Red','AB-','fair',64,'N','Y'],['Jane','F','Green','A+','poor',83,'Y','Y'],
                ['Sam','M','Blue','A-','good',71,'N','N'],['Michelle','F','Blue','O-','good',90,'N','N']]


df = pd.DataFrame( data, columns = ['Name', 'Gender', 'Favorite Color', 'Blood Type', 'General Health', 'Test1', 'Cough', 'High Blood Pressure'] ).drop( 'Name', axis=1 )

#this is for the function where the column is tested for ordinal or not and then used when making it numeric
#I'll give you that function because of this
ordinal_order = ['poor', 'fair', 'good', 'excellent']
df['General Health'] = pd.Categorical(df['General Health'], categories=ordinal_order, ordered=True)

# Create a 10x10 DataFrame with columns of different types

#here's more training data. It's just a function that creates training sets with a set seed and size
def createDataset( seed, rows ):
    np.random.seed(seed)
    data1 = {
            'Gender': np.random.choice(['Male', 'Female'], size=rows),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], size=rows),
            'Is_Eligible': np.random.choice(['Y', 'N'], size=rows),
            'Pet_Type': np.random.choice(['Dog', 'Cat', 'Bird'], size=rows),
            'Favorite_Color': np.random.choice(['Red', 'Blue', 'Green'], size=rows),
            'Education_Level': pd.Categorical(np.random.choice(['High School', 'College', 'Graduate School'], size=rows, p=[0.2, 0.5, 0.3]), categories=['High School', 'College', 'Graduate School'], ordered=True),
            'Age': np.random.randint(18, 60, size=rows),
            'Income': np.random.uniform(20000, 80000, size=rows),
            'Height': np.random.normal(160, 10, size=rows),
            'Has_Pet_Insurance': np.random.choice(['Y', 'N'], size=rows)
    }

    return pd.DataFrame(data1)

#2 for you to try out since I promised I'd provide 3 training sets
train2 = createDataset( 14, 10 )
train3 = createDataset( 42, 15 )

df = train2
df = train3

def getDataType(column):

        #Easiest test of all, juts test if its dtype is numeric
        if pd.api.types.is_numeric_dtype(column):
                return 'Numeric'

        countValues = column.unique()

        #In all of the data I give you to test and use to grade this assignment, Asymmetric Binary data will be Y/N
        if len(countValues) == 2 and set(countValues) == {'Y', 'N'}:
                return 'Asymmetric Binary'

        #Since Asymmetric Binary is Y/N, Symmetric Binary will have 2 unique values that aren't Y/N
        if len(countValues) == 2 and set(countValues) != {'Y', 'N'}:
                return 'Symmetric Binary'

        #This uses the code from the previous block to test for ordinal
        if isinstance(column.dtype, pd.CategoricalDtype) and column.dtype.ordered:
                return 'Ordinal'

        #If it's categorical and not ordinal, it's nominal
        if pd.api.types.is_categorical_dtype(column) or pd.api.types.is_object_dtype(column):
                return 'Nominal'

#This is just a dictionary mapping each column and its datatype. Might be useful
dataTypes = {}
for column in df.columns:
    dataTypes[column] = getDataType(df[column])

    #You should pass a dataframe of just the symmetric binary columns here and return a matrix of the distance
def symmetric_binary_dist(data):
    n = len(data)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n): 
            dist_matrix[i, j] = dist_matrix[j, i] = np.sum(data.iloc[i] != data.iloc[j])
    return dist_matrix

#distance_matrix = symmetric_binary_dist(df)
#print(distance_matrix)

def nominal_dist(data):
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.sum(data.iloc[i] != data.iloc[j])
            dist_matrix[i, j] = dist_matrix[j, i] = mismatches / data.shape[1]
    return dist_matrix

#distance_matrix = nominal_dist(df)
#print(distance_matrix)

#You should pass a dataframe of just the ordinal columns here and return a matrix of the distance

#Suggestion: maybe just convert to numeric and run the numeric_dist method
def ordinal_dist( data ):
    dist = np.empty( shape = [ len( df ), len( df ) ] )
    n = len(data)
    for i in range(n):
        if getDataType(df[:,i]) == 'Ordinal':
            conv_dict={'poor':0,'fair':.33,'good':.66,'excellent':1,'None':np.nan}
            df['data_c']=df.data.apply(conv_dict.get)
    return numeric_dist(data)

#ordinal_dist_matrix = ordinal_dist(df)
#print(ordinal_dist_matrix)

#You should pass a dataframe of just the numeric columns here and return a matrix of the distance


def numeric_dist(data):
    dist = []
    values = data.loc[:, 'Test1'].tolist()
    minimum = min(values)
    maximum = max(values)

    for i in range(len(values)):
        row = []
        for j in range(len(values)):
            row.append(values[j] - (minimum / (maximum - minimum)))
        dist.append(row)

    return dist

#distance_matrix = numeric_dist(df)
#print(distance_matrix) 

#You should pass a dataframe of just the asymmetric binary columns here and return a matrix of the distance
def asymmetric_binary_dist(data):
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    converted = lambda x: True if x == 'Y' else False
    data_bool = data.applymap(converted)
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.logical_xor(data_bool.iloc[i], data_bool.iloc[j]).sum()
            dist_matrix[i, j] = dist_matrix[j, i] = mismatches
    return dist_matrix

#distance_matrix = asymmetric_binary_dist(df)
#print(distance_matrix)

#This should probably call every distance method and aggregate the results.
#Don't forget to make sure each individual distance matrix is weighted by the number of columns of that data type
def get_dist( data ):
   n = len(df)
   aggregate_dist = np.zeros((n, n))
   for column in df.columns:
        col_data = df[column]
        data_type = getDataType(col_data)
        if  data_type == 'Nominal':
            dist_matrix = nominal_dist(col_data.to_frame())
        elif data_type == 'Symmetric Binary':
            dist_matrix = symmetric_binary_dist(col_data.to_frame())
        elif data_type == 'Asymmetric Binary':
            dist_matrix = asymmetric_binary_dist(col_data.to_frame())
       # elif data_type == 'Numeric':
       #     dist_matrix = numeric_dist(col_data.to_frame())
       # elif data_type == 'Ordinal':
       #     dist_matrix = ordinal_dist(col_data.to_frame())
        else:
            continue  
        aggregate_dist += dist_matrix
   return aggregate_dist

#aggregate_distance_matrix = get_dist(df)
#print(aggregate_distance_matrix)