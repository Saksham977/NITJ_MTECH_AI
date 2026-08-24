def compute_Cov_matrix(data):
    n_samples=len(data)
    n_features=len(data[0])
    
    #Computation of the mean of the data
    mean=[]
    for j in range(n_features):
        colsum=0
        for i in range(n_samples):
            colsum+=data[i][j]
        mean.append(colsum/n_samples)

    #Compute the covariance matrix
    cov_mat=[]
    for j in range(n_features):
        row=[]
        for k in range(n_features):
            total=0
            for i in range(n_samples):
                diff_j=data[i][j]-mean[j]
                diff_k=data[i][k]-mean[k]
                total+=(diff_j*diff_k)
            cov=total/(n_samples-1)
            row.append(cov)
        cov_mat.append(row)
    return cov_mat
    
def print_mat(matrix):
    for row in matrix:
        formatted=[f"{val:.4f}" for val in row]
        print(formatted)
            
def get_user_data():
    rows = int(input("Enter number of rows (sample): "))
    col = int(input("Enter number of columns (feature): "))
    data = []
    print(f"Enter {col} values for each row: ")
    for i in range(rows):
        while True:
            entries=input(f"Row {i+1}: ").split()
            if len(entries)!=col:
                print(f"Please enter n columns values.")
                continue
            try:
                row=[float(val) for val in entries]
                data.append(row)
                break
            except ValueError:
                print("Please enter thr valid numbers.")
    return data
    
if __name__=="__main__":
    data=get_user_data()
        
    if len(data)<2:
        print("Please enter atleast 2 rows to compute covarince ")
    else:
        cov=compute_Cov_matrix(data)
        print("\n Covariance Matrix is :")
        print_mat(cov)