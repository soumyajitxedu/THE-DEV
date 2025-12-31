def find_max(arr):
    if len(arr) == 0:
        return None
    max_val = arr[0]

    for i in range(1,len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val
arr = [1,2,3,4,5,6,7,9,5,87,45,88,66]
print(f"max element is : {find_max(arr)}")
