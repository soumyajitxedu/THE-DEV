def quick_sort_array(arr):
    if len(arr) <= 1:
        return arr 
    p = arr[0]
    ps = [x for x in arr if x == p]
    l = quick_sort_array([x for x in arr if x < p])
    g = quick_sort_array([x for x in arr if x > p])
    return l + p + g 
array = [0,25,4,1,0,23,52,5,9,535,3,5,36,]
sorted_arr = quick_sort_array(array)
print("orginal array ", array)
print("sorted array : ", sorted_arr)