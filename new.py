def recursive_function(a, b):
    if a <= 0 or b <= 0:
        return 1
    else:
        return recursive_function(a - 1, b) + recursive_function(a, b - 2)

print(recursive_function(4, 5))

L1 = [ ] 

L1.append([1, [2, 3], 4]) 

L1.extend([7, 8, 9]) 

print(L1[0][1][1] + L1[2])
original_dict = {'apple': 3, 'banana':4 , 'cherry': 5}

modified_dict = {k: v * 2 for k, v in original_dict.items() if v % 2 != 0}
print(modified_dict)