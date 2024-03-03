import timeit
import numpy as np
def bisearch(arr, target, left, right):
    if left > right:
        return -1  # Base case: Element not found

    mid = (left + right) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return bisearch(arr, target, mid + 1, right)
    else:
        return bisearch(arr, target, left, mid - 1)

from functools import lru_cache

@lru_cache(maxsize=None)  # maxsize=None means unlimited cache
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1  # Element not found

    mid = (left + right) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(tuple(arr), target, mid + 1, right)
    else:
        return binary_search_recursive(tuple(arr), target, left, mid - 1)

# Note: Arrays/lists are not hashable, so we convert `arr` to a tuple before passing it.


# To call the function, you'd use it like so:
# result = binary_search_recursive(arr, target, 0, len(arr)-1)


arr = np.arange(100000000) 
target = 10000
left = 0
right = len(arr)-1
print(timeit.timeit(lambda: binary_search_recursive(arr, target, left, right), number=1000))


