import pathlib
path = pathlib.Path('Morphological')
pathglob = list(path.glob('*'))
# print(pathglob)

print(next(path.iterdir(),"Hey There"))

numbers = iter([1, 2, 3])
print(next(numbers, 'No more items'))  # Output: 1
print(next(numbers, 'No more items'))  # Output: 2
print(next(numbers, 'No more items'))  # Output: 3
print(next(numbers, 'No more items'))  # Output: No more