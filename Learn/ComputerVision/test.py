import pathlib
path = pathlib.Path('Morphological')
pathglob = list(path.glob('*'))
# print(pathglob)

pathiter = next(path.iterdir(), 2)
print(pathiter)