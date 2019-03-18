import random
random.seed(123)
filenames = []
for i in range(201):
  filenames.append('data/splits/bySamples/binnedOpennessSample' + str(i) + '.npy')

random.shuffle(filenames)

split1 = 181
split2 = 191

with open('data/splits/bySamples/trainFilenames.txt', 'a') as f:
  for i in range(split1):
    f.write(filenames[i] +'\n')

with open('data/splits/bySamples/trainSamples.txt', 'a') as f:
  for i in range(split1):
    f.write(''.join([i for i in filenames[i] if not i.isdigit()]) + '\n')

with open('data/splits/bySamples/devFilenames.txt', 'a') as f:
  for i in range(split1, split2):
    f.write(filenames[i] + '\n')

with open('data/splits/bySamples/devSamples.txt', 'a') as f:
  for i in range(split1, split2):
    f.write(''.join([i for i in filenames[i] if not i.isdigit()]) + '\n')


with open('data/splits/bySamples/testFilenames.txt', 'a') as f:
  for i in range(split2, len(filenames)):
    f.write(filenames[i] + '\n')

with open('data/splits/bySamples/testSamples.txt', 'a') as f:
  for i in range(split2, len(filenames)):
    f.write(''.join([i for i in filenames[i] if not i.isdigit()]) + '\n')
