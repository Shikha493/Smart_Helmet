import os
folder = 'assets'
for f in ['rose.jpg','tulip.jpg','lily.jpg','lotus.jpg','hibiscus.jpg','marigold.jpg']:
    with open(os.path.join(folder, f), 'rb') as fh:
        print(f, '->', fh.read(3))