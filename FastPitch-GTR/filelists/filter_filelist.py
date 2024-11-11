import sys

# filter file list according to file list
def filter_filelist(filelist, filelist_filter):
    ids = set()
    with open(filelist_filter, 'r') as f:
        for line in f:
            ids.add(line.strip())

    with open(filelist, 'r') as f:
        lines = f.readlines()

    with open(filelist.replace('.txt', '_run.txt'), 'w') as f:
        for line in lines:
            if line.strip().split('|')[1].split('/')[-1] in ids:
                f.write(line)


if __name__ == "__main__":
    filter_filelist(sys.argv[1], sys.argv[2])