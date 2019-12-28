import glob, os

def pad_center(s):
    if s[6] == '_': # Needs padding
        return s[:5] + str(0) + s[5:]
    else:
        return s

def rename(dir, pattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        fname = pad_center(title + ext)
        #print(pathAndFilename, os.path.join(dir, str(int(fname[0:4]) + 3150) + fname[4:]))
        os.rename(pathAndFilename, os.path.join(dir, '{0:04d}'.format(int(fname[0:4])) + fname[4:]))


rename('/home/harry/eeg/data/trials/trial_hf/', '*.jpg')