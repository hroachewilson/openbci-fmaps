import shutil
import random
import os
import csv

src = '/home/harry/eeg/data/trial_mixed/'
dest = '/home/harry/eeg/code/final-four-class-five-video-classification-methods/data'

file_list = os.listdir(src)

# train	ApplyLipstick	v_ApplyLipstick_g14_c04	106
data_file = []
folders = ['train', 'test']
classes = ['1', '2', '3', '4']

def img_num(x):
    return x[0:4]


def win_num(x):
    return x[5:7]

def class_num(x):
    return x[11:12]

file_sorted = sorted(file_list, key=img_num)
match = 0


for vid in range(0, 4800, 2):
    str_match = '{0:04d}'.format(vid//2)
    point = 0
    for check in range(0, 80):
        vid_str = file_sorted[match][0:4]
        if vid_str == str_match:
            point += 1

        match += 1
    print(point)

    if point == 80:

        frame_sorted = sorted(file_sorted[match - 80:match], key=win_num)
        class_sorted = sorted(frame_sorted, key=class_num)

        frame_num = 0
        test = random.randint(1, 5) == 1
        holdout = random.randint(1, 20) == 1
        for frame in class_sorted[0:40]:
            fname = 'v_' + '{0:04d}'.format(vid) + '_c_' + frame[11] + '_frame-{0:04d}.jpg'.format(frame_num)
            if holdout:
                write_dir = dest + '/holdout/class' + str(frame[11]) + '/'
                shutil.copy(src + frame, write_dir + fname)
            else:
                if test:    # Take 20% for test set
                    data_file.append(['test', 'class' + str(frame[11]), fname[0:16], 40])
                    write_dir = dest + '/test/class' + str(frame[11]) + '/'
                    shutil.copy(src + frame, write_dir + fname)

                else:
                    data_file.append(['train', 'class' + str(frame[11]), fname[0:16], 40])
                    write_dir = dest + '/train/class' + str(frame[11]) + '/'
                    shutil.copy(src + frame, write_dir + fname)

            frame_num += 1
        frame_num = 0
        for frame in class_sorted[40:80]:
            fname = 'v_' + '{0:04d}'.format(vid + 1) + '_c_' + frame[11] + '_frame-{0:04d}.jpg'.format(frame_num)
            if holdout:
                write_dir = dest + '/holdout/class' + str(frame[11]) + '/'
                shutil.copy(src + frame, write_dir + fname)
            else:
                if test:  # Take 20% for test set
                    data_file.append(['test', 'class' + str(frame[11]), fname[0:16], 40])
                    write_dir = dest + '/test/class' + str(frame[11]) + '/'
                    shutil.copy(src + frame, write_dir + fname)

                else:
                    data_file.append(['train', 'class' + str(frame[11]), fname[0:16], 40])
                    write_dir = dest + '/train/class' + str(frame[11]) + '/'
                    shutil.copy(src + frame, write_dir + fname)

            frame_num += 1

        # train_or_test, classname, filename_no_ext, filename = video_parts
        # data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

        print(data_file[-1])

with open('data_file.csv', 'w') as fout:
    writer = csv.writer(fout)
    writer.writerows(data_file)

print("Extracted and wrote %d video files." % (len(data_file)))
#
# image_list = []
# for filename in glob.glob('/home/harry/eeg/data/timeseries_orig_imgs/*.png'):
#     print(filename)
#     #im=Image.open(filename)
#     #image_list.append(im)
