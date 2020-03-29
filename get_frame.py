import cv2
import os
import argparse
from tqdm import tqdm

def Dataset(opt):
    data_root="/disk1/shchen/IsoGD/" + opt.Dir
    frm_root = '/disk1/shchen/dataset/'

    for file_list in os.listdir(data_root):

        pbar = tqdm(total=len(os.listdir(os.path.join(data_root, file_list))))

        for filename in os.listdir(os.path.join(data_root,file_list)):

            M = filename.split("_")
            modiality = "rgb" if M[0] == 'M' else "depth"

            frm_filename = filename.split('.')
            video_save_path = os.path.join(frm_root, modiality, opt.Dir, file_list,frm_filename[0])
            if not os.path.exists(video_save_path): os.makedirs(video_save_path)

            video_load_path = data_root+'/'+file_list+'/'+filename
            # print(video_load_path)
            cap = cv2.VideoCapture(video_load_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_s = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            frame_count = 0
            success = True
            while (success):
                success, frame = cap.read()
                if success is False:
                    # print('The video is empty!')
                    break

                #cut down start and end
                # if 3<frame_count<=total_s-5:
                cv2.imwrite(video_save_path + "/%06d.jpg" % frame_count, frame)

                frame_count = frame_count + 1
            cap.release()
            # print("The video: {} is Done!".format(filename))

            pbar.update(1)
        pbar.close()
        print('The end in dir {}'.format(file_list))

def main():

    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-Dir', type=str)
    opt = parser.parse_args()

    Dataset(opt)




if __name__ == '__main__':
    main()
