import cv2
import numpy as np
import pathlib
import re

from torch.utils.data import Dataset

class OcrDataset(Dataset):
    def __init__(self, path_to_data: str, transforms=None):
        # TODO: Here you can create samples from dirs and initialize transfroms
        self.__path_to_data = path_to_data
        self.__transforms = transforms
        self.__all_posix_filenames = self.__get_all_posix_filenames()
        self.__all_str_filenames = self.__get_all_str_filenames()
        self.__region_codes = self.__get_region_codes()


    def __get_region_codes(self):
        region_codes = list(np.arange(1, 100))
        region_codes.remove(20)
        region_codes.extend([102, 113, 116, 716, 121, 123, 193, 124, 125, 126, 134, 136,
                             138, 142, 147, 150, 190, 750, 152, 154, 156, 159, 161, 761, 163,
                             763, 164, 196, 173, 174, 177, 197, 199, 777, 797, 799, 178, 198, 186])

        region_codes = set(['0' + str(item) if len(str(item)) == 1 else str(item) for item in region_codes])
        return region_codes

    def __get_all_posix_filenames(self):
        dataset_paths = pathlib.Path(self.__path_to_data)
        return list(dataset_paths.glob('**/*'))

    def __get_all_str_filenames(self):
        return [str(file.name) for file in self.__all_posix_filenames if file.is_file()]

    def get_train_sample(self, path_to_file: str):
        img = cv2.imread(path_to_file)
        file = pathlib.Path(path_to_file)

        if file.is_file():
            name = file.name

            if '_nolabel' in name:
                return img, ""

            name = re.sub('\\s\\(2\\)', '', name)
            found = re.search('[A-Z]\d{3}[A-Z]{2}\\s\d{2,5}', name)

            if found:
                carnumber = found.group(0)
                region = carnumber.split(' ')[-1]

                if len(region) == 5:
                    carnumber = carnumber[:-2]

                elif len(region) == 4:
                    if region[:-1] in self.__region_codes:
                        carnumber = carnumber[:-1]

                    elif region[:-2] in self.__region_codes:
                        carnumber = carnumber[:-2]

                elif len(region) == 3:
                    splt = name.split('.')
                    short_name = splt[0][:-1] + '.' + splt[1]

                    if short_name in self.__all_str_filenames:
                        carnumber = short_name.split('.')[0]

                return img, carnumber
            else:
                return img, ""

