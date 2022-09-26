import os
from glob import glob
from os.path import isfile

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from wholeslidedata import WholeSlideAnnotation, WholeSlideImage
from wholeslidedata.accessories.asap.parser import AsapAnnotationParser

from nnunet.training.dataloading.dataset_loading import DataLoader2D


class DataLoader2DROIsMultiScale(DataLoader2D):
    def __init__(self, data_origin, spacing, crop_to_patch_size=True, training=True, key_to_class=None, *args,
                 **kwargs):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        self.training = training
        self.crop_to_patch_size = crop_to_patch_size
        self.data_origin = data_origin
        self.spacing = spacing
        self.key_to_class = key_to_class
        super(DataLoader2DROIsMultiScale, self).__init__(*args, **kwargs)

    def __next__(self):
        return self.generate_train_batch() if self.training else self.generate_single_roi()

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        b, c, w, h = self.data_shape
        data = np.zeros((b, 2, c, w, h), dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data'][:, None][:, 0]
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)[:, 0]

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint
            need_to_pad = self.need_to_pad.copy()
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = 0  # Don't sample below this or you will get black borders
            # (future: sample ROIs larger, since CP has a lot of WSI)
            ub_x = shape[0] - self.patch_size[0]
            lb_y = 0
            ub_y = shape[1] - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            width, height = case_all_data.shape[-2:]

            offset_x = (valid_bbox_x_lb + (valid_bbox_x_ub - valid_bbox_x_lb) // 2) - width // 2
            offset_y = (valid_bbox_y_lb + (valid_bbox_y_ub - valid_bbox_y_lb) // 2) - height // 2

            case_properties[j]['offset_x'] = int(offset_y)
            case_properties[j]['offset_y'] = int(offset_x)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})
            if self.key_to_class:
                case_properties[j]['context_class'] = self.key_to_class[selected_keys[j]]
            data[j] = np.stack([case_all_data_donly,
                                self.sample_context(case_properties[j], selected_keys[j],
                                                    case_all_data_donly.shape[-2:])])
            seg[j] = case_all_data_segonly

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys}

    def generate_single_roi(self):
        if not len(self.list_of_keys):
            raise StopIteration
        selected_key = self.list_of_keys.pop()

        if 'properties' in self._data[selected_key].keys():
            properties = self._data[selected_key]['properties']
        else:
            properties = load_pickle(self._data[selected_key]['properties_file'])

        if self.key_to_class:
            properties['context_class'] = self.key_to_class[selected_key]

        if not isfile(self._data[selected_key]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[selected_key]['data_file'][:-4] + ".npz")['data'][:, None][:, 0]
        else:
            case_all_data = np.load(self._data[selected_key]['data_file'][:-4] + ".npy", self.memmap_mode)[:, 0]

        c, w, h = case_all_data.shape

        data = np.stack([case_all_data[:-1], self.sample_context(properties, selected_key, (w, h))])

        return {'data': data[np.newaxis],
                'seg': np.array(case_all_data[-1:])[np.newaxis],
                'properties': properties,
                "keys": [selected_key]}

    def sample_context(self, props, key, shape=None):
        parser = AsapAnnotationParser(labels={'none': 0}, sample_label_names=['none'])
        anno_number = int(key.split("_")[-1].strip("ROI"))
        file_identifier = os.path.join(self.data_origin, '_'.join(key.split('_')[:-1]))
        wsa = WholeSlideAnnotation(glob(f"{file_identifier}.xml")[0], parser=parser)
        wsi = WholeSlideImage(glob(f"{file_identifier}.tif")[0], backend='asap')
        anno = wsa.sampling_annotations[anno_number]
        x, y = anno.center
        x += props.get("offset_x", 0)
        y += props.get("offset_y", 0)
        return wsi.get_patch(x, y, *shape[::-1], spacing=self.spacing).transpose(2, 0, 1) / 255.
