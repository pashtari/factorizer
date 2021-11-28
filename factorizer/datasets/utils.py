from typing import Callable, Dict, List, Optional, Sequence, Union
import os

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from monai.data import (
    CacheDataset,
    SmartCacheDataset,
    load_decathlon_datalist,
    load_decathlon_properties,
)
from monai.transforms import LoadImaged, Randomizable
from monai.utils.misc import ensure_tuple
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils import GridSampleMode, GridSamplePadMode, InterpolateMode
from monai.utils import ImageMetaKey as Key
from monai.utils import optional_import
from monai.data.png_saver import PNGSaver
from monai.data.nifti_writer import write_nifti


__all__ = [
    "StandardDataset",
    "SmartDataset",
    "SmartDatasetCallback",
    "NiftiSaver",
    "SaveImage",
    "SaveImaged",
    "SaveImageD",
]


###################################
# Dataset module
###################################


class StandardDataset(Randomizable, CacheDataset):
    """
    A standard dataset module based on :py:class:`monai.data.CacheDataset` to accelerate the training process.

    Args:
        root_dir: user's local directory for caching and loading the dataset.
        section: expected data section, can be: `training`, `validation` or `test`.
        transform: transforms to execute operations on input data.
            for further usage, use `AddChanneld` or `AsChannelFirstd` to convert the shape to [C, H, W, D].
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads to use.
            if 0 a single thread will be used. Default is 0.

    Raises:
        ValueError: When ``root_dir`` is not a directory.

    Example::

        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_data = StandardDataset(root_dir="./", transform=transform, seed=42)

        print(val_data[0]["image"], val_data[0]["label"])

    """

    def __init__(
        self,
        root_dir: str,
        section: str,
        transform: Optional[Union[Sequence[Callable], Callable]] = None,
        **kwargs,
    ) -> None:
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")

        self.section = section

        if transform is None:
            transform = LoadImaged(["image", "label"])

        self.indices: np.ndarray = np.array([])
        self.datalist = self._generate_data_list(root_dir)
        property_keys = [
            "name",
            "description",
            "reference",
            "licence",
            "tensorImageSize",
            "modality",
            "labels",
            "numTraining",
            "numTest",
        ]
        self._properties = load_decathlon_properties(
            os.path.join(root_dir, "dataset.json"), property_keys
        )

        CacheDataset.__init__(self, self.datalist, transform, **kwargs)

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.

        """
        return self.indices

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def get_properties(self, keys: Optional[Union[Sequence[str], str]] = None):
        """
        Get the loaded properties of dataset with specified keys.
        If no keys specified, return all the loaded properties.

        """
        if keys is None:
            return self._properties
        if self._properties is not None:
            return {key: self._properties[key] for key in ensure_tuple(keys)}
        return {}

    def _generate_data_list(self, root_dir: str) -> List[Dict]:
        section = (
            "training"
            if self.section in ["training", "validation"]
            else "test"
        )
        datalist = load_decathlon_datalist(
            os.path.join(root_dir, "dataset.json"), True, section
        )
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
        return datalist


class SmartDataset(SmartCacheDataset):
    """Add SmartCache functionality StandardDataset`.

    Args:
        root_dir: user's local directory for caching and loading the dataset.
        transform: transforms to be executed on input data.
        replace_rate: percentage of the cached items to be replaced in every epoch.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_init_workers: the number of worker threads to initialize the cache for first epoch.
            If num_init_workers is None then the number returned by os.cpu_count() is used.
        num_replace_workers: the number of worker threads to prepare the replacement cache for every epoch.
            If num_replace_workers is None then the number returned by os.cpu_count() is used.
        progress: whether to display a progress bar when caching for the first epoch.

    """

    def __init__(
        self,
        root_dir: str,
        section: str,
        transform: Union[Sequence[Callable], Callable],
        **kwargs,
    ) -> None:
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")

        self.section = section

        if transform is None:
            transform = LoadImaged(["image", "label"])

        self.indices: np.ndarray = np.array([])
        self.datalist = self._generate_data_list(root_dir)
        property_keys = [
            "name",
            "description",
            "reference",
            "licence",
            "tensorImageSize",
            "modality",
            "labels",
            "numTraining",
            "numTest",
        ]
        self._properties = load_decathlon_properties(
            os.path.join(root_dir, "dataset.json"), property_keys
        )
        SmartCacheDataset.__init__(self, self.datalist, transform, **kwargs)

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.

        """
        return self.indices

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def get_properties(self, keys: Optional[Union[Sequence[str], str]] = None):
        """
        Get the loaded properties of dataset with specified keys.
        If no keys specified, return all the loaded properties.

        """
        if keys is None:
            return self._properties
        if self._properties is not None:
            return {key: self._properties[key] for key in ensure_tuple(keys)}
        return {}

    def _generate_data_list(self, root_dir: str) -> List[Dict]:
        section = (
            "training"
            if self.section in ["training", "validation"]
            else "test"
        )
        datalist = load_decathlon_datalist(
            os.path.join(root_dir, "dataset.json"), True, section
        )
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
        return datalist


class SmartDatasetCallback(Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        trainer.datamodule.train_set.start()

    def on_train_epoch_end(self, trainer, pl_module, unused=None) -> None:
        trainer.datamodule.train_set.update_cache()

    def on_train_end(self, trainer, pl_module) -> None:
        trainer.datamodule.train_set.shutdown()


###################################
# Data reader and writer
###################################


nib, _ = optional_import("nibabel")
Image, _ = optional_import("PIL.Image")


def create_file_basename(
    postfix: str,
    input_file_name: str,
    folder_path: str,
    data_root_dir: str = "",
    patch_index: Optional[int] = None,
) -> str:
    """
    Utility function to create the path to the output file based on the input
    filename (file name extension is not added by this function).
    When `data_root_dir` is not specified, the output file name is:

        `folder_path/input_file_name (no ext.) /input_file_name (no ext.)[_postfix]`

    otherwise the relative path with respect to `data_root_dir` will be inserted, for example:
    input_file_name: /foo/bar/test1/image.png,
    postfix: seg
    folder_path: /output,
    data_root_dir: /foo/bar,
    output will be: /output/test1/image/image_seg

    Args:
        postfix: output name's postfix
        input_file_name: path to the input image file.
        folder_path: path for the output file
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. This is used to compute `input_file_rel_path`, the relative path to the file from
            `data_root_dir` to preserve folder structure when saving in case there are files in different
            folders with the same file names.
        patch_index: if not None, append the patch index to filename.
    """

    # get the filename and directory
    filedir, filename = os.path.split(input_file_name)
    # remove extension
    filename, ext = os.path.splitext(filename)
    if ext == ".gz":
        filename, ext = os.path.splitext(filename)
    # use data_root_dir to find relative path to file
    filedir_rel_path = ""
    if data_root_dir and filedir:
        filedir_rel_path = os.path.relpath(filedir, data_root_dir)

    # sub-folder path will be original name without the extension
    subfolder_path = os.path.join(folder_path, filedir_rel_path)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    if len(postfix) > 0:
        # add the sub-folder plus the postfix name to become the file basename in the output path
        output = os.path.join(subfolder_path, filename + "_" + postfix)
    else:
        output = os.path.join(subfolder_path, filename)

    if patch_index is not None:
        output += f"_{patch_index}"

    return os.path.abspath(output)


class NiftiSaver:
    """
    Save the data as NIfTI file, it can support single data content or a batch of data.
    Typically, the data can be segmentation predictions, call `save` for single data
    or call `save_batch` to save a batch of data together. If no meta data provided,
    use index from 0 as the filename prefix.

    NB: image should include channel dimension: [B],C,H,W,[D].
    """

    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "seg",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        print_log: bool = True,
    ) -> None:
        """
        Args:
            output_dir: output image directory.
            output_postfix: a string appended to all output file names.
            output_ext: output file extension name.
            resample: whether to resample before saving the data array.
            mode: {``"bilinear"``, ``"nearest"``}
                This option is used when ``resample = True``.
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                This option is used when ``resample = True``.
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data.
            output_dtype: data type for saving data. Defaults to ``np.float32``.
            squeeze_end_dims: if True, any trailing singleton dimensions will be removed (after the channel
                has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
                then if C==1, it will be saved as (H,W,D). If D also ==1, it will be saved as (H,W). If false,
                image will always be saved as (H,W,D,C).
            data_root_dir: if not empty, it specifies the beginning parts of the input file's
                absolute path. it's used to compute `input_file_rel_path`, the relative path to the file from
                `data_root_dir` to preserve folder structure when saving in case there are files in different
                folders with the same file names. for example:
                input_file_name: /foo/bar/test1/image.nii,
                postfix: seg
                output_ext: nii.gz
                output_dir: /output,
                data_root_dir: /foo/bar,
                output will be: /output/test1/image/image_seg.nii.gz
            print_log: whether to print log about the saved NIfTI file path, etc. default to `True`.

        """
        self.output_dir = output_dir
        self.output_postfix = output_postfix
        self.output_ext = output_ext
        self.resample = resample
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.align_corners = align_corners
        self.dtype = dtype
        self.output_dtype = output_dtype
        self._data_index = 0
        self.squeeze_end_dims = squeeze_end_dims
        self.data_root_dir = data_root_dir
        self.print_log = print_log

    def save(
        self,
        data: Union[torch.Tensor, np.ndarray],
        meta_data: Optional[Dict] = None,
    ) -> None:
        """
        Save data into a Nifti file.

        The meta_data could optionally have the following keys:

            - ``'filename_or_obj'`` -- for output file name creation, corresponding to filename or object.
            - ``'original_affine'`` -- for data orientation handling, defaulting to an identity matrix.
            - ``'affine'`` -- for data output affine, defaulting to an identity matrix.
            - ``'spatial_shape'`` -- for data output shape.
            - ``'patch_index'`` -- if the data is a patch of big image, append the patch index to filename.

        When meta_data is specified, the saver will try to resample batch data from the space
        defined by "affine" to the space defined by "original_affine".

        If meta_data is None, use the default index (starting from 0) as the filename.

        Args:
            data: target data content that to be saved as a NIfTI format file.
                Assuming the data shape starts with a channel dimension and followed by spatial dimensions.
            meta_data: the meta data information corresponding to the data.

        See Also
            :py:meth:`monai.data.nifti_writer.write_nifti`
        """
        filename = (
            meta_data[Key.FILENAME_OR_OBJ]
            if meta_data
            else str(self._data_index)
        )
        self._data_index += 1
        original_affine = (
            meta_data.get("original_affine", None) if meta_data else None
        )
        affine = meta_data.get("affine", None) if meta_data else None
        spatial_shape = (
            meta_data.get("spatial_shape", None) if meta_data else None
        )
        patch_index = (
            meta_data.get(Key.PATCH_INDEX, None) if meta_data else None
        )

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        path = create_file_basename(
            self.output_postfix,
            filename,
            self.output_dir,
            self.data_root_dir,
            patch_index,
        )
        path = f"{path}{self.output_ext}"
        # change data shape to be (channel, h, w, d)
        while len(data.shape) < 4:
            data = np.expand_dims(data, -1)
        # change data to "channel last" format and write to nifti format file
        data = np.moveaxis(np.asarray(data), 0, -1)

        # if desired, remove trailing singleton dimensions
        if self.squeeze_end_dims:
            while data.shape[-1] == 1:
                data = np.squeeze(data, -1)

        write_nifti(
            data,
            file_name=path,
            affine=affine,
            target_affine=original_affine,
            resample=self.resample,
            output_spatial_shape=spatial_shape,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            dtype=self.dtype,
            output_dtype=self.output_dtype,
        )

        if self.print_log:
            print(f"file written: {path}.")

    def save_batch(
        self,
        batch_data: Union[torch.Tensor, np.ndarray],
        meta_data: Optional[Dict] = None,
    ) -> None:
        """
        Save a batch of data into Nifti format files.

        Spatially it supports up to three dimensions, that is, H, HW, HWD for
        1D, 2D, 3D respectively (with resampling supports for 2D and 3D only).

        When saving multiple time steps or multiple channels `batch_data`,
        time and/or modality axes should be appended after the batch dimensions.
        For example, the shape of a batch of 2D eight-class
        segmentation probabilities to be saved could be `(batch, 8, 64, 64)`;
        in this case each item in the batch will be saved as (64, 64, 1, 8)
        NIfTI file (the third dimension is reserved as a spatial dimension).

        Args:
            batch_data: target batch data content that save into NIfTI format.
            meta_data: every key-value in the meta_data is corresponding to a batch of data.

        """
        for i, data in enumerate(batch_data):  # save a batch of files
            self.save(
                data=data,
                meta_data={k: meta_data[k][i] for k in meta_data}
                if meta_data is not None
                else None,
            )


class SaveImage(Transform):
    """
    Save transformed data into files, support NIfTI and PNG formats.

    It can work for both numpy array and PyTorch Tensor in both pre-transform chain
    and post transform chain.

    NB: image should include channel dimension: [B],C,H,W,[D].

    Args:
        output_dir: output image directory.
        output_postfix: a string appended to all output file names, default to `trans`.
        output_ext: output file extension name, available extensions: `.nii.gz`, `.nii`, `.png`.
        resample: whether to resample before saving the data array.
            if saving PNG format image, based on the `spatial_shape` from metadata.
            if saving NIfTI format image, based on the `original_affine` from metadata.
        mode: This option is used when ``resample = True``. Defaults to ``"nearest"``.

            - NIfTI files {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - PNG files {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.

            - NIfTI files {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - PNG files
                This option is ignored.

        scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
            [0, 255] (uint8) or [0, 65535] (uint16). Default is None to disable scaling.
            it's used for PNG format only.
        dtype: data type during resampling computation. Defaults to ``np.float64`` for best precision.
            if None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
            it's used for NIfTI format only.
        output_dtype: data type for saving data. Defaults to ``np.float32``.
            it's used for NIfTI format only.
        save_batch: whether the import image is a batch data, default to `False`.
            usually pre-transforms run for channel first data, while post-transforms run for batch data.
        squeeze_end_dims: if True, any trailing singleton dimensions will be removed (after the channel
            has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
            then if C==1, it will be saved as (H,W,D). If D also ==1, it will be saved as (H,W). If false,
            image will always be saved as (H,W,D,C).
            it's used for NIfTI format only.
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. it's used to compute `input_file_rel_path`, the relative path to the file from
            `data_root_dir` to preserve folder structure when saving in case there are files in different
            folders with the same file names. for example:
            input_file_name: /foo/bar/test1/image.nii,
            output_postfix: seg
            output_ext: nii.gz
            output_dir: /output,
            data_root_dir: /foo/bar,
            output will be: /output/test1/image/image_seg.nii.gz
        print_log: whether to print log about the saved file path, etc. default to `True`.

    """

    def __init__(
        self,
        output_dir: str = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: Union[GridSampleMode, InterpolateMode, str] = "nearest",
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        scale: Optional[int] = None,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
        save_batch: bool = False,
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        print_log: bool = True,
    ) -> None:
        self.saver: Union[NiftiSaver, PNGSaver]
        if output_ext in (".nii.gz", ".nii"):
            self.saver = NiftiSaver(
                output_dir=output_dir,
                output_postfix=output_postfix,
                output_ext=output_ext,
                resample=resample,
                mode=GridSampleMode(mode),
                padding_mode=padding_mode,
                dtype=dtype,
                output_dtype=output_dtype,
                squeeze_end_dims=squeeze_end_dims,
                data_root_dir=data_root_dir,
                print_log=print_log,
            )
        elif output_ext == ".png":
            self.saver = PNGSaver(
                output_dir=output_dir,
                output_postfix=output_postfix,
                output_ext=output_ext,
                resample=resample,
                mode=InterpolateMode(mode),
                scale=scale,
                data_root_dir=data_root_dir,
                print_log=print_log,
            )
        else:
            raise ValueError(f"unsupported output extension: {output_ext}.")

        self.save_batch = save_batch

    def __call__(
        self,
        img: Union[torch.Tensor, np.ndarray],
        meta_data: Optional[Dict] = None,
    ):
        """
        Args:
            img: target data content that save into file.
            meta_data: key-value pairs of meta_data corresponding to the data.

        """
        if self.save_batch:
            self.saver.save_batch(img, meta_data)
        else:
            self.saver.save(img, meta_data)


class SaveImaged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SaveImage`.

    Note:
        Image should include channel dimension: [B],C,H,W,[D].
        If the data is a patch of big image, will append the patch index to filename.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        meta_key_postfix: `key_{postfix}` was used to store the metadata in `LoadImaged`.
            so need the key to extract metadata to save images, default is `meta_dict`.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the meta data is a dictionary object which contains: filename, affine, original_shape, etc.
            if no corresponding metadata, set to `None`.
        output_dir: output image directory.
        output_postfix: a string appended to all output file names, default to `trans`.
        output_ext: output file extension name, available extensions: `.nii.gz`, `.nii`, `.png`.
        resample: whether to resample before saving the data array.
            if saving PNG format image, based on the `spatial_shape` from metadata.
            if saving NIfTI format image, based on the `original_affine` from metadata.
        mode: This option is used when ``resample = True``. Defaults to ``"nearest"``.

            - NIfTI files {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - PNG files {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate

        padding_mode: This option is used when ``resample = True``. Defaults to ``"border"``.

            - NIfTI files {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            - PNG files
                This option is ignored.

        scale: {``255``, ``65535``} postprocess data by clipping to [0, 1] and scaling
            [0, 255] (uint8) or [0, 65535] (uint16). Default is None to disable scaling.
            it's used for PNG format only.
        dtype: data type during resampling computation. Defaults to ``np.float64`` for best precision.
            if None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
            it's used for NIfTI format only.
        output_dtype: data type for saving data. Defaults to ``np.float32``.
            it's used for NIfTI format only.
        save_batch: whether the import image is a batch data, default to `False`.
            usually pre-transforms run for channel first data, while post-transforms run for batch data.
        allow_missing_keys: don't raise exception if key is missing.
        squeeze_end_dims: if True, any trailing singleton dimensions will be removed (after the channel
            has been moved to the end). So if input is (C,H,W,D), this will be altered to (H,W,D,C), and
            then if C==1, it will be saved as (H,W,D). If D also ==1, it will be saved as (H,W). If false,
            image will always be saved as (H,W,D,C).
            it's used for NIfTI format only.
        data_root_dir: if not empty, it specifies the beginning parts of the input file's
            absolute path. it's used to compute `input_file_rel_path`, the relative path to the file from
            `data_root_dir` to preserve folder structure when saving in case there are files in different
            folders with the same file names. for example:
            input_file_name: /foo/bar/test1/image.nii,
            output_postfix: seg
            output_ext: nii.gz
            output_dir: /output,
            data_root_dir: /foo/bar,
            output will be: /output/test1/image/image_seg.nii.gz
        print_log: whether to print log about the saved file path, etc. default to `True`.

    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_key_postfix: str = "meta_dict",
        output_dir: str = "./",
        output_postfix: str = "trans",
        output_ext: str = ".nii.gz",
        resample: bool = True,
        mode: Union[GridSampleMode, InterpolateMode, str] = "nearest",
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        scale: Optional[int] = None,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike = np.float32,
        save_batch: bool = False,
        allow_missing_keys: bool = False,
        squeeze_end_dims: bool = True,
        data_root_dir: str = "",
        print_log: bool = True,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.meta_key_postfix = meta_key_postfix
        self._saver = SaveImage(
            output_dir=output_dir,
            output_postfix=output_postfix,
            output_ext=output_ext,
            resample=resample,
            mode=mode,
            padding_mode=padding_mode,
            scale=scale,
            dtype=dtype,
            output_dtype=output_dtype,
            save_batch=save_batch,
            squeeze_end_dims=squeeze_end_dims,
            data_root_dir=data_root_dir,
            print_log=print_log,
        )

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            meta_data = (
                d[f"{key}_{self.meta_key_postfix}"]
                if self.meta_key_postfix is not None
                else None
            )
            self._saver(img=d[key], meta_data=meta_data)
        return d


SaveImageD = SaveImageDict = SaveImaged

