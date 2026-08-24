from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, uniform_filter, convolve


class SWOTInternalWaveDetector:
    """Detect internal waves in SWOT pass xarray data."""

    def __init__(
        self,
        lat_min,
        lat_max,
        model_path=None,
        base_var="ssha_unfiltered",
        quality_var="quality_flag",
        pixel_size_km=2.0,
        gaussian_sigma=8,
        rolling_window=None,
        inner_radius_km=5,
        middle_radius_km=15,
        outer_radius_km=30,
        inner_weight=20.0,
        middle_weight=0.5,
        outer_weight=0.2,
    ):
        self.lat_min = lat_min
        self.lat_max = lat_max

        self.model_path = model_path
        self.model = None

        if model_path is not None:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))

        self.base_var = base_var
        self.quality_var = quality_var

        self.pixel_size_km = pixel_size_km
        self.gaussian_sigma = gaussian_sigma
        self.rolling_window = rolling_window or int(gaussian_sigma) * 2

        self.inner_radius_km = inner_radius_km
        self.middle_radius_km = middle_radius_km
        self.outer_radius_km = outer_radius_km

        self.inner_weight = inner_weight
        self.middle_weight = middle_weight
        self.outer_weight = outer_weight

        self.stepped_kernel = self.build_stepped_kernel()

    # ============================================================
    # Clipping
    # ============================================================

    def clip_pass_by_lat_only_keep_width(self, l3_file):
        """
        Open full L3 file and clip only by latitude along num_lines.

        This preserves the full SWOT swath width / num_pixels dimension.
        It does NOT mask or crop by longitude.
        """
        ds = xr.open_dataset(l3_file)

        lat_name = "latitude" if "latitude" in ds else "lat"
        lat = ds[lat_name]

        lat_mask = (lat >= self.lat_min) & (lat <= self.lat_max)

        if "num_lines" in lat_mask.dims and "num_pixels" in lat_mask.dims:
            line_mask = lat_mask.any(dim="num_pixels")
            line_idx = np.where(line_mask.values)[0]

            if line_idx.size == 0:
                ds.close()
                raise ValueError(f"No intersecting latitude lines found for {l3_file}")

            first_line = int(line_idx[0])
            last_line = int(line_idx[-1]) + 1

            ds_clip = ds.isel(num_lines=slice(first_line, last_line)).load()

        elif "num_lines" in lat_mask.dims:
            line_idx = np.where(lat_mask.values)[0]

            if line_idx.size == 0:
                ds.close()
                raise ValueError(f"No intersecting latitude lines found for {l3_file}")

            first_line = int(line_idx[0])
            last_line = int(line_idx[-1]) + 1

            ds_clip = ds.isel(num_lines=slice(first_line, last_line)).load()

        else:
            ds_clip = ds.where(lat_mask, drop=True).load()

        ds.close()

        ds_clip.attrs["region"] = "Bay of Bengal"
        ds_clip.attrs["clip_type"] = "latitude_only_full_width"
        ds_clip.attrs["clip_lat_min"] = float(self.lat_min)
        ds_clip.attrs["clip_lat_max"] = float(self.lat_max)
        ds_clip.attrs["note"] = (
            "Clipped only by latitude range. Full cross-track width was preserved."
        )
        ds_clip.attrs["source_l3_file"] = str(l3_file)

        return ds_clip

    # ============================================================
    # Basic array helpers
    # ============================================================

    def get_2d_array(self, da):
        """
        Convert an xarray DataArray to a clean 2D numpy array.
        """
        arr = da.values
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array after squeeze, got shape {arr.shape}")

        return arr

    def make_vertical(self, arr):
        """
        Make long dimension vertical while preserving pixel aspect ratio.
        """
        if arr.shape[1] > arr.shape[0]:
            return arr.T

        return arr

    def get_transform_info(self, arr):
        """
        Store how the original 2D SWOT array will be transformed.

        Needed to map YOLO boxes back to the original clipped xarray grid.
        """
        original_shape = arr.shape
        was_transposed = arr.shape[1] > arr.shape[0]

        if was_transposed:
            transformed_shape = arr.T.shape
        else:
            transformed_shape = arr.shape

        return {
            "original_shape": original_shape,
            "transformed_shape": transformed_shape,
            "was_transposed": was_transposed,
            "was_flipped_lr": True,
        }

    # ============================================================
    # Quality mask
    # ============================================================

    def apply_quality_flag_mask(self, ssha, quality_flag):
        """
        Keep only quality_flag values 0, 5, 30, and 50.
        Everything else becomes NaN.
        """
        ssha = ssha.astype(np.float32).copy()

        good_quality = (
            (quality_flag == 0)
            | (quality_flag == 5)
            | (quality_flag == 30)
            | (quality_flag == 50)
        )

        ssha[~good_quality] = np.nan

        return ssha

    def prepare_base_array(self, ds):
        """
        Extract SSHA and quality flag arrays, apply quality masking,
        and remove unrealistic values.

        Returns a 2D array aligned to the clipped xarray grid.
        """
        ds_plot = ds.isel(file=0) if "file" in ds.dims else ds

        if self.base_var not in ds_plot:
            raise KeyError(f"Missing base variable: {self.base_var}")

        if self.quality_var not in ds_plot:
            raise KeyError(f"Missing quality variable: {self.quality_var}")

        ssha = self.get_2d_array(ds_plot[self.base_var]).astype(np.float32)
        quality_flag = self.get_2d_array(ds_plot[self.quality_var])

        if ssha.shape != quality_flag.shape:
            raise ValueError(
                f"Shape mismatch: {self.base_var} shape={ssha.shape}, "
                f"{self.quality_var} shape={quality_flag.shape}"
            )

        ssha = self.apply_quality_flag_mask(ssha, quality_flag)

        # Mask unrealistic/bad values.
        ssha = np.where(ssha <= 10, ssha, np.nan).astype(np.float32)

        return ssha

    # ============================================================
    # NaN-safe filters
    # ============================================================

    def nan_uniform_filter(self, arr, window):
        """
        Uniform/rolling mean filter that handles NaNs.
        """
        arr = arr.astype(np.float32)

        valid = np.isfinite(arr)
        arr_filled = np.where(valid, arr, 0.0).astype(np.float32)
        weights = valid.astype(np.float32)

        smooth_sum = uniform_filter(
            arr_filled,
            size=window,
            mode="nearest",
        )

        weight_sum = uniform_filter(
            weights,
            size=window,
            mode="nearest",
        )

        smooth = smooth_sum / np.maximum(weight_sum, 1e-6)
        smooth[weight_sum <= 1e-6] = np.nan

        return smooth.astype(np.float32)

    def nan_gaussian_filter(self, arr, sigma):
        """
        Gaussian filter that handles NaNs.
        """
        arr = arr.astype(np.float32)

        valid = np.isfinite(arr)
        arr_filled = np.where(valid, arr, 0.0).astype(np.float32)
        weights = valid.astype(np.float32)

        smooth_sum = gaussian_filter(
            arr_filled,
            sigma=sigma,
            mode="nearest",
        )

        weight_sum = gaussian_filter(
            weights,
            sigma=sigma,
            mode="nearest",
        )

        smooth = smooth_sum / np.maximum(weight_sum, 1e-6)
        smooth[weight_sum <= 1e-6] = np.nan

        return smooth.astype(np.float32)

    def build_stepped_kernel(self):
        """
        Build stepped radial convolution kernel.
        """
        inner_radius_px = self.inner_radius_km / self.pixel_size_km
        middle_radius_px = self.middle_radius_km / self.pixel_size_km
        outer_radius_px = self.outer_radius_km / self.pixel_size_km

        kernel_radius_px = int(np.ceil(outer_radius_px))

        y, x = np.ogrid[
            -kernel_radius_px : kernel_radius_px + 1,
            -kernel_radius_px : kernel_radius_px + 1,
        ]

        r = np.sqrt(x**2 + y**2)

        kernel = np.zeros_like(r, dtype=np.float32)

        kernel[r <= inner_radius_px] = self.inner_weight

        kernel[
            (r > inner_radius_px)
            & (r <= middle_radius_px)
        ] = self.middle_weight

        kernel[
            (r > middle_radius_px)
            & (r <= outer_radius_px)
        ] = self.outer_weight

        kernel_sum = kernel.sum()

        if kernel_sum <= 0:
            raise ValueError("Stepped convolution kernel sum is zero.")

        kernel = kernel / kernel_sum

        return kernel.astype(np.float32)

    def nan_stepped_convolution_filter(self, arr, kernel):
        """
        Stepped convolution filter that handles NaNs.
        """
        arr = arr.astype(np.float32)

        valid = np.isfinite(arr)
        arr_filled = np.where(valid, arr, 0.0).astype(np.float32)
        weights = valid.astype(np.float32)

        smooth_sum = convolve(
            arr_filled,
            weights=kernel,
            mode="nearest",
        )

        weight_sum = convolve(
            weights,
            weights=kernel,
            mode="nearest",
        )

        smooth = smooth_sum / np.maximum(weight_sum, 1e-6)
        smooth[weight_sum <= 1e-6] = np.nan

        return smooth.astype(np.float32)

    # ============================================================
    # High-pass filters
    # ============================================================

    def rolling_highpass(self, arr):
        """
        High-pass = original - rolling/uniform low-pass.
        """
        arr = arr.astype(np.float32)

        valid = np.isfinite(arr)
        smooth = self.nan_uniform_filter(arr, window=self.rolling_window)

        highpass = arr - smooth
        highpass[~valid] = np.nan

        return highpass.astype(np.float32)

    def gaussian_highpass(self, arr):
        """
        High-pass = original - Gaussian low-pass.
        """
        arr = arr.astype(np.float32)

        valid = np.isfinite(arr)
        smooth = self.nan_gaussian_filter(arr, sigma=self.gaussian_sigma)

        highpass = arr - smooth
        highpass[~valid] = np.nan

        return highpass.astype(np.float32)

    def stepped_convolution_highpass(self, arr):
        """
        High-pass = original - stepped-convolution low-pass.
        """
        arr = arr.astype(np.float32)

        valid = np.isfinite(arr)

        smooth = self.nan_stepped_convolution_filter(
            arr,
            kernel=self.stepped_kernel,
        )

        highpass = arr - smooth
        highpass[~valid] = np.nan

        return highpass.astype(np.float32)

    # ============================================================
    # Filter selection and YOLO orientation
    # ============================================================

    def apply_selected_filter(self, arr, filter_type="normal"):
        """
        Apply one selected filter while keeping the array aligned to
        the original clipped xarray grid.

        This is the array that should be saved back into the xarray dataset.

        filter_type options:
        - "normal"
        - "rolling"
        - "gaussian"
        - "stepped"
        """
        filter_type = filter_type.lower()

        if filter_type in ["normal", "unfiltered", "none"]:
            filtered = arr

        elif filter_type in ["rolling", "rolling_mean", "rolling_highpass"]:
            filtered = self.rolling_highpass(arr)

        elif filter_type in ["gaussian", "gaussian_filter", "gaussian_highpass"]:
            filtered = self.gaussian_highpass(arr)

        elif filter_type in [
            "stepped",
            "stepped_convolution",
            "stepped_convolution_highpass",
        ]:
            filtered = self.stepped_convolution_highpass(arr)

        else:
            raise ValueError(
                f"Unknown filter_type: {filter_type}. "
                "Use 'normal', 'rolling', 'gaussian', or 'stepped'."
            )

        return filtered.astype(np.float32)

    def orient_array_for_yolo(self, arr):
        """
        Apply the image-orientation transform used for YOLO.

        This output is NOT saved to xarray because it may be transposed
        and flipped relative to the original SWOT grid.
        """
        transformed = self.make_vertical(arr)
        transformed = np.fliplr(transformed)

        return transformed.astype(np.float32)

    def transform_array(self, arr, filter_type="normal"):
        """
        Apply selected filter, then orient the result for YOLO.

        This method preserves the old behavior of transform_array.
        """
        filtered = self.apply_selected_filter(
            arr,
            filter_type=filter_type,
        )

        transformed = self.orient_array_for_yolo(filtered)

        return transformed.astype(np.float32)

    def transform_dataset_for_yolo(
        self,
        ds,
        filter_type="normal",
        return_filtered=False,
    ):
        """
        Prepare clipped SWOT dataset for YOLO.

        Returns
        -------
        transformed_arr : np.ndarray
            Filtered array after YOLO image orientation.
        transform_info : dict
            Metadata needed to map YOLO boxes back to original grid.
        filtered_arr : np.ndarray, optional
            Filtered array still aligned to the original clipped xarray grid.
            Returned only when return_filtered=True.
        """
        base_arr = self.prepare_base_array(ds)
        transform_info = self.get_transform_info(base_arr)

        filtered_arr = self.apply_selected_filter(
            base_arr,
            filter_type=filter_type,
        )

        transformed_arr = self.orient_array_for_yolo(filtered_arr)

        if return_filtered:
            return transformed_arr, transform_info, filtered_arr

        return transformed_arr, transform_info

    def transform_l3_file(self, l3_file, filter_type="normal"):
        """
        Clip L3 file by latitude, then transform it.
        """
        ds_clip = self.clip_pass_by_lat_only_keep_width(l3_file)

        try:
            transformed_arr, transform_info = self.transform_dataset_for_yolo(
                ds_clip,
                filter_type=filter_type,
            )
        finally:
            ds_clip.close()

        return transformed_arr, transform_info

    # ============================================================
    # Image creation
    # ============================================================

    def array_to_pil_image(self, arr, cmap="magma"):
        """
        Convert transformed filtered SWOT array into a PIL RGB image for YOLO.

        Uses matplotlib colormap.
        NaN/masked pixels are white.
        """
        arr = arr.astype(np.float32).copy()

        # Mask unrealistic/bad values.
        arr = np.where(arr <= 10, arr, np.nan)

        if not np.isfinite(arr).any():
            raise ValueError("Cannot convert array to image because all values are NaN.")

        vmin, vmax = np.nanpercentile(arr, [2.5, 97.5])

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            raise ValueError(f"Bad image scaling limits: vmin={vmin}, vmax={vmax}")

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="white")

        arr_masked = np.ma.masked_invalid(arr)

        normalized = (arr_masked - vmin) / (vmax - vmin)
        normalized = np.ma.clip(normalized, 0, 1)

        rgba = cmap_obj(normalized)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

        return Image.fromarray(rgb)

    # ============================================================
    # YOLO detection
    # ============================================================

    def run_yolo_detection(self, pil_image, conf=0.25):
        """
        Run YOLO detection on a PIL image.
        """
        if self.model is None:
            raise ValueError(
                "No YOLO model loaded. Pass model_path when creating the class."
            )

        results = self.model.predict(
            pil_image,
            conf=conf,
        )

        return results[0]

    def yolo_results_to_text(self, result):
        """
        Convert YOLO detection results into simple text labels.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return "No detections."

        lines = []

        names = result.names

        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            label = names.get(cls_id, str(cls_id))

            line = (
                f"{i}: {label} "
                f"conf={conf:.3f} "
                f"bbox_xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )

            lines.append(line)

        return "\n".join(lines)

    def draw_yolo_bounding_boxes(self, pil_image, result):
        """
        Draw YOLO bounding boxes on a copy of the PIL image.
        """
        boxed_image = pil_image.copy()
        draw = ImageDraw.Draw(boxed_image)

        if result.boxes is None or len(result.boxes) == 0:
            return boxed_image

        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            label = names.get(cls_id, str(cls_id))
            text = f"{label} {conf:.2f}"

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3,
            )

            draw.text(
                (x1, max(0, y1 - 14)),
                text,
                fill="red",
            )

        return boxed_image

    # ============================================================
    # YOLO boxes back to original SWOT grid
    # ============================================================

    def yolo_boxes_to_original_mask(self, result, transform_info):
        """
        Convert YOLO bounding boxes from transformed image coordinates
        back to the original clipped SWOT pixel grid.
        """
        transformed_shape = transform_info["transformed_shape"]
        original_shape = transform_info["original_shape"]
        was_transposed = transform_info["was_transposed"]

        mask_transformed = np.zeros(transformed_shape, dtype=np.uint8)

        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros(original_shape, dtype=np.uint8)

        height, width = transformed_shape

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = int(np.floor(x1))
            y1 = int(np.floor(y1))
            x2 = int(np.ceil(x2))
            y2 = int(np.ceil(y2))

            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 >= x1 and y2 >= y1:
                mask_transformed[y1 : y2 + 1, x1 : x2 + 1] = 1

        # Undo left-right flip.
        mask_unflipped = np.fliplr(mask_transformed)

        # Undo transpose if make_vertical() transposed the original array.
        if was_transposed:
            mask_original = mask_unflipped.T
        else:
            mask_original = mask_unflipped

        if mask_original.shape != original_shape:
            raise ValueError(
                f"Mask shape mismatch. Expected {original_shape}, "
                f"got {mask_original.shape}"
            )

        return mask_original.astype(np.uint8)

    def add_yolo_bbox_mask_to_dataset(
        self,
        clipped_ds,
        mask,
        mask_var_name="internal_wave_bbox_mask",
    ):
        """
        Add YOLO bounding-box mask back onto the clipped xarray dataset.
        """
        ds_out = clipped_ds.copy()

        ds_plot = ds_out.isel(file=0) if "file" in ds_out.dims else ds_out

        if self.base_var not in ds_plot:
            raise KeyError(f"Missing base variable: {self.base_var}")

        base_da = ds_plot[self.base_var].squeeze()

        if base_da.ndim != 2:
            raise ValueError(
                f"Expected 2D base variable after squeeze, got dims {base_da.dims}"
            )

        if mask.shape != base_da.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match base data shape {base_da.shape}"
            )

        mask_da = xr.DataArray(
            mask.astype(np.uint8),
            dims=base_da.dims,
            coords={
                dim: base_da.coords[dim]
                for dim in base_da.dims
                if dim in base_da.coords
            },
            name=mask_var_name,
            attrs={
                "description": "Pixels inside YOLO internal-wave bounding boxes.",
                "mask_values": "1 = inside YOLO bounding box, 0 = outside",
            },
        )

        ds_out[mask_var_name] = mask_da

        return ds_out

    # ============================================================
    # Save filtered data back to xarray
    # ============================================================

    def make_filtered_var_name(self, filter_type):
        """
        Create a clean default variable name for the saved filtered data.
        """
        filter_type = filter_type.lower()

        if filter_type in ["normal", "unfiltered", "none"]:
            suffix = "quality_masked"

        elif filter_type in ["rolling", "rolling_mean", "rolling_highpass"]:
            suffix = "rolling_highpass"

        elif filter_type in ["gaussian", "gaussian_filter", "gaussian_highpass"]:
            suffix = "gaussian_highpass"

        elif filter_type in [
            "stepped",
            "stepped_convolution",
            "stepped_convolution_highpass",
        ]:
            suffix = "stepped_convolution_highpass"

        else:
            raise ValueError(
                f"Unknown filter_type: {filter_type}. "
                "Use 'normal', 'rolling', 'gaussian', or 'stepped'."
            )

        return f"{self.base_var}_{suffix}"

    def add_filtered_array_to_dataset(
        self,
        clipped_ds,
        filtered_arr,
        filter_type="normal",
        filtered_var_name=None,
    ):
        """
        Add the filtered SSHA array back onto the clipped xarray dataset.

        Important:
        This saves the grid-aligned filtered array, not the YOLO-oriented
        transformed image array.
        """
        ds_out = clipped_ds.copy()

        ds_plot = ds_out.isel(file=0) if "file" in ds_out.dims else ds_out

        if self.base_var not in ds_plot:
            raise KeyError(f"Missing base variable: {self.base_var}")

        base_da = ds_plot[self.base_var].squeeze()

        if base_da.ndim != 2:
            raise ValueError(
                f"Expected 2D base variable after squeeze, got dims {base_da.dims}"
            )

        if filtered_arr.shape != base_da.shape:
            raise ValueError(
                f"Filtered array shape {filtered_arr.shape} does not match "
                f"base data shape {base_da.shape}"
            )

        if filtered_var_name is None:
            filtered_var_name = self.make_filtered_var_name(filter_type)

        filtered_da = xr.DataArray(
            filtered_arr.astype(np.float32),
            dims=base_da.dims,
            coords={
                dim: base_da.coords[dim]
                for dim in base_da.dims
                if dim in base_da.coords
            },
            name=filtered_var_name,
            attrs={
                "description": (
                    "Filtered SSHA array used to create the YOLO detection image. "
                    "This array is aligned to the original clipped xarray grid."
                ),
                "source_variable": self.base_var,
                "quality_variable": self.quality_var,
                "filter_type": str(filter_type),
                "orientation": "original_xarray_grid",
                "note": (
                    "This is not the transposed/flipped YOLO image array. "
                    "It matches the original clipped SWOT grid."
                ),
            },
        )

        ds_out[filtered_var_name] = filtered_da

        return ds_out

    # ============================================================
    # Full detection wrappers
    # ============================================================

    def detect_yolo_on_clipped_dataset(
        self,
        clipped_ds,
        filter_type="normal",
        cmap="magma",
        conf=0.25,
        mask_var_name="internal_wave_bbox_mask",
        filtered_var_name=None,
        save_filtered=True,
    ):
        """
        Run the full YOLO detection pipeline on an already-clipped dataset.

        Returns
        -------
        ds_with_outputs : xr.Dataset
            Clipped dataset with:
            - added YOLO bounding-box mask variable
            - added filtered SSHA variable, if save_filtered=True

        yolo_text : str
            Text summary of YOLO detections.

        boxed_image : PIL.Image.Image
            Image with YOLO bounding boxes drawn.

        result : ultralytics result
            Raw YOLO result object.
        """
        transformed_arr, transform_info, filtered_arr = self.transform_dataset_for_yolo(
            clipped_ds,
            filter_type=filter_type,
            return_filtered=True,
        )

        pil_image = self.array_to_pil_image(
            transformed_arr,
            cmap=cmap,
        )

        result = self.run_yolo_detection(
            pil_image,
            conf=conf,
        )

        yolo_text = self.yolo_results_to_text(result)

        boxed_image = self.draw_yolo_bounding_boxes(
            pil_image,
            result,
        )

        mask = self.yolo_boxes_to_original_mask(
            result,
            transform_info,
        )

        ds_with_outputs = self.add_yolo_bbox_mask_to_dataset(
            clipped_ds=clipped_ds,
            mask=mask,
            mask_var_name=mask_var_name,
        )

        if save_filtered:
            ds_with_outputs = self.add_filtered_array_to_dataset(
                clipped_ds=ds_with_outputs,
                filtered_arr=filtered_arr,
                filter_type=filter_type,
                filtered_var_name=filtered_var_name,
            )

        ds_with_outputs.attrs["yolo_filter_type"] = str(filter_type)
        ds_with_outputs.attrs["yolo_mask_variable"] = str(mask_var_name)

        if save_filtered:
            if filtered_var_name is None:
                filtered_var_name = self.make_filtered_var_name(filter_type)

            ds_with_outputs.attrs["saved_filtered_variable"] = str(filtered_var_name)

        return ds_with_outputs, yolo_text, boxed_image, result

    def detect_yolo_on_l3_file(
        self,
        l3_file,
        filter_type="normal",
        cmap="magma",
        conf=0.25,
        mask_var_name="internal_wave_bbox_mask",
        filtered_var_name=None,
        save_filtered=True,
    ):
        """
        Clip an L3 file by latitude, run YOLO detection, and return
        the clipped dataset with:
        - a YOLO bounding-box mask
        - the filtered SSHA data used for detection, if save_filtered=True
        """
        clipped_ds = self.clip_pass_by_lat_only_keep_width(l3_file)

        try:
            ds_with_outputs, yolo_text, boxed_image, result = (
                self.detect_yolo_on_clipped_dataset(
                    clipped_ds=clipped_ds,
                    filter_type=filter_type,
                    cmap=cmap,
                    conf=conf,
                    mask_var_name=mask_var_name,
                    filtered_var_name=filtered_var_name,
                    save_filtered=save_filtered,
                )
            )
        finally:
            clipped_ds.close()

        return ds_with_outputs, yolo_text, boxed_image, result