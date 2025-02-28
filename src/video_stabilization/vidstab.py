import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vidstab import VidStab, layer_overlay, layer_blend


class VidStabWrapper:
    """
    Wrapper class for the vidstab module functionality.
    Provides a simplified interface for video stabilization.
    """

    def __init__(self, kp_method="GFTT", threshold=None, nonmaxSuppression=None):
        """
        Initialize video stabilizer with specified keypoint detection method.

        Args:
            kp_method (str): Keypoint detection method ('ORB', 'FAST', 'BRISK', 'SIFT', 'SURF', 'GFTT')
            threshold (int, optional): Threshold parameter for certain keypoint detectors
            nonmaxSuppression (bool, optional): Whether to use non-maximum suppression
        """
        kwargs = {}
        if threshold is not None:
            kwargs["threshold"] = threshold
        if nonmaxSuppression is not None:
            kwargs["nonmaxSuppression"] = nonmaxSuppression

        self.stabilizer = VidStab(kp_method=kp_method, **kwargs)
        self._transforms = None
        self._trajectory = None

    def stabilize(
        self,
        input_path,
        output_path,
        smoothing_window=30,
        border_type="black",
        border_size="auto",
        layer_func=None,
        playback=False,
        max_frames=None,
        use_stored_transforms=False,
    ):
        """
        Stabilize a video and save the result to a file.

        Args:
            input_path (str or int): Path to input video file or device index for webcam
            output_path (str): Path to save the stabilized video
            smoothing_window (int): Window size for trajectory smoothing
            border_type (str): Border handling method ('black', 'reflect', 'replicate')
            border_size (int or 'auto'): Size of border or 'auto' for automatic sizing
            layer_func (callable): Function for frame layering effects
            playback (bool): Whether to display preview during processing
            max_frames (int): Maximum number of frames to process
            use_stored_transforms (bool): Use previously computed transformations

        Returns:
            bool: True if stabilization was successful
        """
        try:
            if use_stored_transforms and self._transforms is not None:
                self.stabilizer.transforms = self._transforms
                # Only pass params that are accepted by VidStab.apply_transforms
                self.stabilizer.apply_transforms(
                    input_path=input_path,
                    output_path=output_path,
                    border_type=border_type,
                    border_size=border_size,
                    layer_func=layer_func,
                    playback=playback,
                    # max_frames is not passed here as it's not accepted by apply_transforms
                )
            else:
                # Pass all params for stabilize method which may accept max_frames
                self.stabilizer.stabilize(
                    input_path=input_path,
                    output_path=output_path,
                    smoothing_window=smoothing_window,
                    border_type=border_type,
                    border_size=border_size,
                    layer_func=layer_func,
                    playback=playback,
                    max_frames=max_frames,
                )
                # Store transforms for later use
                self._transforms = (
                    self.stabilizer.transforms.copy()
                    if self.stabilizer.transforms is not None
                    else None
                )

            return True
        except Exception as e:
            import traceback

            print(f"Error during stabilization: {traceback.format_exc()}")
            return False

    def gen_transforms(self, input_path, smoothing_window=30, max_frames=None):
        """
        Generate transformation matrices without applying them.

        Args:
            input_path (str or int): Path to input video file or device index for webcam
            smoothing_window (int): Window size for trajectory smoothing
            max_frames (int): Maximum number of frames to process (Note: passed to read_frames, not directly to gen_transforms)

        Returns:
            bool: True if transformation generation was successful
        """
        try:
            # The VidStab.gen_transforms() doesn't accept max_frames directly
            # It's actually a parameter for read_frames method in VidStab
            if max_frames is not None:
                # Use the approach from the VidStab implementation
                # Read frames with max_frames limit
                self.stabilizer.cap = cv2.VideoCapture(input_path)
                if not self.stabilizer.cap.isOpened():
                    print(f"Error: Could not open video file {input_path}")
                    return False

                # Generate transforms on those frames
                self.stabilizer.transforms = []
                self.stabilizer.prev_gray = None

                frame_count = 0
                while True:
                    ret, frame = self.stabilizer.cap.read()
                    if not ret:
                        break

                    self.stabilizer._gen_next_transform(frame)
                    frame_count += 1

                    if max_frames is not None and frame_count >= max_frames:
                        break

                self.stabilizer.cap.release()
                self.stabilizer.transforms = np.array(self.stabilizer.transforms)

                # Apply smoothing
                if self.stabilizer.transforms.shape[0] > 0:
                    self.stabilizer._gen_smoothed_trajectory(smoothing_window)
                    self.stabilizer._gen_new_transforms()
            else:
                # If no max_frames is provided, use the standard gen_transforms method
                self.stabilizer.gen_transforms(
                    input_path=input_path, smoothing_window=smoothing_window
                )

            self._transforms = (
                self.stabilizer.transforms.copy()
                if self.stabilizer.transforms is not None
                else None
            )
            return True
        except Exception as e:
            import traceback

            print(f"Error in gen_transforms: {traceback.format_exc()}")
            return False

    def apply_transforms(
        self,
        input_path,
        output_path,
        border_type="black",
        border_size="auto",
        layer_func=None,
        playback=False,
        max_frames=None,
    ):
        """
        Apply previously generated transforms to a video.

        Args:
            input_path (str or int): Path to input video file or device index for webcam
            output_path (str): Path to save the stabilized video
            border_type (str): Border handling method ('black', 'reflect', 'replicate')
            border_size (int or 'auto'): Size of border or 'auto' for automatic sizing
            layer_func (callable): Function for frame layering effects
            playback (bool): Whether to display preview during processing
            max_frames (int): Maximum number of frames to process

        Returns:
            bool: True if applying transforms was successful
        """
        try:
            if self._transforms is None:
                print("No transforms available. Call gen_transforms() first.")
                return False

            self.stabilizer.transforms = self._transforms

            # Convert border_size to integer if it's a string number (not "auto")
            if isinstance(border_size, str) and border_size != "auto":
                try:
                    border_size = int(border_size)
                except ValueError:
                    print(f"Warning: Invalid border_size '{border_size}', using 'auto' instead")
                    border_size = "auto"

            # Simplified approach - directly use parameters that vidstab supports
            self.stabilizer.apply_transforms(
                input_path=input_path,
                output_path=output_path,
                border_type=border_type,
                border_size=border_size,
                layer_func=layer_func,
                playback=playback
            )
            return True
        except Exception as e:
            import traceback
            print(f"Error applying transforms: {traceback.format_exc()}")
            return False

    def save_transforms(self, output_path):
        """
        Save transforms to a CSV file.

        Args:
            output_path (str): Path to save the transforms

        Returns:
            bool: True if successfully saved
        """
        try:
            if self._transforms is None:
                print("No transforms available to save")
                return False

            np.savetxt(output_path, self._transforms, delimiter=",")
            return True
        except Exception as e:
            import traceback

            print(f"Error saving transforms: {traceback.format_exc()}")
            return False

    def load_transforms(self, input_path):
        """
        Load transforms from a CSV file.

        Args:
            input_path (str): Path to the transforms file

        Returns:
            bool: True if successfully loaded
        """
        try:
            self._transforms = np.loadtxt(input_path, delimiter=",")
            self.stabilizer.transforms = self._transforms
            return True
        except Exception as e:
            import traceback

            print(f"Error loading transforms: {traceback.format_exc()}")
            return False

    def plot_trajectory(self):
        """
        Plot the original and smoothed trajectory.

        Returns:
            matplotlib.figure.Figure: Trajectory plot
        """
        # The vidstab plot_trajectory returns a tuple of (fig, ax)
        # We need to return just the figure
        fig, ax = self.stabilizer.plot_trajectory()
        return fig

    def plot_transforms(self):
        """
        Plot the transforms.

        Returns:
            matplotlib.figure.Figure: Transforms plot
        """
        # The vidstab plot_transforms returns a tuple of (fig, ax)
        # We need to return just the figure
        fig, ax = self.stabilizer.plot_transforms()
        return fig

    def stabilize_frame(
        self, frame, smoothing_window=30, border_size=0, border_type="black"
    ):
        """
        Stabilize a single frame.

        Args:
            frame (np.ndarray): Input frame
            smoothing_window (int): Window size for trajectory smoothing
            border_size (int): Size of border
            border_type (str): Border handling method

        Returns:
            np.ndarray: Stabilized frame
        """
        return self.stabilizer.stabilize_frame(
            input_frame=frame,
            smoothing_window=smoothing_window,
            border_size=border_size,
            border_type=border_type,
        )


# Helper functions to provide layer options
def get_layer_overlay(image, background):
    """Layer overlay effect for vidstab"""
    return layer_overlay(image, background)


def get_layer_blend(image, background, alpha=0.5):
    """Layer blend effect for vidstab with configurable alpha"""
    return layer_blend(image, background, foreground_alpha=alpha)
