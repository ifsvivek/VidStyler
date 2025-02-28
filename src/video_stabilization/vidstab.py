import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VidStab:
    """
    Video stabilization class based on the work by Nghia Ho.
    
    This class implements a simple video stabilization algorithm using OpenCV.
    It tracks feature points across frames and calculates transformations
    to stabilize the video.
    """
    
    def __init__(self, kp_method='GFTT', **kp_kwargs):
        """
        Initialize VidStab with keypoint detection method and parameters
        
        Args:
            kp_method: String of OpenCV keypoint detection method (e.g. 'GFTT', 'FAST', 'ORB', 'SIFT')
            **kp_kwargs: Keyword arguments for keypoint detector
        """
        self.kp_method = kp_method
        self.kp_kwargs = kp_kwargs
        self.trajectory = None
        self.smoothed_trajectory = None
        self.transforms = None
        self._reset_bookkeeping_params()
    
    def _reset_bookkeeping_params(self):
        """Reset all trajectory and transform data"""
        self.prev_gray = None
        self.prev_kps = None
        self.prev_desc = None
        self.trajectory = []
        self.smoothed_trajectory = []
        self.transforms = []
        
    def _get_keypoint_detector(self):
        """
        Initialize the keypoint detector based on method name
        
        Returns:
            OpenCV keypoint detector object
        """
        if self.kp_method == 'GFTT':
            # Default params for GFTT (Good Features To Track)
            default_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
            params = {**default_params, **self.kp_kwargs}
            
            def detector(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray, **params)
                if corners is None:
                    return [], None
                return corners.reshape(-1, 2), None
            
            return detector
            
        elif self.kp_method == 'FAST':
            default_params = dict(threshold=20, nonmaxSuppression=True)
            params = {**default_params, **self.kp_kwargs}
            
            fast = cv2.FastFeatureDetector_create(**params)
            
            def detector(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kps = fast.detect(gray, None)
                kps = np.array([kp.pt for kp in kps])
                return kps, None
            
            return detector
            
        elif self.kp_method == 'ORB':
            default_params = dict(nfeatures=500)
            params = {**default_params, **self.kp_kwargs}
            
            orb = cv2.ORB_create(**params)
            
            def detector(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kps, desc = orb.detectAndCompute(gray, None)
                kps = np.array([kp.pt for kp in kps])
                return kps, desc
            
            return detector
            
        elif self.kp_method == 'SIFT':
            try:
                sift = cv2.SIFT_create(**self.kp_kwargs)
            except AttributeError:
                try:
                    sift = cv2.xfeatures2d.SIFT_create(**self.kp_kwargs)
                except AttributeError:
                    raise AttributeError("SIFT is not available in this OpenCV version")
            
            def detector(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kps, desc = sift.detectAndCompute(gray, None)
                kps = np.array([kp.pt for kp in kps])
                return kps, desc
            
            return detector
        
        else:
            raise ValueError(f"Keypoint detection method {self.kp_method} not supported")
    
    def _estimate_transform(self, frame):
        """
        Estimate transformation between current frame and previous frame
        
        Args:
            frame: Current frame
            
        Returns:
            transform: Estimated transformation matrix
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize detector if not already done
        detector = self._get_keypoint_detector()
        
        # On first frame
        if self.prev_gray is None:
            # Get keypoints and descriptors
            curr_kps, curr_desc = detector(frame)
            
            # Store for next frame
            self.prev_gray = curr_gray
            self.prev_kps = curr_kps
            self.prev_desc = curr_desc
            
            # Return identity transform for first frame
            return np.array([[1, 0, 0], [0, 1, 0]])
        
        # Get keypoints for current frame
        curr_kps, curr_desc = detector(frame)
        
        # GFTT doesn't provide descriptors, use optical flow instead
        if self.kp_method == 'GFTT' or self.kp_method == 'FAST':
            if len(self.prev_kps) > 0:
                # Calculate optical flow
                curr_kps, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, curr_gray, 
                    self.prev_kps.astype(np.float32), 
                    None
                )
                
                # Filter only valid points
                status = status.reshape(-1)
                prev_kps_valid = self.prev_kps[status == 1]
                curr_kps_valid = curr_kps[status == 1]
                
                if len(prev_kps_valid) >= 2:  # Need at least 2 points for transform
                    # Find transformation matrix
                    transform_matrix, _ = cv2.estimateAffinePartial2D(
                        prev_kps_valid, curr_kps_valid
                    )
                    
                    if transform_matrix is not None:
                        # Store for next frame
                        self.prev_gray = curr_gray
                        self.prev_kps = curr_kps_valid
                        
                        return transform_matrix
        
        # For ORB, SIFT which provide descriptors
        else:
            if len(self.prev_kps) > 0 and len(curr_kps) > 0:
                # Create BFMatcher object
                bf = cv2.BFMatcher(cv2.NORM_HAMMING if self.kp_method == 'ORB' else cv2.NORM_L2)
                matches = bf.knnMatch(self.prev_desc, curr_desc, k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) >= 4:  # Need at least 4 good matches
                    src_pts = np.float32([self.prev_kps[m.queryIdx] for m in good_matches])
                    dst_pts = np.float32([curr_kps[m.trainIdx] for m in good_matches])
                    
                    # Find transformation matrix
                    transform_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    
                    if transform_matrix is not None:
                        # Store for next frame
                        self.prev_gray = curr_gray
                        self.prev_kps = curr_kps
                        self.prev_desc = curr_desc
                        
                        return transform_matrix
        
        # If transform estimation fails, return identity transform
        logger.warning("Transform estimation failed, using identity transform")
        return np.array([[1, 0, 0], [0, 1, 0]])
        
    def gen_transforms(self, input_path, max_frames=None, show_progress=True):
        """
        Generate frame to frame transformations
        
        Args:
            input_path: Path to input video or camera device ID
            max_frames: Maximum number of frames to process
            show_progress: Whether to display a progress bar
            
        Returns:
            transforms: List of transformations
        """
        # Reset any existing trajectory and transform data
        self._reset_bookkeeping_params()
        
        # Open video
        if isinstance(input_path, (int, str)):
            cap = cv2.VideoCapture(input_path)
        else:
            raise ValueError("Input path must be a string or camera device ID")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Handle camera input which doesn't have frame_count
        if frame_count == 0:
            frame_count = max_frames if max_frames else float('inf')
        
        # Limit frames if max_frames specified
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)
        
        # Process frames
        pbar = tqdm(total=frame_count, disable=not show_progress)
        i = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret or i >= frame_count:
                break
                
            # Skip frames at beginning if they are corrupted
            if frame is None or frame.size == 0:
                pbar.update(1)
                i += 1
                continue
                
            # Get transform from previous to current frame
            transform = self._estimate_transform(frame)
            
            if transform is not None:
                # Extract translation and rotation
                dx = transform[0, 2]
                dy = transform[1, 2]
                da = np.arctan2(transform[1, 0], transform[0, 0])
                
                # Store the trajectory
                self.trajectory.append((dx, dy, da))
                
                # Store the transform
                self.transforms.append(transform)
            
            pbar.update(1)
            i += 1
        
        # Release video capture
        cap.release()
        pbar.close()
        
        # Convert trajectory to numpy array
        if self.trajectory:
            self.trajectory = np.array(self.trajectory)
        else:
            self.trajectory = np.empty((0, 3))
            
        logger.info(f"Generated transforms for {len(self.transforms)} frames")
        return self.transforms
        
    def smooth_trajectory(self, smoothing_radius=30):
        """
        Smooth the trajectory using a moving average filter
        
        Args:
            smoothing_radius: Radius of the moving average filter
            
        Returns:
            smoothed_trajectory: Smoothed trajectory
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            raise ValueError("No trajectory to smooth. Run gen_transforms first.")
        
        # Apply moving average filter to trajectory
        smoothed_trajectory = np.copy(self.trajectory)
        
        # Filter the trajectory using a moving average
        kernel = np.ones(2 * smoothing_radius + 1) / (2 * smoothing_radius + 1)
        for i in range(3):
            smoothed_trajectory[:, i] = np.convolve(
                self.trajectory[:, i], kernel, mode='same'
            )
            
            # Handle the borders
            smoothed_trajectory[:smoothing_radius, i] = smoothed_trajectory[smoothing_radius, i]
            smoothed_trajectory[-smoothing_radius:, i] = smoothed_trajectory[-smoothing_radius - 1, i]
        
        self.smoothed_trajectory = smoothed_trajectory
        return self.smoothed_trajectory
        
    def _get_optimal_border_size(self):
        """
        Calculate optimal border size based on transformations
        
        Returns:
            border_size: Optimal border size
        """
        if not hasattr(self, 'transforms_smooth') or self.transforms_smooth is None:
            raise ValueError("Need to compute smooth transforms first")
        
        # Get differences between smooth and original trajectory
        differences = self.smoothed_trajectory - self.trajectory
        
        # Calculate maximum translations
        max_dx = max(abs(differences[:, 0]))
        max_dy = max(abs(differences[:, 1]))
        
        # Convert to border size with a safety factor
        border_size = int(max(max_dx, max_dy) * 1.5)
        
        return border_size
        
    def apply_transforms(self, input_path, output_path, smoothing_radius=30, 
                        border_type='black', border_size=0, layer_func=None,
                        playback=False, max_frames=None, show_progress=True):
        """
        Apply stabilizing transforms to input video
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            smoothing_radius: Radius of smoothing window
            border_type: Type of border ('black', 'reflect', 'replicate')
            border_size: Size of border in pixels (0 for no border, 'auto' for automatic sizing)
            layer_func: Function for frame layering effects
            playback: Whether to show output while processing
            max_frames: Maximum number of frames to process
            show_progress: Whether to display a progress bar
            
        Returns:
            output_path: Path to stabilized video
        """
        # Generate transforms if not already done
        if self.transforms is None or len(self.transforms) == 0:
            self.gen_transforms(input_path, max_frames, show_progress)
        
        # Smooth trajectory
        self.smooth_trajectory(smoothing_radius)
        
        # Calculate new transforms based on smoothed trajectory
        differences = self.smoothed_trajectory - self.trajectory
        self.transforms_smooth = []
        
        for i, transform in enumerate(self.transforms):
            # Get smooth transform
            dx, dy, da = differences[i]
            
            # Reconstruct transformation matrix
            cos_da = np.cos(da)
            sin_da = np.sin(da)
            
            # Apply rotation and translation correction
            m_rot = np.array([[cos_da, -sin_da, dx],
                             [sin_da, cos_da, dy]])
            
            # Combine with original transform
            transform_smooth = m_rot @ np.vstack([transform, [0, 0, 1]])[:2]
            
            self.transforms_smooth.append(transform_smooth)
        
        # Handle automatic border sizing
        if border_size == 'auto':
            border_size = self._get_optimal_border_size()
            logger.info(f"Auto border size: {border_size}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Handle camera input which doesn't have frame_count
        if frame_count == 0:
            frame_count = max_frames if max_frames else float('inf')
        
        # Limit frames if max_frames specified
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)
        
        # Set up border handling
        border_modes = {
            'black': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE
        }
        border_mode = border_modes.get(border_type, cv2.BORDER_CONSTANT)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            output_path, fourcc, fps, 
            (width, height)
        )
        
        # Process frames
        pbar = tqdm(total=min(frame_count, len(self.transforms_smooth)), disable=not show_progress)
        i = 0
        last_frame = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret or i >= min(frame_count, len(self.transforms_smooth)):
                break
                
            # Skip frames at beginning if they are corrupted
            if frame is None or frame.size == 0:
                pbar.update(1)
                i += 1
                continue
            
            # Apply transform
            transform = self.transforms_smooth[i]
            
            # Apply transformation
            frame_stabilized = cv2.warpAffine(
                frame, transform, (width, height),
                flags=cv2.INTER_LINEAR, 
                borderMode=border_mode,
                borderValue=0 if border_type == 'black' else None
            )
            
            # Apply border
            if border_size > 0:
                frame_stabilized = frame_stabilized[
                    border_size:-border_size if border_size > 0 else None,
                    border_size:-border_size if border_size > 0 else None
                ]
                
                # Resize back to original size
                frame_stabilized = cv2.resize(frame_stabilized, (width, height))
            
            # Apply frame layering if specified
            if layer_func is not None and last_frame is not None:
                frame_stabilized = layer_func(frame_stabilized, last_frame)
            
            # Store frame for next iteration
            last_frame = frame_stabilized.copy()
            
            # Write frame to output video
            out.write(frame_stabilized)
            
            # Display output
            if playback:
                cv2.imshow('Stabilized', frame_stabilized)
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):  # ESC or q to exit
                    break
            
            pbar.update(1)
            i += 1
        
        # Release resources
        cap.release()
        out.release()
        if playback:
            cv2.destroyAllWindows()
        pbar.close()
        
        logger.info(f"Video stabilization completed: {output_path}")
        return output_path
        
    def stabilize(self, input_path, output_path, smoothing_radius=30, 
                 border_type='black', border_size=0, layer_func=None,
                 playback=False, max_frames=None, show_progress=True):
        """
        Convenience method that generates transforms and applies them in one step
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            smoothing_radius: Radius of smoothing window
            border_type: Type of border ('black', 'reflect', 'replicate')
            border_size: Size of border in pixels (0 for no border, 'auto' for automatic sizing)
            layer_func: Function for frame layering effects
            playback: Whether to show output while processing
            max_frames: Maximum number of frames to process
            show_progress: Whether to display a progress bar
            
        Returns:
            output_path: Path to stabilized video
        """
        self.gen_transforms(input_path, max_frames, show_progress)
        return self.apply_transforms(
            input_path, output_path, smoothing_radius,
            border_type, border_size, layer_func,
            playback, max_frames, show_progress
        )
    
    def stabilize_frame(self, input_frame, smoothing_window=30, border_type='black', border_size=0):
        """
        Stabilize a single frame for real-time processing
        
        Args:
            input_frame: Input frame as numpy array
            smoothing_window: Window size for smoothing
            border_type: Border handling method
            border_size: Size of border in pixels
            
        Returns:
            stabilized_frame: Stabilized frame
        """
        # If the input frame is None, we've reached the end
        if input_frame is None:
            return None
        
        # Get transform for this frame
        transform = self._estimate_transform(input_frame)
        
        if transform is None:
            return input_frame
        
        # Extract motion params
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])
        
        # Add to trajectory
        self.trajectory.append((dx, dy, da))
        
        # Wait until we have enough frames for smoothing
        if len(self.trajectory) < smoothing_window:
            return input_frame
        
        # Convert to numpy array
        self.trajectory_array = np.array(self.trajectory)
        
        # Smooth the trajectory with a moving average filter
        kernel = np.ones(smoothing_window) / smoothing_window
        
        # Filter the last 'smoothing_window' frames
        smoothed_x = np.convolve(self.trajectory_array[-smoothing_window:, 0], kernel, mode='valid')[-1]
        smoothed_y = np.convolve(self.trajectory_array[-smoothing_window:, 1], kernel, mode='valid')[-1]
        smoothed_a = np.convolve(self.trajectory_array[-smoothing_window:, 2], kernel, mode='valid')[-1]
        
        # Calculate the difference between smoothed and original position
        diff_x = smoothed_x - dx
        diff_y = smoothed_y - dy
        diff_a = smoothed_a - da
        
        # Calculate the new transformation matrix
        cos_a = np.cos(diff_a)
        sin_a = np.sin(diff_a)
        
        smooth_transform = np.array([
            [cos_a, -sin_a, diff_x],
            [sin_a, cos_a, diff_y]
        ])
        
        # Apply transformation
        h, w = input_frame.shape[:2]
        
        # Set up border handling
        border_modes = {
            'black': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE
        }
        border_mode = border_modes.get(border_type, cv2.BORDER_CONSTANT)
        
        # Apply the stabilizing transform
        stabilized_frame = cv2.warpAffine(
            input_frame, smooth_transform, (w, h),
            flags=cv2.INTER_LINEAR, 
            borderMode=border_mode,
            borderValue=0 if border_type == 'black' else None
        )
        
        # Apply border crop if specified
        if border_size > 0:
            stabilized_frame = stabilized_frame[
                border_size:-border_size if border_size > 0 else None,
                border_size:-border_size if border_size > 0 else None
            ]
            
            # Resize back to original size
            stabilized_frame = cv2.resize(stabilized_frame, (w, h))
        
        return stabilized_frame
        
    def plot_trajectory(self):
        """
        Plot the original and smoothed trajectory
        
        Returns:
            fig: Matplotlib figure
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            raise ValueError("No trajectory to plot. Run gen_transforms first.")
        
        if self.smoothed_trajectory is None or len(self.smoothed_trajectory) == 0:
            # Smooth trajectory if not already done
            self.smooth_trajectory()
        
        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot x trajectory
        ax1.plot(self.trajectory[:, 0], 'b-', label='Original')
        ax1.plot(self.smoothed_trajectory[:, 0], 'r-', label='Smoothed')
        ax1.set_title('X-Trajectory')
        ax1.set_ylabel('dx (pixels)')
        ax1.legend()
        ax1.grid()
        
        # Plot y trajectory
        ax2.plot(self.trajectory[:, 1], 'b-', label='Original')
        ax2.plot(self.smoothed_trajectory[:, 1], 'r-', label='Smoothed')
        ax2.set_title('Y-Trajectory')
        ax2.set_ylabel('dy (pixels)')
        ax2.legend()
        ax2.grid()
        
        # Plot angle trajectory
        ax3.plot(np.rad2deg(self.trajectory[:, 2]), 'b-', label='Original')
        ax3.plot(np.rad2deg(self.smoothed_trajectory[:, 2]), 'r-', label='Smoothed')
        ax3.set_title('Angle Trajectory')
        ax3.set_ylabel('da (degrees)')
        ax3.set_xlabel('Frame Number')
        ax3.legend()
        ax3.grid()
        
        plt.tight_layout()
        return fig
        
    def plot_transforms(self):
        """
        Plot the differences between original and smoothed transforms
        
        Returns:
            fig: Matplotlib figure
        """
        if self.trajectory is None or len(self.trajectory) == 0:
            raise ValueError("No trajectory to plot. Run gen_transforms first.")
        
        if self.smoothed_trajectory is None or len(self.smoothed_trajectory) == 0:
            # Smooth trajectory if not already done
            self.smooth_trajectory()
        
        # Calculate differences
        differences = self.smoothed_trajectory - self.trajectory
        
        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot x differences
        ax1.plot(differences[:, 0])
        ax1.set_title('X-Translation Differences')
        ax1.set_ylabel('dx (pixels)')
        ax1.grid()
        
        # Plot y differences
        ax2.plot(differences[:, 1])
        ax2.set_title('Y-Translation Differences')
        ax2.set_ylabel('dy (pixels)')
        ax2.grid()
        
        # Plot angle differences
        ax3.plot(np.rad2deg(differences[:, 2]))
        ax3.set_title('Angle Differences')
        ax3.set_ylabel('da (degrees)')
        ax3.set_xlabel('Frame Number')
        ax3.grid()
        
        plt.tight_layout()
        return fig
