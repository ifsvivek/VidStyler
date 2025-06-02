# AI – powered Image and Video Editing: VidStyler

## Students
1. Vivek Sharma (1CD22CS187)
2. Satya Bonthala (1CD22CS137)
3. K Prathyusha (1CD22CS059)

**Batch No:** 38

## Title
**AI – powered Image and Video Editing**

VidStyler is an AI-powered image and video editing suite with Neural Style Transfer (NST), Video Stabilization, and Object Removal capabilities.

## Introduction
NST transfers artistic styles to photos by separating content and style in CNNs. Early methods were slow; later research focused on speed and handling diverse styles. Video Stabilization aims to remove unwanted camera motion, using traditional feature-based or modern deep learning approaches, facing challenges with different video types. Object Removal, often using inpainting, erases objects while maintaining visual coherence, with evaluation being a key challenge.

## Methodology Comparison
Traditional NST used non-photorealistic rendering. Gatys et al. used CNNs to separate content and style via Gram matrices. Faster methods trained feed-forward CNNs with perceptual losses. Arbitrary style transfer uses techniques like AdaIN. Recent trends involve diffusion models and multi-modal guidance. Traditional video stabilization uses feature point tracking and motion smoothing (2D, 2.5D, or 3D). Learning-based methods predict optical flow. Selective stabilization targets jittery segments. Object Removal uses traditional inpainting and deep learning methods like GANs and diffusion models.

## Implemented Methodology
VidStyler uses VGG16 for NST feature extraction, optimization-based style transfer with LBFGS, Gram matrices for style, adjustable style-content balance, and progress tracking. For Video Stabilization, it uses multiple keypoint detection methods (GFTT, SIFT, SURF, ORB, BRISK, FAST), customizable smoothing radius and border handling, layer effects for motion trails, and trajectory/transform visualization with frame-by-frame processing. Object Removal features interactive drawing, multiple inpainting methods (OpenCV and DeepFill), alpha channel support, real-time web interface feedback, and memory-efficient processing.

## Outcomes
VidStyler's NST should produce high-quality artistic stylizations but will likely be computationally intensive. Video stabilization should effectively reduce jitter, with performance depending on keypoint detection robustness and motion smoothing. Object Removal using both OpenCV and DeepFill offers flexibility, with DeepFill potentially providing more plausible results for complex backgrounds. Further development could explore faster NST techniques and learning-based video stabilization. Evaluating output quality is also crucial.

## Approvals
**Guide:** Dr. Yashaswini S  
**Guide Signature:** __________________ 
**Date:** ______________  
**Project Coordinator Signature:** __________________
