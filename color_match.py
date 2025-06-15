import cv2
import numpy as np
from skimage import exposure, filters
from scipy import ndimage

class ColorMatcher:
    def __init__(self):
        self.blur_kernel_size = 15
        
    def extract_reference_colors(self, background, mask, sample_size=50):
        kernel = np.ones((sample_size, sample_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        edge_zone = dilated_mask - mask

        edge_pixels = background[edge_zone > 0]
        if len(edge_pixels) == 0:
            
            h, w = background.shape[:2]
            edge_pixels = background[mask == 0]
        
        return edge_pixels
    
    def advanced_color_transfer(self, source, target, mask=None, preserve_luminance=0.3):
        
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        
        if mask is not None:
            valid_source = source_lab[mask > 0]
            valid_target = target_lab
        else:
            valid_source = source_lab.reshape(-1, 3)
            valid_target = target_lab.reshape(-1, 3)
        
        source_stats = []
        target_stats = []
        
        for i in range(3):
            src_mean = np.mean(valid_source[:, i])
            src_std = np.std(valid_source[:, i])
            tgt_mean = np.mean(valid_target[:, i])
            tgt_std = np.std(valid_target[:, i])
            
            source_stats.append((src_mean, src_std))
            target_stats.append((tgt_mean, tgt_std))
        
        result_lab = source_lab.copy()
        
        for i in range(3):
            src_mean, src_std = source_stats[i]
            tgt_mean, tgt_std = target_stats[i]
            
            channel = result_lab[:, :, i]
             #DONT DEVIDE BY 0!!
            if src_std > 1e-6:
                channel = (channel - src_mean) * (tgt_std / src_std) + tgt_mean
            else:
                channel = channel + (tgt_mean - src_mean)
            
            
            if i == 0 and preserve_luminance > 0:
                channel = (1 - preserve_luminance) * channel + preserve_luminance * source_lab[:, :, i]
            
            result_lab[:, :, i] = channel
        

        result_lab = np.clip(result_lab, 0, 255)
        result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result_bgr
    
    def local_color_adaptation(self, person, background, mask, adaptation_strength=0.5):
    
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_dist = np.max(dist_transform)
        
        if max_dist > 0:
            
            weights = 1.0 - (dist_transform / max_dist)
            weights = np.clip(weights * adaptation_strength, 0, 1)
        else:
            weights = np.ones_like(mask, dtype=np.float32) * adaptation_strength
        
        reference_colors = self.extract_reference_colors(background, mask)
        if len(reference_colors) > 0:

            ref_color = np.median(reference_colors, axis=0)
            
            result = person.copy().astype(np.float32)
            for c in range(3):
                result[:, :, c] = (1 - weights) * result[:, :, c] + weights * ref_color[c]
            
            return np.clip(result, 0, 255).astype(np.uint8)
        
        return person
    
    def match_histogram_selective(self, source, reference, mask=None, channels='all'):
       
        if channels == 'all':
            channels = [0, 1, 2]
        elif isinstance(channels, str):
            channels = [0] if channels == 'luminance' else [1, 2]
        
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        
        matched_lab = source_lab.copy()
        
        for channel in channels:
            if mask is not None:
                source_channel = source_lab[:, :, channel][mask > 0]
            else:
                source_channel = source_lab[:, :, channel]
            
            reference_channel = reference_lab[:, :, channel]
            
            #match histo
            matched_channel = exposure.match_histograms(
                source_lab[:, :, channel], 
                reference_channel
            )
            
            matched_lab[:, :, channel] = matched_channel
        
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

def create_seamless_blend(person, background, mask, blend_mode='multiband'):
    
    if blend_mode == 'multiband':
        return multiband_blending(person, background, mask)
    elif blend_mode == 'poisson':
        return poisson_blending(person, background, mask)
    else:
        return alpha_blending(person, background, mask)

def multiband_blending(person, background, mask, levels=4):
   
    def build_gaussian_pyramid(img, levels):
        pyramid = [img.copy()]
        for i in range(levels):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        return pyramid
    
    def build_laplacian_pyramid(gaussian_pyramid):
        laplacian_pyramid = []
        for i in range(len(gaussian_pyramid) - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
            laplacian_pyramid.append(laplacian)
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid
   
    person_gaussian = build_gaussian_pyramid(person.astype(np.float32), levels)
    background_gaussian = build_gaussian_pyramid(background.astype(np.float32), levels)
    mask_gaussian = build_gaussian_pyramid(mask.astype(np.float32) / 255.0, levels)
    
    person_laplacian = build_laplacian_pyramid(person_gaussian)
    background_laplacian = build_laplacian_pyramid(background_gaussian)
    
    # blend at each level!!
    blended_laplacian = []
    for i in range(len(person_laplacian)):
        mask_level = mask_gaussian[i]
        if len(mask_level.shape) == 2:
            mask_level = np.stack([mask_level] * 3, axis=2)
        
        blended_level = person_laplacian[i] * mask_level + background_laplacian[i] * (1 - mask_level)
        blended_laplacian.append(blended_level)
    
    # reconstruct img!
    result = blended_laplacian[-1]
    for i in range(len(blended_laplacian) - 2, -1, -1):
        size = (blended_laplacian[i].shape[1], blended_laplacian[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result = cv2.add(result, blended_laplacian[i])
    
    return np.clip(result, 0, 255).astype(np.uint8)

def alpha_blending(person, background, mask, feather_size=5):
    
    mask_float = mask.astype(np.float32) / 255.0
    
    if feather_size > 0:
        mask_float = cv2.GaussianBlur(mask_float, (feather_size * 2 + 1, feather_size * 2 + 1), 0)
    
    if len(mask_float.shape) == 2:
        mask_float = np.stack([mask_float] * 3, axis=2)
    
    # blend
    result = person.astype(np.float32) * mask_float + background.astype(np.float32) * (1 - mask_float)
    return np.clip(result, 0, 255).astype(np.uint8)

def enhance_lighting_consistency(person, background, mask):
    gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
   
    grad_x = cv2.Sobel(gray_bg, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_bg, cv2.CV_64F, 0, 1, ksize=3)
    
    avg_grad_x = np.mean(grad_x)
    avg_grad_y = np.mean(grad_y)
    
    person_hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    lighting_factor = np.sqrt(avg_grad_x**2 + avg_grad_y**2) / 100.0
    lighting_factor = np.clip(lighting_factor, 0.8, 1.2)
    
    person_hsv[:, :, 2] *= lighting_factor
    person_hsv[:, :, 2] = np.clip(person_hsv[:, :, 2], 0, 255)
    
    return cv2.cvtColor(person_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def complete_color_matching_pipeline():

    try:
        fg_rgba = cv2.imread("person_no_bg.png", cv2.IMREAD_UNCHANGED)
        bg = cv2.imread("bg.jpg")
        
        if fg_rgba is None or bg is None:
            print("Error: Could not load input images")
            return
        
        if fg_rgba.shape[2] == 4:
            person_rgb = fg_rgba[:, :, :3]
            alpha = fg_rgba[:, :, 3]
        else:
            person_rgb = fg_rgba
            alpha = np.ones((fg_rgba.shape[0], fg_rgba.shape[1]), dtype=np.uint8) * 255
        
        mask = (alpha > 128).astype(np.uint8) * 255
        
        bg_resized = cv2.resize(bg, (person_rgb.shape[1], person_rgb.shape[0]))
        
        color_matcher = ColorMatcher()
   
        print("Step 1: applying advanced color transfer!!!!")
        color_matched = color_matcher.advanced_color_transfer(
            person_rgb, bg_resized, mask, preserve_luminance=0.2
        )
        
        print("step 2: applying local color adaptation!!")
        locally_adapted = color_matcher.local_color_adaptation(
            color_matched, bg_resized, mask, adaptation_strength=0.3
        )
        
        print("step 3: matching color histograms!!!")
        histogram_matched = color_matcher.match_histogram_selective(
            locally_adapted, bg_resized, mask, channels=[1, 2] 
        )
        
        print("Step 4: Adjusting lighting consistency...")
        lighting_adjusted = enhance_lighting_consistency(histogram_matched, bg_resized, mask)
        
      
        print("Step 5: Creating seamless blend...")
        final_result = create_seamless_blend(
            lighting_adjusted, bg_resized, mask, blend_mode='multiband'
        )
        
        print("Step 6: Final compositing...")
        composite = alpha_blending(lighting_adjusted, bg_resized, mask, feather_size=3)
        
        cv2.imwrite("step1_color_transfer.png", color_matched)
        cv2.imwrite("step2_local_adaptation.png", locally_adapted)
        cv2.imwrite("step3_histogram_matched.png", histogram_matched)
        cv2.imwrite("step4_lighting_adjusted.png", lighting_adjusted)
        cv2.imwrite("step5_seamless_blend.png", final_result)
        cv2.imwrite("final_composite.png", composite)
        
        print("Pipeline complete! Check the generated images.")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")

if __name__ == "__main__":
    complete_color_matching_pipeline()