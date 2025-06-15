# Flam_repo
 
# 🔍 Seamlessly Integrating a Person into a Scene

## 🎯 Objective

The goal of this project was to insert a person into a background image in a way that looks photorealistic – meaning the lighting, shadows, colour, and overall blending should appear as natural as possible.

While some steps were provided as part of the assignment, several had to be expanded or adapted to create a convincing final result.

---

## 🧩 Overview of Steps Taken

### 📸 Task 1: Capturing and Preparing the Person's Image

**Step 1: Image Capture**  
Used OpenCV to access the system’s webcam and capture a high-resolution image of a person in a well-lit, frontal pose.

**Step 2: Background Removal**  
Used the `rembg` Python library (which uses U²-Net) to remove the background and save the person as a transparent PNG.

---

### 🌑 Task 2: Analyzing Shadows and Lighting of the Background

**Step 1: Shadow Detection**  
HSV thresholding + gradient analysis in OpenCV to create a binary shadow mask.

**Step 2: Shadow Classification**  
Classified shadows as soft or hard using edge sharpness via the Laplacian operator.

---

### 💡 Task 3: Determining Light Direction

**Outdoor Scenes:**  
Annotated base and shadow tip to calculate a light direction vector and angle using trigonometry.

**Indoor Scenes:**  
Used Sobel gradient filters to estimate the dominant lighting direction based on brightness falloff.

---

### 🎨 Task 4: Coloring and Blending

**Step 1: Advanced Color Transfer**  
Used LAB color statistics with partial luminance preservation to match person color tone to the background.

**Step 2: Local Color Adaptation**  
Sampled edge-adjacent background regions to blend those tones into the subject based on proximity (via distance transform).

**Step 3: Histogram Matching**  
Performed histogram matching on LAB a/b channels for hue/saturation alignment.

**Step 4: Lighting Correction**  
Used background gradients to adjust the person’s brightness in HSV space to match ambient light.

**Step 5: Seamless Blending**  
Applied multi-band Laplacian pyramid blending to eliminate harsh edges between person and background.

**Step 6: Alpha Blending**  
Applied a feathered Gaussian blur on the alpha mask for soft transitions.

---

### 🖼️ Task 5: Generating the Final Output

Simulated a soft shadow from the alpha mask using blur and directional offset, then placed the person into a full-scene background.

### ✅ Final Result:

![Final Composite](final_composite.png)

---

## 🛠️ Tools and Libraries Used

- Python 3.10  
- OpenCV  
- NumPy  
- Rembg (U²-Net)  
- Scikit-Image  
- Matplotlib  
- Streamlit (optional deployment)

---

## 🔍 Missing / Additional Steps Identified

- Colour harmonization beyond LAB mean/std matching  
- Edge-aware local color blending  
- Lighting adjustment based on brightness gradients  
- Shadow simulation using alpha mask and vector offset

---

## 💭 Reflection

This project involved more than just code — it was a creative problem-solving process where visual realism mattered. Matching color and light realistically was a major challenge. Iterative testing and layering techniques led to a final composite that closely resembles an authentic photograph.

---

